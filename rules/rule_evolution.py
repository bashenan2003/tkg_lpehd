"""
自适应规则演化模块
====================
核心机制:
1. 基于当前TKG数据重新计算cp和C (公式8, 9)
2. 更新优先级评分 P = w1*cp + w2*C
3. 淘汰低价值规则: P < P_th 且 cp < C_th
4. 对保留的高优先级规则, 基于最新数据采样迭代更新
5. 输出演化后的规则集 Sr

参数
- P_th = 0.6 (最优优先级阈值)
- C_th = 0.6 (置信度次阈值, 与cp阈值一致)
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict

from ..config import (
    PRIORITY_THRESHOLD, CONFIDENCE_EVOLUTION_THRESHOLD,
    PRIORITY_W_CONFIDENCE, PRIORITY_W_COVERAGE,
    is_real_llm_enabled,
)
from ..data.dataset import TKGDataSet
from .rule_mining import TemporalRule, RulePriorityRanker


class AdaptiveRuleEvolution:
    """
    自适应规则演化器
    "规则优先级动态更新 + 低优先级规则淘汰"

    解决传统方法仅更新低置信度规则的局限性,
    实现适应效率和规则质量的双重优化。
    """

    def __init__(self, dataset: TKGDataSet):
        self.dataset = dataset
        self.priority_ranker = RulePriorityRanker(dataset)
        self.p_th = PRIORITY_THRESHOLD        # P_th = 0.6
        self.c_th = CONFIDENCE_EVOLUTION_THRESHOLD  # C_th = 0.6

        # 演化历史记录
        self.evolution_history: List[dict] = []
        self.eliminated_rules: List[TemporalRule] = []

    # --------------------------------------------------------
    # 重新评估规则
    # --------------------------------------------------------
    def re_evaluate_rules(self, rules: List[TemporalRule]) -> List[TemporalRule]:
        """
        基于当前TKG数据重新计算规则的置信度和有效性覆盖
        cp和C使用第4.3节公式重新计算
        """
        for rule in rules:
            # 重新计算置信度 (公式8)
            self.priority_ranker.compute_confidence(rule)

            # 重新计算有效性覆盖 (公式9)
            self.priority_ranker.compute_validity_coverage(rule)

            # 更新优先级评分 P = w1*cp + w2*C
            self.priority_ranker.compute_priority(rule)

        return rules

    # --------------------------------------------------------
    # 淘汰低价值规则
    # --------------------------------------------------------
    def eliminate_low_value_rules(self,
                                   rules: List[TemporalRule]) -> Tuple[List[TemporalRule], List[TemporalRule]]:
        """
        淘汰低价值规则
        当 P < P_th 且 cp < C_th 时淘汰

        避免: 低质量规则消耗资源, 低价值规则积累

        Returns:
            (retained_rules, eliminated_rules)
        """
        retained = []
        eliminated = []

        for rule in rules:
            if rule.priority_score < self.p_th and rule.confidence < self.c_th:
                eliminated.append(rule)
            else:
                retained.append(rule)

        self.eliminated_rules.extend(eliminated)

        # 记录演化历史
        self.evolution_history.append({
            "step": len(self.evolution_history) + 1,
            "total_rules": len(rules),
            "retained": len(retained),
            "eliminated": len(eliminated),
            "elimination_rate": len(eliminated) / max(1, len(rules)),
        })

        return retained, eliminated

    # --------------------------------------------------------
    # 迭代更新高优先级规则
    # --------------------------------------------------------
    def iteratively_update_rules(self,
                                  retained_rules: List[TemporalRule],
                                  current_data: List) -> List[TemporalRule]:
        """
        对保留的高优先级规则进行迭代更新
        基于当前数据额外路径采样, 迭代更新规则

        主要更新: 微调有效期, 重新计算优先级
        """
        updated_rules = []

        for rule in retained_rules:
            # 基于当前数据重新计算有效期
            current_times = [q.time_val for q in current_data] if current_data else []

            if rule.source_path:
                path_span = (max(q.time_val for q in rule.source_path) -
                             min(q.time_val for q in rule.source_path))
            else:
                path_span = 30.0

            if current_times:
                current_span = max(current_times) - min(current_times)
                # 动态调整有效期
                rule.validity_period = (rule.validity_period + current_span) / 2

            # 确保有效期不低于路径跨度
            rule.validity_period = max(rule.validity_period, path_span * 1.2)

            # 重新计算优先级
            self.priority_ranker.compute_priority(rule)

            updated_rules.append(rule)

        return updated_rules

    # --------------------------------------------------------
    # 完整演化流程
    # --------------------------------------------------------
    def evolve(self,
               rules: List[TemporalRule],
               current_quads: List = None) -> List[TemporalRule]:
        """
        完整规则演化流程 (LLM 引导 + 算法回退)

        1. 重新评估规则统计
        2. LLM 语义判断保留/淘汰 + 调整有效期
        3. API 失败时回退到算术演化
        4. 按优先级排序输出
        """
        if current_quads is None:
            current_quads = self.dataset.test_quads or self.dataset.quadruples[-1000:]

        print(f"\n[规则演化] - Adaptive Rule Evolution")
        print(f"  输入规则数: {len(rules)}")
        print(f"  阈值: P_th={self.p_th}, C_th={self.c_th}")

        # Step 1: 重新评估
        rules = self.re_evaluate_rules(rules)
        print(f"  重新评估完成")

        # Step 2: LLM 语义演化 (或回退)
        if is_real_llm_enabled() and len(rules) > 0:
            evolved = self._llm_evolve_rules(rules, current_quads)
            if evolved is not None:
                evolved.sort(key=lambda r: r.priority_score, reverse=True)
                if evolved:
                    top_ps = [r.priority_score for r in evolved[:5]]
                    print(f"  [LLM] 演化后规则集Sr: {len(evolved)}条, "
                          f"最高P={max(top_ps):.3f}, 最低P={min(top_ps):.3f}")
                return evolved
            # LLM 失败 → 回退

        # Step 3: 算法演化 (回退)
        retained, eliminated = self.eliminate_low_value_rules(rules)
        print(f"  保留: {len(retained)}, 淘汰: {len(eliminated)}")
        evolved = self.iteratively_update_rules(retained, current_quads)
        print(f"  迭代更新完成")
        evolved.sort(key=lambda r: r.priority_score, reverse=True)

        if evolved:
            top_ps = [r.priority_score for r in evolved[:5]]
            print(f"  演化后规则集Sr: {len(evolved)}条, "
                  f"最高P={max(top_ps):.3f}, 最低P={min(top_ps):.3f}")

        return evolved

    def _llm_evolve_rules(self, rules, current_quads):
        """调用 DeepSeek 评估规则并指导演化。失败返回 None。"""
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        # 构建规则摘要
        rule_lines = []
        for i, r in enumerate(rules):
            rule_lines.append(
                f"  R{i}: {r.head_relation}←{'∧'.join(r.body_relations[:3])} "
                f"cp={r.confidence:.3f} P={r.priority_score:.3f} V={r.validity_period:.0f}d"
            )
        rules_text = "\n".join(rule_lines[:20])  # 最多 20 条

        # 数据摘要
        if current_quads:
            times = [q.time_val for q in current_quads]
            data_span = max(times) - min(times) if times else 0
        else:
            data_span = 30.0

        system = (
            "You are an expert in temporal knowledge graph rule management. "
            "Evaluate each rule and decide whether to keep or eliminate it. "
            "For kept rules, suggest an adjusted validity period (days). "
            "Rules with low confidence AND low priority should be eliminated. "
            "Borderline rules with unique body patterns may be kept. "
            "Return ONLY a JSON object: "
            '{"keep": [0, 2, 5], "adjust_V": {"0": 45, "2": 60, "5": 30}}. '
            "No explanation."
        )

        user = (
            f"Current data time span: {data_span:.0f} days\n"
            f"Elimination thresholds: P_th={self.p_th}, cp_th={self.c_th}\n\n"
            f"Rules to evaluate:\n{rules_text}\n\n"
            f"Decide which rules to keep and what validity period each should have."
        )

        result = llm_chat(system, user, max_tokens=512, temperature=0.0)
        if result is None:
            return None

        import json, re
        try:
            data = json.loads(result)
            keep_indices = set(data.get("keep", []))
            adjust_v = data.get("adjust_V", {})
        except json.JSONDecodeError:
            # 宽松解析
            keep_match = re.search(r'"keep"\s*:\s*\[([^\]]+)\]', result)
            if keep_match:
                keep_indices = set(int(n) for n in re.findall(r'\d+', keep_match.group(1)))
            else:
                return None
            adjust_v = {}

        # 应用 LLM 决策
        evolved = []
        for i, rule in enumerate(rules):
            if i not in keep_indices:
                self.eliminated_rules.append(rule)
                continue
            # 调整有效期
            new_v = adjust_v.get(str(i), adjust_v.get(i))
            if new_v is not None:
                rule.validity_period = float(new_v)
            else:
                # 保留原有有效期, 但结合数据跨度微调
                if current_quads:
                    times = [q.time_val for q in current_quads[:100]]
                    if times:
                        rule.validity_period = max(
                            rule.validity_period * 0.8,
                            min(rule.validity_period * 1.5,
                                (rule.validity_period + (max(times) - min(times))) / 2)
                        )
            # 重新计算优先级
            self.priority_ranker.compute_priority(rule)
            evolved.append(rule)

        self.evolution_history.append({
            "step": len(self.evolution_history) + 1,
            "total_rules": len(rules),
            "retained": len(evolved),
            "eliminated": len(rules) - len(evolved),
            "elimination_rate": (len(rules) - len(evolved)) / max(1, len(rules)),
            "mode": "LLM",
        })

        if evolved:
            print(f"  [LLM] 保留: {len(evolved)}, 淘汰: {len(rules) - len(evolved)}")
        return evolved

    # --------------------------------------------------------
    # 获取演化摘要
    # --------------------------------------------------------
    def get_evolution_summary(self) -> dict:
        """获取演化历史摘要"""
        if not self.evolution_history:
            return {"message": "无演化记录"}
        last = self.evolution_history[-1]
        return {
            "total_evolution_steps": len(self.evolution_history),
            "last_step": last,
            "total_eliminated": len(self.eliminated_rules),
            "current_elimination_rate": last["elimination_rate"],
        }

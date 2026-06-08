"""
规则优先级排序模块
====================
核心算法:
1. 公式(8): cp = Hb / B - 规则置信度
   - B: 匹配规则体且在有效期内的总事实数
   - Hb: 成功推导出规则头的事实数

2. 公式(9): C = B_valid / B_total - 有效性覆盖
   - B_valid: 有效期内的规则体事实数
   - B_total: 规则体总事实数

3. 优先级评分: P = w1*cp + w2*C
   - w1=0.7 (置信度权重更高, 确保可靠性)
   - w2=0.3
   - 置信度阈值: cp ≥ 0.6

4. 按P降序排列, 高优先级规则优先用于推理
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

from ..config import (
    CONFIDENCE_THRESHOLD, PRIORITY_W_CONFIDENCE, PRIORITY_W_COVERAGE,
    is_real_llm_enabled,
)
from ..data.dataset import TKGDataSet, Quadruple


class TemporalRule:
    """
    时序逻辑规则
    公式(1): 规则 ∧(esi, ri, eoi, ti) → r(es, eo, tr)
    """

    def __init__(self,
                 head_relation: str,
                 body_relations: List[str],
                 confidence: float = 0.0,
                 validity_period: float = 30.0,
                 source_path: List[Quadruple] = None,
                 time_constraints: List[Tuple[float, float]] = None):
        self.head_relation = head_relation          # 规则头关系
        self.body_relations = body_relations        # 规则体关系列表
        self.confidence = confidence                # 置信度 cp (公式8)
        self.validity_period = validity_period      # 有效期 V
        self.validity_coverage = 0.0                # 有效性覆盖 C (公式9)
        self.priority_score = 0.0                   # 优先级 P
        self.source_path = source_path or []        # 来源路径
        self.time_constraints = time_constraints or []
        self.support_count = 0                      # Hb
        self.body_match_count = 0                   # B
        self.body_valid_count = 0                   # B_valid
        self.body_total_count = 0                   # B_total

    def signature(self) -> str:
        """生成唯一签名 (用于去重)"""
        body_sig = "_".join(self.body_relations)
        return f"{self.head_relation}<-{body_sig}"

    def __repr__(self):
        body_str = " ∧ ".join(
            f"(x{i}, {r}, x{i+1})" for i, r in enumerate(self.body_relations)
        )
        return (f"Rule({self.head_relation}): {body_str} → (x0, {self.head_relation}, xn) "
                f"[cp={self.confidence:.3f}, V={self.validity_period:.0f}d, "
                f"P={self.priority_score:.3f}]")

    def to_dict(self) -> dict:
        return {
            "head_relation": self.head_relation,
            "body_relations": self.body_relations,
            "confidence": self.confidence,
            "validity_period": self.validity_period,
            "validity_coverage": self.validity_coverage,
            "priority_score": self.priority_score,
            "support_count": self.support_count,
            "signature": self.signature(),
        }


class RulePriorityRanker:
    """
    规则优先级排序器
    第4.3节: 基于置信度和有效性覆盖的优先级评分系统
    """

    def __init__(self, dataset: TKGDataSet):
        self.dataset = dataset
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.w1 = PRIORITY_W_CONFIDENCE  # 置信度权重
        self.w2 = PRIORITY_W_COVERAGE    # 覆盖权重

    # --------------------------------------------------------
    # 公式(8): cp = Hb / B
    # --------------------------------------------------------
    def compute_confidence(self, rule: TemporalRule,
                           validity_period: float = None) -> float:
        """
        计算规则置信度
        公式(8): cp = Hb / B

        B: 匹配规则体且在有效期内的四元组总数
        Hb: 可以成功推导规则头的四元组数

        遍历TKG数据, 查找匹配规则体的路径,
        验证是否满足时间约束和规则头预测
        """
        if validity_period is None:
            validity_period = rule.validity_period

        # 使用滑动时间窗口在数据中匹配规则
        all_quads = sorted(self.dataset.quadruples, key=lambda q: q.time_val)

        # 按实体对分组
        entity_pair_groups = defaultdict(list)
        for q in all_quads:
            entity_pair_groups[(q.subject, q.object)].append(q)

        body_match_count = 0  # B
        head_match_count = 0  # Hb

        # 为每个实体对检查规则匹配
        for (s, o), quads in entity_pair_groups.items():
            quads_sorted = sorted(quads, key=lambda q: q.time_val)

            # 滑动窗口匹配规则体
            for i in range(len(quads_sorted)):
                match = self._match_rule_body(rule, quads_sorted[i:], validity_period)
                if match["body_matched"]:
                    body_match_count += 1
                    if match["head_matched"]:
                        head_match_count += 1
                        break  # 每个实体对只计数一次

        rule.body_match_count = body_match_count
        rule.support_count = head_match_count

        if body_match_count > 0:
            rule.confidence = head_match_count / body_match_count
        else:
            # 严格匹配失败时使用启发式置信度 (用于合成/稀疏数据)
            heuristic = self._heuristic_confidence(rule, validity_period)
            rule.confidence = heuristic

        return rule.confidence

    def _heuristic_confidence(self, rule: TemporalRule,
                               validity_period: float) -> float:
        """
        LLM + 统计混合置信度估计 (严格匹配不可行时)

        真实模式: 将规则体/头的关系频率、时间重叠等统计信息发送给
        DeepSeek, 让 LLM 结合语义判断规则置信度。
        API 失败时回退到纯统计启发式。
        """
        all_quads = self.dataset.quadruples

        # 收集统计特征 (供 LLM 和 fallback 共享)
        body_stats = {}
        for body_rel in rule.body_relations:
            rel_quads = [q for q in all_quads if q.relation == body_rel]
            if rel_quads:
                body_stats[body_rel] = {
                    "freq": len(rel_quads) / max(1, len(all_quads)),
                    "count": len(rel_quads),
                }
            else:
                body_stats[body_rel] = {"freq": 0.0, "count": 0}

        head_quads = [q for q in all_quads if q.relation == rule.head_relation]

        # ---- 真实 LLM 调用 ----
        if is_real_llm_enabled():
            llm_conf = self._llm_estimate_confidence(
                rule, body_stats, head_quads, validity_period
            )
            if llm_conf is not None:
                return llm_conf

        # ---- 模拟 / 回退 ----
        body_scores = []
        for body_rel in rule.body_relations:
            bs = body_stats[body_rel]
            freq_score = bs["freq"]

            if head_quads:
                head_times = [q.time_val for q in head_quads]
                rel_quads = [q for q in all_quads if q.relation == body_rel]
                body_times = [q.time_val for q in rel_quads]
                head_span = (min(head_times), max(head_times))
                body_span = (min(body_times), max(body_times))
                overlap = max(0, min(head_span[1], body_span[1]) -
                              max(head_span[0], body_span[0]))
                total_span = max(head_span[1], body_span[1]) - min(head_span[0], body_span[0])
                time_score = overlap / max(1, total_span)
            else:
                time_score = 0.1

            body_scores.append(0.5 * freq_score * 100 + 0.5 * time_score)

        if body_scores:
            avg_score = sum(body_scores) / len(body_scores)
            confidence = 0.3 + 0.4 * min(1.0, avg_score)
        else:
            confidence = 0.3

        return confidence

    def _llm_estimate_confidence(self, rule, body_stats, head_quads,
                                  validity_period):
        """调用 DeepSeek 评估规则置信度。失败返回 None。"""
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        body_desc = ", ".join(
            f"\"{r}\" (freq={s['freq']:.4f}, count={s['count']})"
            for r, s in body_stats.items()
        )
        head_count = len(head_quads)
        head_freq = head_count / max(1, len(self.dataset.quadruples))
        head_desc = f"\"{rule.head_relation}\" (freq={head_freq:.4f}, count={head_count})"

        system = (
            "You are an expert in temporal knowledge graph rule mining. "
            "Evaluate the confidence of a temporal rule based on statistical "
            "features of its body and head relations in the dataset. "
            "Higher frequency of body relations and strong temporal overlap "
            "with the head relation suggest higher confidence. "
            "Reply with ONLY a single float between 0.0 and 1.0. No explanation."
        )

        user = (
            f"Rule: {rule.head_relation} ← "
            f"{' ∧ '.join(rule.body_relations)}\n"
            f"Validity period: {validity_period:.0f} days\n\n"
            f"Head relation stats: {head_desc}\n"
            f"Body relations stats: {body_desc}\n\n"
            f"Based on these statistics, estimate the rule's confidence (0-1)."
        )

        result = llm_chat(system, user, max_tokens=64, temperature=0.0)
        if result is None:
            return None

        import re
        nums = re.findall(r'[\d.]+', result)
        if nums:
            return max(0.0, min(1.0, float(nums[0])))
        return None

    def _match_rule_body(self, rule: TemporalRule,
                          quads: List[Quadruple],
                          validity_period: float) -> dict:
        """
        检查四元组序列是否匹配规则体 + 规则头

        支持两种匹配模式:
        1. 严格顺序匹配: 连续四元组精确匹配规则体关系序列
        2. 宽松顺序匹配: 在时间窗口内按顺序找到所有规则体关系

        Returns:
            {"body_matched": bool, "head_matched": bool}
        """
        result = {"body_matched": False, "head_matched": False}

        if len(quads) < len(rule.body_relations):
            return result

        body_len = len(rule.body_relations)

        # ---- 模式1: 严格连续匹配 ----
        if len(quads) >= body_len + 1:
            matched_body = True
            for j in range(body_len):
                if quads[j].relation != rule.body_relations[j]:
                    matched_body = False
                    break

            if matched_body:
                times = [q.time_val for q in quads[:body_len + 1]]
                times_in_order = all(
                    times[k] <= times[k + 1] for k in range(len(times) - 1)
                )
                if times_in_order:
                    time_span = max(times) - min(times)
                    if time_span <= validity_period:
                        result["body_matched"] = True
                        if len(quads) > body_len:
                            if quads[body_len].relation == rule.head_relation:
                                result["head_matched"] = True
                        return result

        # ---- 模式2: 宽松顺序匹配 (在时间窗口内按序找到) ----
        body_idx = 0
        matched_times = []
        for q in quads:
            if body_idx >= body_len:
                break
            if q.relation == rule.body_relations[body_idx]:
                matched_times.append(q.time_val)
                body_idx += 1

        if body_idx == body_len:
            # 检查时间约束
            if len(matched_times) >= 2:
                times_in_order = all(
                    matched_times[k] <= matched_times[k + 1]
                    for k in range(len(matched_times) - 1)
                )
                time_span = max(matched_times) - min(matched_times)
            else:
                times_in_order = True
                time_span = 0

            if times_in_order and time_span <= validity_period:
                result["body_matched"] = True

                # 检查规则头匹配
                for q in quads:
                    if q.relation == rule.head_relation:
                        result["head_matched"] = True
                        break

        return result

    # --------------------------------------------------------
    # 公式(9): C = B_valid / B_total
    # --------------------------------------------------------
    def compute_validity_coverage(self, rule: TemporalRule) -> float:
        """
        计算有效性覆盖
        公式(9): C = B_valid / B_total

        B_valid: 有效期内匹配规则体的四元组数
        B_total: 匹配规则体的总四元组数
        """
        all_quads = sorted(self.dataset.quadruples, key=lambda q: q.time_val)

        body_total = 0  # B_total
        body_valid = 0  # B_valid

        for i in range(len(all_quads) - len(rule.body_relations)):
            window = all_quads[i:i + len(rule.body_relations)]

            # 检查关系序列匹配
            rels_match = all(
                window[j].relation == rule.body_relations[j]
                for j in range(len(rule.body_relations))
            )

            if rels_match:
                body_total += 1
                times = [q.time_val for q in window]
                time_span = max(times) - min(times)
                if time_span <= rule.validity_period:
                    body_valid += 1

        rule.body_total_count = body_total
        rule.body_valid_count = body_valid

        if body_total > 0:
            rule.validity_coverage = body_valid / body_total
        else:
            rule.validity_coverage = 0.0

        return rule.validity_coverage

    # --------------------------------------------------------
    # 优先级评分: P = w1*cp + w2*C
    # --------------------------------------------------------
    def compute_priority(self, rule: TemporalRule) -> float:
        """
        计算规则优先级
        第4.3节: P = w1*cp + w2*C

        w1=0.7 (置信度优先, 确保推理可靠性)
        w2=0.3 (有效性覆盖)
        """
        # 确保置信度和覆盖已被计算
        if rule.confidence == 0.0 and rule.body_match_count == 0:
            self.compute_confidence(rule)
        if rule.validity_coverage == 0.0 and rule.body_total_count == 0:
            self.compute_validity_coverage(rule)

        rule.priority_score = (
            self.w1 * rule.confidence +
            self.w2 * rule.validity_coverage
        )
        return rule.priority_score

    # --------------------------------------------------------
    # 高置信度过滤 + 优先级排序
    # --------------------------------------------------------
    def filter_and_rank(self,
                        rules: List[TemporalRule],
                        min_confidence: float = None) -> List[TemporalRule]:
        """
        过滤低置信度规则并按优先级排序

        第4.3节:
        1. 过滤: 保留 cp ≥ 0.6 的规则
        2. 排序: 按P = w1*cp + w2*C 降序
        """
        if min_confidence is None:
            min_confidence = self.confidence_threshold

        # Step 1: 计算所有规则的置信度和优先级
        for rule in rules:
            if rule.confidence == 0.0:
                self.compute_confidence(rule)
            if rule.validity_coverage == 0.0:
                self.compute_validity_coverage(rule)
            self.compute_priority(rule)

        # Step 2: 高置信度过滤 (cp ≥ 0.6)
        filtered = [r for r in rules if r.confidence >= min_confidence]

        # Step 3: 按P降序排列
        filtered.sort(key=lambda r: r.priority_score, reverse=True)

        return filtered

    # --------------------------------------------------------
    # 获取高优先级规则集 Sr' (公式10)
    # --------------------------------------------------------
    def get_high_priority_rules(self, rules: List[TemporalRule],
                                 priority_threshold: float = 0.5) -> List[TemporalRule]:
        """
        获取高优先级规则集
        公式(10): Sr' = {ρ | ρ ∈ Sr ∧ score(ρ) ≥ θ}

        Args:
            rules: 规则集 Sr
            priority_threshold: 优先级阈值 θ

        Returns:
            List[TemporalRule]: 高优先级规则集 Sr'
        """
        # 确保滤除低置信度规则
        qualified = self.filter_and_rank(rules, self.confidence_threshold)
        return [r for r in qualified if r.priority_score >= priority_threshold]

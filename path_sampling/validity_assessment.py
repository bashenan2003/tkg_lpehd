"""
LLM辅助时序有效性评估模块
============================

核心算法:
1. 公式(7): Δti = t_target - t_final (路径时间差)
2. 非结构化语义样本集 T={t1, t2, ..., ti}
3. LLM结合Δti和T确定有效期V
4. 约束: max(Δti) ≤ V, 符合最严格的文本规定

目的: 克服传统方法仅依赖路径时间差的局限,
通过LLM语义理解准确判断规则的时间有效范围
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from ..config import is_real_llm_enabled, get_ablation_mode
from ..data.dataset import Quadruple


class TemporalValidityAssessor:
    """
    时序有效性评估器
    第4.2节: LLM辅助确定动态有效期
    """

    def __init__(self, use_simulated_llm: bool = None):
        if use_simulated_llm is None:
            self.use_simulated_llm = not is_real_llm_enabled()
        else:
            self.use_simulated_llm = use_simulated_llm

        # 不同关系类型的默认有效期 (天)
        self.default_validity_periods = {
            "Make_statement": 14,
            "Make_a_visit": 30,
            "Consult": 21,
            "Make_an_appeal_or_request": 14,
            "Express_intent_to_cooperate": 30,
            "Engage_in_diplomatic_cooperation": 60,
            "Host_a_visit": 30,
            "Accuse": 21,
            "Criticize_or_denounce": 14,
            "Threaten": 30,
            "Protest_or_demonstrate": 7,
            "Fight_with_artillery_and_tanks": 7,
            "Use_conventional_military_force": 14,
            "Provide_aid": 60,
            "Provide_military_aid": 90,
            "Provide_economic_aid": 90,
            "Express_intent_to_provide_aid": 30,
            "Cooperate_on_economic_issues": 90,
            "Engage_in_negotiation": 45,
            "Mediate": 60,
            "Sign_formal_agreement": 180,
            "Ratify_treaty": 365,
            "Apologize": 14,
            "Praise_or_endorse": 30,
            "Meet_with": 7,
            "Discuss_by_telephone": 3,
        }

    # --------------------------------------------------------
    # 公式(7): Δti = t_target - t_final
    # --------------------------------------------------------
    @staticmethod
    def compute_path_time_diff(target_time: float,
                                final_event_time: float) -> float:
        """
        计算路径时间差
        公式(7): Δti = t_target - t_final

        返回目标事件与路径最后一个事件之间的时间间隔
        """
        return abs(target_time - final_event_time)

    def compute_all_path_time_diffs(self, target_time: float,
                                     paths: List[List[Quadruple]]) -> list:
        """
        为所有路径计算时间差
        """
        diffs = []
        for path in paths:
            if path:
                final_time = path[-1].time_val
                diff = self.compute_path_time_diff(target_time, final_time)
                diffs.append(diff)
        return diffs

    # --------------------------------------------------------
    # 构建语义样本集 T
    # --------------------------------------------------------
    def build_semantic_sample_set(self,
                                   paths: List[List[Quadruple]],
                                   target_relation: str) -> Dict:
        """
        提取路径的非结构化文本信息, 构建语义样本集T
        第4.2节: T = {t1, t2, ..., ti}

        每条路径的文本描述包含:
        - 路径中每个事件的实体-关系-时间信息
        - 路径的时序结构
        - 目标关系类型
        """
        sample_set = {
            "target_relation": target_relation,
            "num_paths": len(paths),
            "path_descriptions": [],
            "path_time_ranges": [],
            "path_entities": [],
        }

        for i, path in enumerate(paths):
            # 格式化路径文本
            events_text = []
            for j, quad in enumerate(path):
                events_text.append(
                    f"({quad.subject}, {quad.relation}, {quad.object}, t={quad.time_val:.0f})"
                )

            desc = f"Path_{i}: " + " → ".join(events_text)
            sample_set["path_descriptions"].append(desc)

            # 时间范围
            if path:
                times = [q.time_val for q in path]
                sample_set["path_time_ranges"].append({
                    "min": min(times),
                    "max": max(times),
                    "span": max(times) - min(times) if len(times) > 1 else 0,
                })

            # 涉及的实体
            entities = set()
            for q in path:
                entities.add(q.subject)
                entities.add(q.object)
            sample_set["path_entities"].append(list(entities))

        return sample_set

    # --------------------------------------------------------
    # LLM确定有效期 V
    # --------------------------------------------------------
    def determine_validity_period(self,
                                   target_time: float,
                                   paths: List[List[Quadruple]],
                                   target_relation: str) -> float:
        """
        LLM确定规则有效期V
        第4.2节: LLM结合Δti和文本语义T输出V
        """
        if not paths:
            return self.default_validity_periods.get(target_relation, 30.0)

        time_diffs = self.compute_all_path_time_diffs(target_time, paths)
        max_diff = max(time_diffs) if time_diffs else 30.0
        default_v = self.default_validity_periods.get(target_relation, 30.0)

        # ---- 真实 LLM 调用 ----
        if not self.use_simulated_llm and get_ablation_mode() != "w/o T":
            v = self._llm_determine_validity(
                target_relation, paths, max_diff, default_v
            )
            if v is not None:
                return float(v)
            # API 失败 / 消融 → 回退到启发式

        # ---- 模拟 / 回退 ----
        time_spans = [p[-1].time_val - p[0].time_val for p in paths if len(p) >= 2]
        mean_span = np.mean(time_spans) if time_spans else max_diff
        path_count_factor = min(1.0, len(paths) / 10.0)

        long_term_relations = [
            "Sign_formal_agreement", "Ratify_treaty",
            "Cooperate_on_economic_issues", "Provide_economic_aid",
            "Provide_military_aid", "Engage_in_diplomatic_cooperation",
        ]
        short_term_relations = [
            "Meet_with", "Discuss_by_telephone", "Make_statement",
            "Apologize", "Criticize_or_denounce",
        ]
        if target_relation in long_term_relations:
            type_factor = 1.5
        elif target_relation in short_term_relations:
            type_factor = 0.5
        else:
            type_factor = 1.0

        raw_v = mean_span * type_factor * (1.0 + 0.5 * path_count_factor)
        validity_period = max(max_diff * 1.2, raw_v)
        validity_period = max(default_v * 0.5, min(validity_period, default_v * 3.0))
        return float(validity_period)

    def _llm_determine_validity(self, target_relation, paths, max_diff, default_v):
        """调用 DeepSeek 确定有效期 V。失败返回 None。"""
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        # 构建路径描述
        path_lines = []
        for i, path in enumerate(paths[:8]):  # 最多8条防止 token 溢出
            events = ", ".join(
                f"({q.subject},{q.relation},{q.object},t={q.time_val:.0f})"
                for q in path[:4]
            )
            path_lines.append(f"  Path {i+1}: {events}")
        path_text = "\n".join(path_lines) if path_lines else "(none)"

        system = (
            "You are an expert in temporal knowledge graphs. "
            "Your task is to determine the validity period (in days) "
            "for temporal rules involving a given target relation. "
            "The validity period should be no less than the maximum "
            "observed path time difference. "
            "Reply with ONLY a single number (days). No explanation."
        )

        user = (
            f"Target relation: \"{target_relation}\"\n"
            f"Maximum path time difference (days): {max_diff:.1f}\n"
            f"Default validity from knowledge base: {default_v:.0f} days\n\n"
            f"Sampled temporal paths:\n{path_text}\n\n"
            f"Based on these paths, what is the most appropriate validity "
            f"period V (in days) for rules involving \"{target_relation}\"? "
            f"V must be >= {max_diff:.1f}."
        )

        result = llm_chat(system, user, max_tokens=64, temperature=0.0)
        if result is None:
            return None

        # 解析数字
        import re
        nums = re.findall(r'[\d.]+', result)
        if nums:
            v = float(nums[0])
            return max(max_diff, v)  # 确保不小于 max_diff
        return None

    # --------------------------------------------------------
    # 批量评估
    # --------------------------------------------------------
    def assess_rules(self,
                     rules: list,
                     target_time: float,
                     target_relation: str,
                     all_paths: List[List[Quadruple]]) -> list:
        """
        批量为规则设置有效期

        Args:
            rules: 候选规则列表 (TemporalRule对象)
            target_time: 目标时间
            target_relation: 目标关系
            all_paths: 所有采样路径

        Returns:
            rules: 更新了validity_period的规则列表
        """
        # 确定整体有效期
        global_validity = self.determine_validity_period(
            target_time, all_paths, target_relation
        )

        for rule in rules:
            # 为每条规则微调有效期
            if hasattr(rule, 'source_path') and rule.source_path:
                rule_time_diffs = self.compute_all_path_time_diffs(
                    target_time, [rule.source_path]
                )
                rule_max_diff = max(rule_time_diffs) if rule_time_diffs else 0

                # 规则特有有效期 = max(全局有效期, 规则自身的max_diff)
                rule.validity_period = max(global_validity, rule_max_diff * 1.2)
            else:
                rule.validity_period = global_validity

        return rules

    # --------------------------------------------------------
    # 过滤时序违例的路径
    # --------------------------------------------------------
    @staticmethod
    def filter_valid_temporal_paths(paths: List[List[Quadruple]],
                                    validity_period: float,
                                    target_time: float) -> List[List[Quadruple]]:
        """
        过滤在有效期内满足时序约束的路径

        第4.2节: 仅保留在有效期V内的路径
        """
        valid_paths = []
        for path in paths:
            if not path:
                continue
            # 检查路径中所有事件是否在有效期内
            all_times = [q.time_val for q in path]
            in_validity = all(
                target_time - t <= validity_period
                for t in all_times
            )
            # 检查时序顺序
            in_order = all(
                all_times[i] <= all_times[i + 1]
                for i in range(len(all_times) - 1)
            )
            if in_validity and in_order:
                valid_paths.append(path)
        return valid_paths

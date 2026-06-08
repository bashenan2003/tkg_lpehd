"""
规则推理模块
=============
核心算法:
1. 公式(10): Sr' = {ρ | ρ ∈ Sr ∧ score(ρ) ≥ θ}
   从规则集Sr选择高优先级规则

2. 公式(11): (es, rq, eo, tq) ← ∧(es, ri, eo, ti)
   应用规则推导候选实体eo

3. 公式(12): Score(ρ, eo) = Σ(cρ + exp(-λ(tr - to)))
   计算候选实体得分, 考虑时序衰减
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

from ..config import LAMBDA_DECAY
from ..data.dataset import TKGDataSet, Quadruple
from ..rules.rule_mining import TemporalRule


class RuleReasoner:
    """
    规则推理器
    基于高优先级规则的时序推理
    """

    def __init__(self, dataset: TKGDataSet):
        self.dataset = dataset
        self.decay_rate = LAMBDA_DECAY  # λ (公式12)

    # --------------------------------------------------------
    # 公式(10): 筛选高优先级规则
    # --------------------------------------------------------
    @staticmethod
    def select_high_priority_rules(rules: List[TemporalRule],
                                   priority_threshold: float = 0.5) -> List[TemporalRule]:
        """
        Sr' = {ρ | ρ ∈ Sr ∧ score(ρ) ≥ θ}

        选择优先级分数超过阈值的规则
        """
        return [r for r in rules if r.priority_score >= priority_threshold]

    # --------------------------------------------------------
    # 公式(11): 应用规则推导实体
    # --------------------------------------------------------
    def apply_rules(self,
                    rules: List[TemporalRule],
                    query_subject: str,
                    query_time: float) -> List[Tuple[str, TemporalRule, List[Quadruple]]]:
        """
        应用规则推导候选实体
        论文公式(11): (es, rq, eo, tq) ← ∧(es, ri, eo, ti)

        遍历所有四元组, 匹配规则体模式,
        过滤中间实体满足时间约束 t1 < t2 < ... < ti < tq

        Returns:
            List[(candidate_entity, rule, matched_path)]
        """
        candidates = []
        all_quads = sorted(
            [q for q in self.dataset.quadruples if q.time_val <= query_time],
            key=lambda q: q.time_val
        )

        for rule in rules:
            # 对每个规则, 在数据中匹配规则体
            for i in range(len(all_quads)):
                matched_path = self._match_rule_body_forward(
                    rule, all_quads[i:], query_subject, query_time
                )
                if matched_path:
                    # 路径最后一个实体的object即为候选
                    candidate_entity = matched_path[-1].object
                    candidates.append((candidate_entity, rule, matched_path))

        # 去重并聚合
        entity_candidates = defaultdict(list)
        for eo, rule, path in candidates:
            entity_candidates[eo].append((rule, path))

        return [(eo, rules_paths) for eo, rules_paths in entity_candidates.items()]

    def _match_rule_body_forward(self,
                                  rule: TemporalRule,
                                  quads: List[Quadruple],
                                  query_subject: str,
                                  query_time: float) -> Optional[List[Quadruple]]:
        """
        前向匹配规则体

        从query_subject开始, 匹配规则体中的关系序列,
        确保时间严格递增: t1 < t2 < ... < ti < tq
        """
        path = []
        current_entity = query_subject
        current_time = -float("inf")

        body_idx = 0
        for quad in quads:
            if body_idx >= len(rule.body_relations):
                break

            if (quad.subject == current_entity and
                quad.relation == rule.body_relations[body_idx] and
                quad.time_val > current_time and
                quad.time_val <= query_time):

                path.append(quad)
                current_entity = quad.object
                current_time = quad.time_val
                body_idx += 1

            if body_idx == len(rule.body_relations):
                break

        if body_idx == len(rule.body_relations):
            return path
        return None

    # --------------------------------------------------------
    # 公式(12): Score(ρ, eo) = Σ(cρ + exp(-λ(tr - to)))
    # --------------------------------------------------------
    def compute_rule_score(self,
                           candidate_entity: str,
                           rule: TemporalRule,
                           matched_paths: List[List[Quadruple]],
                           query_time: float) -> float:
        """
        论文公式(12): Score(ρ, eo) = Σ[ cρ + exp(-λ(tr - to)) ]

        对每条匹配路径累加得分:
        - cρ: 规则置信度
        - exp(-λ(tr - to)): 时间衰减项
          - tr: 路径中事件时间
          - to: 目标查询时间

        Returns:
            float: 规则推理得分
        """
        total_score = 0.0

        for path in matched_paths:
            if not path:
                continue

            # 对路径中每个事件累加时间衰减得分
            for quad in path:
                time_diff = abs(query_time - quad.time_val)
                time_decay = np.exp(-self.decay_rate * time_diff)
                total_score += (rule.confidence + time_decay)

        return total_score

    # --------------------------------------------------------
    # 完整规则推理
    # --------------------------------------------------------
    def reason(self,
               rules: List[TemporalRule],
               query_subject: str,
               query_relation: str,
               query_time: str,
               priority_threshold: float = 0.5) -> Dict[str, float]:
        """
        执行完整的规则推理

        Args:
            rules: 可用规则集
            query_subject: 查询主体 es
            query_relation: 查询关系 rq
            query_time: 查询时间 tq

        Returns:
            Dict[entity, score]: 候选实体及其得分
        """
        tq = Quadruple._parse_time(query_time)

        # Step 1: 选择高优先级规则 (公式10)
        high_priority = self.select_high_priority_rules(rules, priority_threshold)

        if not high_priority:
            return {}

        # Step 2: 应用规则推导候选实体 (公式11)
        candidates = self.apply_rules(high_priority, query_subject, tq)

        # Step 3: 计算每个候选实体的得分 (公式12)
        entity_scores = {}
        for eo, rules_paths in candidates:
            total_score = 0.0
            for rule, path_list in rules_paths:
                score = self.compute_rule_score(
                    eo, rule, [path_list], tq
                )
                total_score += score

            if total_score > 0:
                entity_scores[eo] = total_score

        return entity_scores

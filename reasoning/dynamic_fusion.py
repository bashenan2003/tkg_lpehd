"""
动态权重融合推理模块
=======================
核心算法:
1. 公式(14): α_dynamic 动态融合权重
   α_dynamic = α_fixed × exp(-λ·dr) / [α_fixed × exp(-λ·dr)
               + (1-α_fixed) × exp(-λ·dg)]

   - α_fixed: 固定基础权重 (规则推理的基础贡献比例)
   - λ: 时间衰减系数
   - dr: 规则推理时间距离
   - dg: 图推理时间距离

2. 公式(15): Score_f = α_dynamic × Score(ρ,eo) + (1-α_dynamic) × Score_graph(eo)
   最终融合得分

核心思想: 动态适应时序知识图谱的演化特性
- 近期时间戳: 规则推理更可靠 (规则能捕获最新数据模式)
- 远历史数据: 图推理更鲁棒 (规则可能过时, 嵌入更稳定)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from ..config import (
    ALPHA_FIXED, LAMBDA_DECAY
)
from ..data.dataset import TKGDataSet, Quadruple


class DynamicWeightFusion:
    """
    动态权重融合器
    时间感知的动态融合机制
    """

    def __init__(self, dataset: TKGDataSet):
        self.dataset = dataset
        self.alpha_fixed = ALPHA_FIXED   # α_fixed
        self.lambda_decay = LAMBDA_DECAY # λ

    # --------------------------------------------------------
    # 公式(14): α_dynamic
    # --------------------------------------------------------
    def compute_dynamic_weight(self,
                                dr: float,
                                dg: float) -> float:
        """
        论文公式(14): 计算动态融合权重 α_dynamic

        α_dynamic = α_fixed × exp(-λ·dr)
                   ─────────────────────────────────
                   α_fixed × exp(-λ·dr) + (1-α_fixed) × exp(-λ·dg)

        Args:
            dr: 规则推理的时间距离 (规则涉及事件的时间跨度)
            dg: 图推理的时间距离 (实体交互事件的时间跨度)

        Returns:
            float: α_dynamic ∈ (0, 1)
        """
        # 计算分子
        numerator = self.alpha_fixed * np.exp(-self.lambda_decay * dr)

        # 计算分母
        denominator = (
            self.alpha_fixed * np.exp(-self.lambda_decay * dr) +
            (1 - self.alpha_fixed) * np.exp(-self.lambda_decay * dg)
        )

        if denominator == 0:
            return self.alpha_fixed  # 回退到固定权重

        alpha_dynamic = numerator / denominator
        return float(alpha_dynamic)

    def compute_temporal_distances(self,
                                    query_time: float,
                                    rule_scores: Dict[str, float],
                                    graph_scores: Dict[str, float]) -> Tuple[float, float]:
        """
        计算规则推理和图推理的时间距离

        dr: 规则推理分支中事件的时间间隔
            基于匹配规则的数据时间跨度

        dg: 图推理分支中实体交互事件的时间间隔
            基于图中邻接边的时间跨度
        """
        # dr: 规则推理的时间距离
        # 估算为训练数据中相关规则事件的平均时间跨度
        rule_times = [q.time_val for q in self.dataset.quadruples
                      if q.time_val <= query_time]
        if rule_times:
            dr = np.mean([abs(query_time - t) for t in rule_times[-100:]])
        else:
            dr = 10.0  # 默认10天

        # dg: 图推理的时间距离
        # 估算为图中实体对交互事件的平均时间间隔
        graph_times = [q.time_val for q in self.dataset.quadruples
                       if q.time_val <= query_time]
        if graph_times:
            dg = np.mean([abs(query_time - t) for t in graph_times[-50:]])
        else:
            dg = 10.0

        return dr, dg

    # --------------------------------------------------------
    # 公式(15): Score_f
    # --------------------------------------------------------
    def fuse(self,
             rule_scores: Dict[str, float],
             graph_scores: Dict[str, float],
             query_time: float,
             query_relation: str = None) -> Dict[str, float]:
        """
         Score_f = α_dynamic × Score(ρ,eo) + (1-α_dynamic) × Score_graph(eo)

        动态融合规则推理和图推理得分

        核心逻辑:
        - 近期查询: α_dynamic ↑ → 规则推理权重更高
        - 历史查询: α_dynamic ↓ → 图推理权重更高

        Returns:
            Dict[entity, final_score]: 候选实体的最终融合得分
        """
        # 计算时间距离
        dr, dg = self.compute_temporal_distances(
            query_time, rule_scores, graph_scores
        )

        # 计算动态权重 (公式14)
        alpha = self.compute_dynamic_weight(dr, dg)

        # 收集所有候选实体
        all_entities = set(rule_scores.keys()) | set(graph_scores.keys())

        # 归一化准备
        if rule_scores:
            max_rule = max(rule_scores.values())
        else:
            max_rule = 1.0

        if graph_scores:
            max_graph = max(graph_scores.values())
        else:
            max_graph = 1.0

        # 融合得分
        final_scores = {}
        for entity in all_entities:
            # 获取归一化得分
            rs = rule_scores.get(entity, 0.0) / max(1.0, max_rule)
            gs = graph_scores.get(entity, 0.0) / max(1.0, max_graph)

            # Score_f = α × Score_rule + (1-α) × Score_graph
            score_f = alpha * rs + (1 - alpha) * gs
            final_scores[entity] = float(score_f)

        return final_scores

    # --------------------------------------------------------
    # 候选排序
    # --------------------------------------------------------
    def rank_candidates(self,
                        final_scores: Dict[str, float],
                        top_k: int = 10) -> List[Tuple[str, float]]:
        """
        按融合得分降序排列候选实体

        得分最高的候选作为最终预测
        """
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # --------------------------------------------------------
    # 完整融合推理
    # --------------------------------------------------------
    def full_fusion_reasoning(self,
                               rule_scores: Dict[str, float],
                               graph_scores: Dict[str, float],
                               query_subject: str,
                               query_relation: str,
                               query_time: str,
                               top_k: int = 10) -> Dict:
        """
        执行完整的动态权重融合推理

        融合 → 排序 → 输出最高得分候选

        Returns:
            {
                "alpha_dynamic": float,
                "dr": float, "dg": float,
                "ranked_candidates": [(entity, score), ...],
                "top_prediction": str,
                "all_scores": Dict[str, float],
            }
        """
        tq = Quadruple._parse_time(query_time)

        # 计算时间距离和动态权重
        dr, dg = self.compute_temporal_distances(tq, rule_scores, graph_scores)
        alpha = self.compute_dynamic_weight(dr, dg)

        print(f"\n[动态融合] 论文公式(14)(15) - Dynamic Weight Fusion")
        print(f"  规则推理候选: {len(rule_scores)}个")
        print(f"  图推理候选: {len(graph_scores)}个")
        print(f"  时间距离: dr={dr:.1f}, dg={dg:.1f}")
        print(f"  α_fixed={self.alpha_fixed:.2f} → α_dynamic={alpha:.4f}")
        print(f"  融合解释: {'规则推理占主导' if alpha > 0.5 else '图推理占主导'}")

        # 融合得分
        final_scores = self.fuse(rule_scores, graph_scores, tq, query_relation)

        # 排序
        ranked = self.rank_candidates(final_scores, top_k)

        top_prediction = ranked[0][0] if ranked else "无预测"

        print(f"  Top-5预测: {[e for e, s in ranked[:5]]}")

        return {
            "alpha_dynamic": alpha,
            "dr": dr,
            "dg": dg,
            "ranked_candidates": ranked,
            "top_prediction": top_prediction,
            "all_scores": final_scores,
        }

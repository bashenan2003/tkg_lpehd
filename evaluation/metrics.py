"""
模型评估与结果输出模块
=========================
第5.1.3节: Parameter Settings and Evaluation Metrics

评估指标:
- MRR (Mean Reciprocal Rank): 正确答案排名的倒数均值
- Hit@1: 正确答案在Top-1中的比例
- Hit@3: 正确答案在Top-3中的比例
- Hit@10: 正确答案在Top-10中的比例
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from ..config import EVAL_METRICS, OUTPUT_DIR
from ..data.dataset import Quadruple


class Evaluator:
    """
    TKG推理评估器
    严格按第5.1.3节定义的指标计算

    用于评估: MRR, Hit@1, Hit@3, Hit@10
    """

    def __init__(self):
        self.results: Dict[str, List[float]] = {
            "MRR": [],
            "Hit@1": [],
            "Hit@3": [],
            "Hit@10": [],
        }
        self.query_count = 0

    # --------------------------------------------------------
    # 单次查询评估
    # --------------------------------------------------------
    def evaluate_query(self,
                       ranked_candidates: List[str],
                       ground_truth: str) -> Dict[str, float]:
        """
        评估单个查询的推理结果

        Args:
            ranked_candidates: 排序后的候选实体列表 (降序)
            ground_truth: 正确答案 (真实tail实体)

        Returns:
            {"MRR": float, "Hit@1": int, "Hit@3": int, "Hit@10": int}
        """
        if not ranked_candidates:
            return {"MRR": 0.0, "Hit@1": 0, "Hit@3": 0, "Hit@10": 0}

        # 查找正确答案的排名 (1-indexed)
        try:
            rank = ranked_candidates.index(ground_truth) + 1
        except ValueError:
            rank = len(ranked_candidates) + 1  # 未找到

        # MRR: 1/rank
        mrr = 1.0 / rank

        # Hit@N
        hit1 = 1 if rank <= 1 else 0
        hit3 = 1 if rank <= 3 else 0
        hit10 = 1 if rank <= 10 else 0

        return {
            "MRR": mrr,
            "Hit@1": hit1,
            "Hit@3": hit3,
            "Hit@10": hit10,
        }

    # --------------------------------------------------------
    # 批量评估
    # --------------------------------------------------------
    def evaluate_batch(self,
                       predictions: List[Tuple[List[str], str]]) -> Dict[str, float]:
        """
        批量评估

        Args:
            predictions: [(ranked_candidates, ground_truth), ...]

        Returns:
            {"MRR": float, "Hit@1": float, "Hit@3": float, "Hit@10": float}
        """
        per_query_results = []
        for ranked, gt in predictions:
            result = self.evaluate_query(ranked, gt)
            per_query_results.append(result)

        # 聚合
        n = len(per_query_results)
        aggregated = {}
        for metric in ["MRR", "Hit@1", "Hit@3", "Hit@10"]:
            values = [r[metric] for r in per_query_results]
            aggregated[metric] = float(np.mean(values))

        return aggregated

    # --------------------------------------------------------
    # 运行完整评估
    # --------------------------------------------------------
    def run_evaluation(self,
                       test_queries: List[Tuple[str, str, str, str]],
                       predictor_func,
                       top_k: int = 10) -> Dict[str, float]:
        """
        在测试集上运行完整评估

        Args:
            test_queries: [(subject, relation, object_gt, time), ...]
                          注: object_gt是真实答案 (held-out)
            predictor_func: 预测函数, 接受(s, r, t)返回ranked_candidates
            top_k: 考虑的前K个候选

        Returns:
            聚合后的评估指标
        """
        print(f"\n[评估] 运行{len(test_queries)}个测试查询...")

        predictions = []
        for i, (s, r, o_gt, t) in enumerate(test_queries):
            try:
                ranked_candidates = predictor_func(s, r, t, top_k=top_k)
                predictions.append((ranked_candidates, o_gt))
            except Exception as ex:
                print(f"  查询{ i}错误: {ex}")
                predictions.append(([], o_gt))

            if (i + 1) % 50 == 0:
                print(f"  进度: {i + 1}/{len(test_queries)}")

        results = self.evaluate_batch(predictions)
        self.query_count += len(test_queries)

        # 存储结果
        for metric, value in results.items():
            self.results[metric].append(value)

        return results

    # --------------------------------------------------------
    # 打印结果表格 (Table 1格式)
    # --------------------------------------------------------
    def print_results(self, dataset_name: str = "ICEWS14",
                       model_name: str = "TKG-LPEHD"):
        """打印评估结果"""
        print("\n" + "=" * 70)
        print(f"  TKG-LPEHD 推理评估结果")
        print("=" * 70)
        print(f"  数据集: {dataset_name}")
        print(f"  模型: {model_name}")
        print(f"  测试查询数: {self.query_count}")
        print("-" * 70)
        print(f"  {'指标':<12} {'得分':<10}")
        print("-" * 70)

        # 使用最近一次评估结果
        current = {}
        for metric in EVAL_METRICS:
            if self.results[metric]:
                current[metric] = self.results[metric][-1]
            else:
                current[metric] = 0.0

        for metric in EVAL_METRICS:
            print(f"  {metric:<12} {current[metric]:.4f}")

        print("-" * 70)

        # ICEWS14基线对比 (仅在ICEWS14数据集时显示)
        if dataset_name.upper() == "ICEWS14":
            print(f"\n  Table 1中的ICEWS14基线对比:")
            print(f"  {'模型':<20} {'MRR':<8} {'Hit@1':<8} {'Hit@3':<8} {'Hit@10':<8}")
            baselines = {
                "RE-NET": (0.388, 0.290, 0.436, 0.576),
                "TLogic": (0.430, 0.336, 0.483, 0.612),
                "Llama-2 7b-CoH": (0.425, 0.349, 0.470, 0.591),
                "TiRGNN + CoH": (0.439, 0.331, 0.496, 0.649),
                "TKG-LPEHD ()": (0.463, 0.392, 0.528, 0.673),
            }
            for model, (mrr, h1, h3, h10) in baselines.items():
                print(f"  {model:<20} {mrr:<8.3f} {h1:<8.3f} {h3:<8.3f} {h10:<8.3f}")

        print("=" * 70)

        return current

    # --------------------------------------------------------
    # 导出结果
    # --------------------------------------------------------
    def export_results(self, filepath: str = None):
        """导出评估结果"""
        import json
        import os

        if filepath is None:
            filepath = os.path.join(OUTPUT_DIR, "evaluation_results.json")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        output = {
            "query_count": self.query_count,
            "metrics": {
                metric: {
                    "last": self.results[metric][-1] if self.results[metric] else 0.0,
                    "all": self.results[metric],
                }
                for metric in EVAL_METRICS
            }
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[评估] 结果已导出: {filepath}")


def generate_test_queries(dataset, num_queries: int = 100) -> List[Tuple[str, str, str, str]]:
    """
    从测试集生成评估查询

    每个查询格式: (subject, relation, object_gt, time)
    其中object_gt是真实答案
    """
    import random
    rng = random.Random(42)

    test_quads = dataset.test_quads if dataset.test_quads else dataset.quadruples[-500:]

    queries = []
    for q in rng.sample(test_quads, min(num_queries, len(test_quads))):
        queries.append((q.subject, q.relation, q.object, q.timestamp))

    return queries

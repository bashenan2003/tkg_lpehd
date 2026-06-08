"""
工具函数模块
============
通用辅助函数: 时间处理、数据处理、格式化输出等
"""

import os
import sys
import time
import random
import numpy as np
import torch
from datetime import datetime, timedelta


def set_seed(seed: int = 42):
    """固定随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def timestamp_to_days(ts_str: str) -> int:
    """
    将时间戳字符串转换为相对天数
    支持格式: "2014-01-01", "2014-01-01 08:00", "2014"
    """
    try:
        base_date = datetime(2014, 1, 1)
        ts_str = str(ts_str).strip()
        if len(ts_str) == 4:  # 年份格式
            dt = datetime(int(ts_str), 1, 1)
        elif len(ts_str) == 10:  # "2014-01-01"
             dt = datetime.strptime(ts_str, "%Y-%m-%d")
        elif len(ts_str) == 19:  # "2014-01-01 08:00:00"
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        else:
            dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        return (dt - base_date).days
    except Exception:
        return 0


def days_to_timestamp(days: int) -> str:
    """将相对天数转回时间戳字符串"""
    base_date = datetime(2014, 1, 1)
    dt = base_date + timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


def compute_cosine_similarity(vec_a, vec_b) -> float:
    """
    计算余弦相似度 (对应论文公式(2): S1 = cos(A,B))
    """
    if isinstance(vec_a, torch.Tensor):
        vec_a = vec_a.detach().cpu().numpy()
    if isinstance(vec_b, torch.Tensor):
        vec_b = vec_b.detach().cpu().numpy()
    vec_a = np.asarray(vec_a).flatten().astype(np.float64)
    vec_b = np.asarray(vec_b).flatten().astype(np.float64)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def top_k_indices(scores: list, k: int = 10) -> list:
    """返回分数最高的k个索引"""
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:k]]


class ProgressBar:
    """简单进度条"""
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        pct = self.current / self.total * 100
        elapsed = time.time() - self.start_time
        bar_len = 30
        filled = int(bar_len * self.current / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r{self.desc} [{bar}] {pct:.1f}% ({self.current}/{self.total}) {format_time(elapsed)}")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


# ------------------------------------------------------------------
# Safe XSD date literal — clamps invalid dates (e.g. 2023-02-29)
# to the last valid day of the month so rdflib never crashes.
# ------------------------------------------------------------------
def clamp_to_valid_date(ts: str) -> str:
    """Return *ts* unchanged if valid, else the last valid day of the month."""
    try:
        y, m, d = int(ts[:4]), int(ts[5:7]), int(ts[8:10])
        # Fast-path: most dates are fine
        if 1 <= m <= 12 and 1 <= d <= 28:
            return ts
        import calendar
        last_day = calendar.monthrange(y, m)[1]
        if d > last_day:
            return f"{y:04d}-{m:02d}-{last_day:02d}"
        return ts
    except Exception:
        return ts


def safe_xsd_date_literal(t: str):
    """
    Build an rdflib ``Literal`` with datatype ``XSD.date``.

    If *t* is not a valid calendar date (e.g. ``2023-02-29``),
    the day is clamped to the last valid day of the month before
    creating the literal — the underlying data is NOT mutated.
    """
    from rdflib import Literal
    from rdflib.namespace import XSD
    safe = clamp_to_valid_date(t)
    return Literal(safe, datatype=XSD.date)


# ------------------------------------------------------------------
# 性能计时器 (输出 Table 6 数据)
# ------------------------------------------------------------------
class PerfTimer:
    """全局性能计时器, 所有实例共享同一个存储。"""
    _store = {}
    _counters = {}

    @classmethod
    def start(cls, name: str):
        cls._store[name] = time.time()

    @classmethod
    def stop(cls, name: str):
        if name in cls._store:
            elapsed = time.time() - cls._store.pop(name)
            cls._store[f"_{name}_last"] = elapsed
            if name not in cls._counters:
                cls._counters[name] = {"total": 0.0, "count": 0}
            cls._counters[name]["total"] += elapsed
            cls._counters[name]["count"] += 1
            return elapsed
        return 0.0

    @classmethod
    def get(cls, name: str) -> float:
        """获取某阶段的总耗时 (秒)。"""
        return cls._counters.get(name, {}).get("total", 0.0)

    @classmethod
    def get_count(cls, name: str) -> int:
        return cls._counters.get(name, {}).get("count", 0)

    @classmethod
    def get_avg(cls, name: str) -> float:
        c = cls._counters.get(name, {})
        return c["total"] / c["count"] if c.get("count", 0) > 0 else 0.0

    @classmethod
    def get_last(cls, name: str) -> float:
        return cls._store.get(f"_{name}_last", cls._counters.get(name, {}).get("total", 0.0))

    @classmethod
    def summary(cls, dataset_name: str, num_quads: int,
                phase1_relations: int, phase1_paths_per_rel: int) -> str:
        """输出 Table 6 格式的性能报告。"""
        pt = cls
        load_t = pt.get("data_load")
        sbert_t = pt.get("sbert_load")
        path_t = pt.get("path_sampling")
        llm_path_t = pt.get("llm_path_validate")
        llm_valid_t = pt.get("llm_validity")
        rule_ind_t = pt.get("rule_induction")
        llm_evolve_t = pt.get("llm_evolution")
        sparql_t = pt.get("sparql_match")
        offline_t = sum(filter(None, [load_t, sbert_t, path_t, llm_path_t, llm_valid_t, rule_ind_t, llm_evolve_t, sparql_t]))
        n_paths = phase1_relations * phase1_paths_per_rel
        n_llm_path = phase1_relations

        rule_match_avg = pt.get_avg("rule_match_query")
        graph_avg = pt.get_avg("graph_reason_query")
        llm_verify_avg = pt.get_avg("llm_verify_query")
        fusion_avg = pt.get_avg("fusion_query")
        per_query = rule_match_avg + graph_avg + llm_verify_avg + fusion_avg

        lines = [
            f"\n{'='*65}",
            f"  Table 6: Computational Time Cost ({dataset_name}, {num_quads} quads)",
            f"{'='*65}",
            f"  {'Stage':<28} {'Task':<24} {'Time':>10}",
            f"  {'-'*28} {'-'*24} {'-'*10}",
            f"  {'Offline Preprocessing':<28} {'Data load + index':<24} {load_t:>8.1f}s",
            f"  {'':28} {'SBERT model load':<24} {sbert_t:>8.1f}s",
            f"  {'':28} {f'Path sampling ({n_paths} paths)':<24} {path_t:>8.1f}s",
            f"  {'':28} {f'LLM path validate ({n_llm_path} calls)':<24} {llm_path_t:>8.1f}s",
            f"  {'':28} {'LLM validity period':<24} {llm_valid_t:>8.1f}s",
            f"  {'':28} {'Initial rule induction':<24} {rule_ind_t:>8.2f}s",
            f"  {'':28} {'LLM rule evolution':<24} {llm_evolve_t:>8.1f}s",
            f"  {'':28} {'SPARQL matching':<24} {sparql_t:>8.1f}s",
            f"  {'':28} {f'Offline subtotal':<24} {offline_t:>8.1f}s",
            f"",
            f"  {'Online Inference':<28} {'Avg latency per query':<24} {per_query:>8.2f}s",
            f"  {'':28} {'  Rule matching':<24} {rule_match_avg:>8.2f}s",
            f"  {'':28} {'  Graph reasoning':<24} {graph_avg:>8.2f}s",
            f"  {'':28} {'  LLM verification':<24} {llm_verify_avg:>8.2f}s",
            f"  {'':28} {'  Dynamic fusion+Rank':<24} {fusion_avg:>8.3f}s",
            f"",
            f"  {'Extrapolated':<28} {'Path sampling / 1M paths':<24} {path_t/max(n_paths,1)*1e6:>8.0f}s",
            f"  {'':28} {'LLM eval / 50k candidates':<24} {llm_path_t/max(n_llm_path,1)*50000/3600:>8.1f}h",
            f"  {'':28} {'Incremental / 100k quads':<24} {(load_t+rule_ind_t)/max(num_quads,1)*100000:>8.1f}s",
            f"{'='*65}",
        ]
        return "\n".join(lines)

"""
时序归一化模块
================
时间戳归一化处理
功能:
1. 将多种时间格式统一为相对天数 (便于计算delta t)
2. 计算时间区间统计信息
3. 提供时间衰减权重预计算
"""

import numpy as np
from datetime import datetime
from typing import List, Dict


class TemporalNormalizer:
    """
    时序归一化器

    将时间戳字符串归一化为可用于计算的数值格式。
    论文公式(4)中的Δt, 公式(6)中的|t-T|, 公式(7)的t_target - t_final
    都需要基于归一化后的时间值进行计算。
    """

    def __init__(self, base_date: datetime = None):
        self.base_date = base_date or datetime(2014, 1, 1)

    def normalize(self, timestamps: List[str]) -> List[float]:
        """
        将时间戳列表归一化为相对数值 (天数为单位)

        支持格式: "2014-01-01", "2014-01-01 08:00", "2014"
        """
        values = []
        for ts in timestamps:
            values.append(self._parse_single(ts))
        return values

    def _parse_single(self, ts: str) -> float:
        """解析单个时间戳"""
        ts = str(ts).strip()
        try:
            if len(ts) == 4:  # 年份 "2014"
                return float((datetime(int(ts), 1, 1) - self.base_date).days)
            elif len(ts) >= 10:
                dt_str = ts[:10]
                dt = datetime.strptime(dt_str, "%Y-%m-%d")
                return float((dt - self.base_date).days)
            else:
                return 0.0
        except Exception:
            return 0.0

    def compute_time_intervals(self, times_a: List[float],
                                times_b: List[float]) -> List[float]:
        """
        计算两组时间之间的平均间隔 (采样加速)。
        """
        if not times_a or not times_b:
            return [float("inf")]
        # 大数据集时采样以加速 (O(n×m) → O(100×100))
        max_samples = 100
        if len(times_a) > max_samples:
            rng = np.random.RandomState(42)
            times_a = list(rng.choice(times_a, size=max_samples, replace=False))
        if len(times_b) > max_samples:
            rng = np.random.RandomState(42)
            times_b = list(rng.choice(times_b, size=max_samples, replace=False))
        intervals = []
        for ta in times_a:
            for tb in times_b:
                intervals.append(abs(ta - tb))
        return intervals

    def compute_mean_interval(self, times_a: List[float],
                               times_b: List[float]) -> float:
        """
        计算平均时间间隔
        论文公式(4): Δt = mean(|t_c - t_target|)
        """
        intervals = self.compute_time_intervals(times_a, times_b)
        if not intervals:
            return float("inf")
        finite = [d for d in intervals if d != float("inf")]
        if not finite:
            return float("inf")
        return float(np.mean(finite))

    def time_decay_weight(self, t_edge: float, t_target: float,
                           decay_coef: float = 0.1) -> float:
        """
        时间衰减权重
        公式(6): w(t) = exp(-λ · |t - T|)

        Args:
            t_edge: 路径边的时间戳
            t_target: 目标事件的时间戳
            decay_coef: λ, 默认为0.1 (第5.7.1节最优值)

        Returns:
            float: 衰减后的权重
        """
        diff = abs(t_edge - t_target)
        return float(np.exp(-decay_coef * diff))


def compute_validity_period(path_time_diffs: List[float]) -> float:
    """
    计算规则有效期
    公式(7)与第4.2节:
    Δti = t_target - t_final (每条路径的时间差)
    有效期V满足 max(Δti) ≤ V

    Returns:
        float: 建议的有效期V
    """
    if not path_time_diffs:
        return 30.0  # 默认30天
    return float(max(path_time_diffs))

"""
数据加载与预处理模块
====================
第5.1.1节: 数据集加载、清洗、时序归一化、词典构建

支持格式:
- ICEWS系列: CSV格式 (subject, relation, object, timestamp)
- YAGO: TTL/RDF格式或CSV格式

核心功能:
1. 四元组加载与统一格式化 → (s, r, o, t)
2. 实体/关系/时间戳词典构建映射到整数索引
3. 时序归一化 (将日期字符串转为相对数值)
4. 训练/验证/测试划分
5. 时序邻接图构建 (用于GNN推理)
"""

import os
import csv
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Set, Optional

from ..config import (
    DATA_DIR, DATASET_NAME, DATASET_PATHS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


class Quadruple:
    """四元组数据结构 (s, r, o, t) — 公式(1)基础"""
    __slots__ = ["subject", "relation", "object", "timestamp", "time_val"]

    def __init__(self, subject: str, relation: str, obj: str, timestamp: str):
        self.subject = subject
        self.relation = relation
        self.object = obj
        self.timestamp = timestamp
        self.time_val = self._parse_time(timestamp)

    @staticmethod
    def _parse_time(ts: str) -> float:
        """将时间戳字符串转换为数值"""
        try:
            ts = str(ts).strip()
            if len(ts) == 4:
                return float(ts)
            elif len(ts) == 10:
                dt = datetime.strptime(ts, "%Y-%m-%d")
                return dt.timestamp() / 86400  # 转天数
            else:
                dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
                return dt.timestamp() / 86400
        except Exception:
            return 0.0

    def to_tuple(self) -> Tuple[str, str, str, str]:
        return (self.subject, self.relation, self.object, self.timestamp)

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.object}, {self.timestamp})"


class TKGDataSet:
    """
    时序知识图谱数据集类

    管理:
    - 四元组集合
    - 实体/关系/时间戳词典映射
    - 时间划分 (历史/当前/未来 - 第3节)
    """

    def __init__(self, name: str = "ICEWS14"):
        self.name = name
        self.quadruples: List[Quadruple] = []
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.timestamps: Set[str] = set()

        # 词典映射: 字符串 → 整数索引
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        self.time2id: Dict[str, int] = {}
        self.id2time: Dict[int, str] = {}

        # 数据划分
        self.train_quads: List[Quadruple] = []
        self.val_quads: List[Quadruple] = []
        self.test_quads: List[Quadruple] = []

        # 时序邻接结构 (用于GNN)
        self.time_sorted_quads: List[Quadruple] = []
        self.adj_by_time: Dict[float, List[Quadruple]] = defaultdict(list)

    # ------------------------------
    # 加载
    # ------------------------------
    def load(self, data_path: str = None) -> "TKGDataSet":
        """
        从CSV或TTL文件加载四元组数据
        """
        if data_path is None:
            data_path = DATASET_PATHS.get(self.name,
                os.path.join(DATA_DIR, self.name.lower()))

        csv_path = os.path.join(data_path, f"{self.name.lower()}.csv")
        ttl_path = os.path.join(data_path, f"{self.name.lower()}.ttl")

        if os.path.exists(csv_path):
            self._load_csv(csv_path)
        elif os.path.exists(ttl_path):
            self._load_ttl(ttl_path)
        else:
            raise FileNotFoundError(
                f"数据集未找到: {csv_path} 或 {ttl_path}\n"
                f"请将数据文件放入对应数据集目录."
            )

        self._build_vocabularies()
        self._build_indices()             # 实体+关系索引 (加速路径采样)
        self._build_temporal_adjacency()
        self._split_data()
        print(f"[数据集] {self.name}: 总四元组={len(self.quadruples)}, "
              f"实体={self.num_entities}, 关系={self.num_relations}, "
              f"时间戳={self.num_timestamps}")
        print(f"  训练={len(self.train_quads)}, 验证={len(self.val_quads)}, "
              f"测试={len(self.test_quads)}")
        return self

    def _load_csv(self, path: str):
        """加载CSV格式 (ICEWS系列, MHAES) — 自动检测编码"""
        # 尝试多种编码
        text = None
        for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if text is None:
            raise ValueError(f"无法解码文件: {path}")
        lines = text.strip().split("\n")

        # 解析header确定列顺序
        header = lines[0].strip().split(",")
        subj_idx = header.index("subject") if "subject" in header else 0
        rel_idx = header.index("relation") if "relation" in header else 1
        obj_idx = header.index("object") if "object" in header else 2
        ts_idx = header.index("timestamp") if "timestamp" in header else 3

        import re
        _date_re = re.compile(r'^\d{4}-\d{2}-\d{2}$')

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            fields = line.split(",")
            # 修复: 关系名含逗号导致字段 >4 — 取最后一段作 timestamp
            if len(fields) > 4:
                ts = fields[-1].strip()
                obj = fields[-2].strip() if len(fields) >= 3 else ""
                rel = ",".join(f.strip() for f in fields[1:-2]).strip()
                subj = fields[0].strip()
            else:
                subj = fields[subj_idx].strip() if subj_idx < len(fields) else ""
                rel = fields[rel_idx].strip() if rel_idx < len(fields) else ""
                obj = fields[obj_idx].strip() if obj_idx < len(fields) else ""
                ts = fields[ts_idx].strip() if ts_idx < len(fields) else ""

            # 丢弃非法时间戳的行 (如 CSV 逗号分裂导致的错位)
            # 先做中文逗号修复再校验
            if not ts and subj:
                cn_parts = subj.replace("，", ",").split(",")
                if len(cn_parts) >= 4:
                    subj, rel, obj, ts = [p.strip() for p in cn_parts[:4]]
            if not ts and rel:
                cn_parts = rel.replace("，", ",").split(",")
                if len(cn_parts) >= 3:
                    rel, obj, ts = [p.strip() for p in cn_parts[:3]]

            if not _date_re.match(ts):
                continue
            if not subj or not rel or not obj:
                continue

            quad = Quadruple(
                subject=subj,
                relation=rel,
                obj=obj,
                timestamp=ts,
            )
            self.quadruples.append(quad)

    def _load_ttl(self, path: str):
        """
        加载TTL/Turtle格式 (YAGO数据集)
        第5.1.1节: YAGO使用RDF格式
        """
        # 简易Turtle解析器
        triples_buffer = []
        current_subj = None
        current_rel = None
        current_time = None

        text = None
        for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if text is None:
            raise ValueError(f"无法解码文件: {path}")
        content = text

        # 简单逐行解析
        lines = content.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("@prefix"):
                i += 1
                continue

            # 匹配: yago:entity yago:relation yago:entity ;
            parts = line.split()
            if len(parts) >= 3 and "yago:" in parts[0]:
                subj = parts[0].replace("yago:", "").rstrip(";")
                rel = parts[1].replace("yago:", "").rstrip(";")
                obj = parts[2].replace("yago:", "").rstrip(";")
                current_subj = subj
                current_rel = rel
                current_obj = obj

                # 下一行可能有时序信息
                if i + 1 < len(lines) and "hasTime" in lines[i + 1]:
                    time_line = lines[i + 1].strip()
                    time_str = time_line.split('"')[1] if '"' in time_line else "2014"
                    current_time = time_str
                    i += 1

                quad = Quadruple(current_subj, current_rel, current_obj, current_time or "2014")
                self.quadruples.append(quad)

            i += 1

    # ------------------------------
    # 词典构建
    # ------------------------------
    def _build_vocabularies(self):
        """构建实体/关系/时间戳到整数索引的双向词典"""
        for q in self.quadruples:
            self.entities.add(q.subject)
            self.entities.add(q.object)
            self.relations.add(q.relation)
            self.timestamps.add(q.timestamp)

        # 实体词典
        for idx, ent in enumerate(sorted(self.entities)):
            self.entity2id[ent] = idx
            self.id2entity[idx] = ent

        # 关系词典
        for idx, rel in enumerate(sorted(self.relations)):
            self.relation2id[rel] = idx
            self.id2relation[idx] = rel

        # 时间戳词典
        for idx, ts in enumerate(sorted(self.timestamps)):
            self.time2id[ts] = idx
            self.id2time[idx] = ts

    # ------------------------------
    # 索引构建 (加速路径采样)
    # ------------------------------
    def _build_indices(self):
        """构建实体索引和关系索引, O(N) 一次构建, O(1) 查询。"""
        self.quads_by_subject: Dict[str, List[Quadruple]] = defaultdict(list)
        self.quads_by_relation: Dict[str, List[Quadruple]] = defaultdict(list)
        for q in self.quadruples:
            self.quads_by_subject[q.subject].append(q)
            self.quads_by_relation[q.relation].append(q)

    def get_quads_by_subject(self, subject: str) -> List[Quadruple]:
        """O(1) 获取某实体的所有出边。"""
        return self.quads_by_subject.get(subject, [])

    def get_quads_by_relation(self, relation: str) -> List[Quadruple]:
        """O(1) 获取某关系的所有四元组。"""
        return self.quads_by_relation.get(relation, [])

    # ------------------------------
    # 时序邻接结构
    # ------------------------------
    def _build_temporal_adjacency(self):
        """
        构建时序邻接结构
        按时间排序, 为GNN聚合提供时序上下文
        第4.5.2节: 时间感知邻居聚合
        """
        self.time_sorted_quads = sorted(self.quadruples, key=lambda q: q.time_val)
        for q in self.quadruples:
            self.adj_by_time[q.time_val].append(q)

    # ------------------------------
    # 数据划分 (历史/当前/未来)
    # ------------------------------
    def _split_data(self):
        """
        按时间顺序划分训练/验证/测试集
        第3节: 历史数据/当前数据/未来数据
        严格保持时序: train < val < test
        """
        sorted_quads = sorted(self.quadruples, key=lambda q: q.time_val)
        n = len(sorted_quads)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        self.train_quads = sorted_quads[:train_end]
        self.val_quads = sorted_quads[train_end:val_end]
        self.test_quads = sorted_quads[val_end:]

    # ------------------------------
    # 属性
    # ------------------------------
    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)

    @property
    def num_timestamps(self) -> int:
        return len(self.time2id)

    def get_quads_by_time_range(self, t_min: float, t_max: float) -> List[Quadruple]:
        """获取时间范围内的四元组"""
        return [q for q in self.quadruples if t_min <= q.time_val <= t_max]

    def get_subject_relations(self, subject: str) -> List[Tuple[str, str, str]]:
        """获取某实体的所有出边 (r, o, t)"""
        return [(q.relation, q.object, q.timestamp)
                for q in self.quadruples if q.subject == subject]

    def get_object_relations(self, obj: str) -> List[Tuple[str, str, str]]:
        """获取某实体的所有入边 (s, r, t)"""
        return [(q.subject, q.relation, q.timestamp)
                for q in self.quadruples if q.object == obj]

    def get_quads_before(self, timestamp: str) -> List[Quadruple]:
        """获取指定时间戳之前的所有四元组 (用于推理)"""
        target_val = Quadruple._parse_time(timestamp)
        return [q for q in self.quadruples if q.time_val < target_val]

    # ------------------------------
    # 摘要
    # ------------------------------
    def summary(self) -> str:
        lines = [
            f"数据集: {self.name}",
            f"  四元组总数: {len(self.quadruples)}",
            f"  实体数: {self.num_entities}",
            f"  关系数: {self.num_relations}",
            f"  时间戳数: {self.num_timestamps}",
            f"  训练集: {len(self.train_quads)}",
            f"  验证集: {len(self.val_quads)}",
            f"  测试集: {len(self.test_quads)}",
        ]
        if self.quadruples:
            times = [q.time_val for q in self.quadruples]
            lines.append(f"  时间范围: {min(times):.0f} ~ {max(times):.0f}")
        return "\n".join(lines)


def load_or_generate_dataset(dataset_name: str = "ICEWS14",
                              num_quads: int = 5000) -> TKGDataSet:
    """
    加载数据集 (数据文件必须已存在于 data_files/ 目录下)

    Args:
        dataset_name: "ICEWS14" | "ICEWS0515" | "ICEWS18" | "YAGO" | "mhaes"
        num_quads: (保留参数, 仅用于兼容旧调用)

    Returns:
        TKGDataSet实例
    """
    ds_key = dataset_name.upper() if dataset_name.upper() in DATASET_PATHS else dataset_name
    data_path = DATASET_PATHS.get(ds_key)
    if data_path is None:
        data_path = os.path.join(DATA_DIR, dataset_name.lower())
    csv_path = os.path.join(data_path, f"{dataset_name.lower()}.csv")
    ttl_path = os.path.join(data_path, f"{dataset_name.lower()}.ttl")

    if not (os.path.exists(csv_path) or os.path.exists(ttl_path)):
        raise FileNotFoundError(
            f"数据集 '{dataset_name}' 未找到: {csv_path} 或 {ttl_path}\n"
            f"请将数据文件放入 {data_path} 目录."
        )

    dataset = TKGDataSet(name=dataset_name)
    dataset.load(data_path)
    return dataset


if __name__ == "__main__":
    ds = load_or_generate_dataset("ICEWS14", 5000)
    print(ds.summary())

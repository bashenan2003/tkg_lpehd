"""
LLM引导三维关系路径采样模块
=============================

核心算法:
1. 公式(2): S1 = cos(A,B) - 语义相关性 (Sentence-BERT余弦相似度)
2. 公式(3): S2 = Count(rc) / ΣCount(rp) - 出现频率
3. 公式(4): S3 = 1/Δt - 时间相关性
4. 公式(5): S = w1*S1 + w2*S2 + w3*S3 - 三维加权综合得分
5. 公式(6): w(t) = exp(-λ·|t - T|) - 时间衰减权重

路径采样流程:
a) 均匀采样规则头边: (e1, rh, e_{l+1}, t_{l+1})
b) 对每个中间步骤m, 评估候选关系: S1/S2/S3 → 加权S
c) 符号约束: T1 ≤ T2 < T3 (严格时序)
d) 时间加权采样: 按w(t)优先采样
e) LLM逻辑验证: 选择最合理的路径
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

from ..config import (
    W_SEMANTIC, W_FREQUENCY, W_TEMPORAL,
    LAMBDA_DECAY, MAX_PATH_LENGTH, DEVICE,
    is_real_llm_enabled, get_ablation_mode,
)
from ..data.dataset import TKGDataSet, Quadruple
from ..preprocessing.temporal_normalizer import TemporalNormalizer
from ..utils import compute_cosine_similarity


class SBertSimulator:
    """
    Sentence-BERT 语义相似度编码器
    公式(2): S1 = cos(A,B)

    优先加载真实 Sentence-BERT 模型;
    若模型不可用则回退到语义分组 + 确定性伪嵌入。
    """

    # 真实模型 (类级别单例, 所有实例共享)
    _model = None
    _model_loaded = False
    _model_load_attempted = False

    # 关系语义分组 (ICEWS + MHAES) — 仅作为 fallback 使用
    SEMANTIC_GROUPS = {
        # ==================== ICEWS 关系 ====================
        # 声明/发言
        "statement": ["Make_statement", "Express_intent_to_cooperate",
                       "Express_intent_to_provide_aid"],
        # 外交访问
        "visit": ["Make_a_visit", "Host_a_visit", "Meet_with"],
        # 冲突/对抗
        "conflict": ["Accuse", "Criticize_or_denounce", "Threaten",
                      "Protest_or_demonstrate"],
        # 军事行动
        "military": ["Fight_with_artillery_and_tanks",
                      "Use_conventional_military_force", "Provide_military_aid"],
        # 援助/合作
        "aid": ["Provide_aid", "Provide_economic_aid",
                 "Cooperate_on_economic_issues"],
        # 外交合作
        "diplomacy": ["Engage_in_diplomatic_cooperation", "Consult",
                       "Engage_in_negotiation", "Mediate"],
        # 协定/签署
        "agreement": ["Sign_formal_agreement", "Ratify_treaty"],
        # 请求/呼吁
        "appeal": ["Make_an_appeal_or_request"],
        # 会谈
        "discuss": ["Discuss_by_telephone"],
        # 道歉
        "apology": ["Apologize"],
        # 赞扬
        "praise": ["Praise_or_endorse"],

        # ==================== MHAES 医疗防疫关系 ====================
        # 感染/患病
        "med_infection": [
            "感染", "确诊", "疑似", "无症状", "携带", "发病", "传染", "罹患",
            "受感染", "被传染", "检出", "阳性", "阴性", "复核", "排除感染",
            "确诊阳性", "上报", "登记", "筛查", "排查",
        ],
        # 药品/疫苗
        "med_drug": [
            "服用", "接种", "开具", "给药", "注射", "输液", "用药", "配药",
            "疫苗接种", "服用药物", "给药治疗", "用药治疗", "用药缓解",
            "配药服用", "注射给药", "输液治疗", "用药康复", "接种疫苗",
        ],
        # 症状/体征
        "med_symptom": [
            "发热", "咳嗽", "乏力", "咽痛", "鼻塞", "流涕", "肌肉酸痛", "头痛",
            "腹泻", "呕吐", "胸闷", "气促", "呼吸困难", "畏寒", "头晕",
            "体温上升", "体温下降", "体温升高", "体温降低",
            "症状缓解", "症状减轻", "症状加重", "体征稳定", "体征平稳",
        ],
        # 康复/转归
        "med_recovery": [
            "痊愈", "好转", "恶化", "出院", "留观", "康复", "转阴", "治愈",
            "病情稳定", "病情加重", "病情缓解", "病情观察",
            "康复出院", "转阴出院", "治愈出院", "好转出院",
            "康复治疗", "持续治疗",
        ],
        # 防控/隔离
        "med_prevention": [
            "隔离", "管控", "监测", "检测", "消杀", "封控",
            "管控措施", "隔离观察", "居家隔离", "集中隔离",
            "健康监测", "核酸检测", "抗原检测", "抗体检测",
            "环境消杀", "区域封控", "人员管控", "疫情监测", "风险排查",
            "防护", "消毒", "预检", "分诊", "检疫", "查验",
        ],
        # 诊疗/救治
        "med_treatment": [
            "收治", "治疗", "转诊", "监护", "诊断", "救治", "护理", "会诊",
            "住院", "门诊", "急诊", "留院观察",
            "重症监护", "专科治疗", "综合救治",
            "常规治疗", "对症治疗", "支持治疗",
        ],
        # 传播/流行
        "med_spread": [
            "爆发", "流行", "传播", "溯源", "扩散", "蔓延", "受控", "阻断",
            "遏制", "平息",
            "疫情爆发", "病毒传播", "疫情扩散", "疫情蔓延",
            "疫情受控", "传播阻断", "疫情遏制", "疫情平息",
        ],
        # 报告/行政
        "med_admin": [
            "报告", "统计", "预警", "处置", "防控", "值守", "督导", "检查",
            "保障", "支援", "随访", "回访",
            "疫情报告", "数据统计", "数据上报", "疫情上报",
            "风险预警", "风险评估", "应急处置", "疫情防控",
            "医疗保障", "医疗支援", "物资保障", "物资支援",
            "督导检查", "卫生监督",
        ],
    }
    GROUP_MAP = {}
    for group, rels in SEMANTIC_GROUPS.items():
        for r in rels:
            GROUP_MAP[r] = group

    def __init__(self):
        self.relation_vectors: Dict[str, np.ndarray] = {}
        self.dim = 384  # MiniLM 默认输出维度
        self._using_real_model = False

    # --------------------------------------------------------
    # 真实 Sentence-BERT 模型加载 (类级别单例)
    # --------------------------------------------------------
    @classmethod
    def _load_real_model(cls):
        """加载真实 Sentence-BERT 模型 (优先本地路径, 其次国内镜像)。"""
        if cls._model_load_attempted:
            return cls._model_loaded
        cls._model_load_attempted = True
        try:
            import os
            from sentence_transformers import SentenceTransformer
            from ..config import ROOT_DIR

            local_path = os.path.join(ROOT_DIR, "models")
            if os.path.isdir(local_path) and os.path.exists(
                os.path.join(local_path, "model.safetensors")
            ):
                cls._model = SentenceTransformer(local_path)
            else:
                # 回退: 从 HuggingFace (优先国内镜像) 下载
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                from ..config import SBERT_MODEL_NAME
                cls._model = SentenceTransformer(SBERT_MODEL_NAME)
            cls._model_loaded = True
            return True
        except Exception:
            return False

    def build_embeddings(self, relations: Set[str], dim: int = None):
        """
        构建关系语义嵌入。

        优先使用真实 Sentence-BERT (sentence-transformers) 批量编码;
        模型不可用时回退到语义分组确定性伪嵌入。
        """
        rel_list = sorted(relations)

        # ---- 尝试真实模型 ----
        if self._load_real_model():
            try:
                # 将关系名转为可读文本: "Make_a_visit" → "Make a visit"
                texts = [r.replace("_", " ") for r in rel_list]
                embeddings = self._model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False
                )
                for rel, vec in zip(rel_list, embeddings):
                    self.relation_vectors[rel] = vec.astype(np.float32)
                self.dim = embeddings.shape[1]
                self._using_real_model = True
                return
            except Exception:
                pass  # 编码失败 → 回退

        # ---- 回退: 语义分组确定性伪嵌入 ----
        if dim is None:
            dim = 128
        self.dim = dim
        self._using_real_model = False

        np.random.seed(42)
        group_centers = {}
        for group in self.SEMANTIC_GROUPS:
            center = np.random.randn(dim).astype(np.float32)
            group_centers[group] = center / np.linalg.norm(center)

        for rel in rel_list:
            group = self.GROUP_MAP.get(rel, "other")
            if group in group_centers:
                noise = np.random.randn(dim).astype(np.float32) * 0.15
                vec = group_centers[group] + noise
            else:
                vec = np.random.randn(dim).astype(np.float32)
            self.relation_vectors[rel] = vec / np.linalg.norm(vec)

    def similarity(self, rel_a: str, rel_b: str) -> float:
        """计算两个关系的语义相似度 (公式(2): S1 = cos(A,B))"""
        if rel_a not in self.relation_vectors or rel_b not in self.relation_vectors:
            return 0.3
        return compute_cosine_similarity(
            self.relation_vectors[rel_a],
            self.relation_vectors[rel_b]
        )

    def get_vector(self, rel: str) -> np.ndarray:
        """获取关系向量"""
        if rel in self.relation_vectors:
            return self.relation_vectors[rel]
        return np.random.randn(self.dim).astype(np.float32)


class ThreeDimensionalPathSampler:
    """
    三维关系路径采样器
    第4.1节: 集成语义相关性、出现频率和时间相关性
    """

    def __init__(self, dataset: TKGDataSet):
        self.dataset = dataset
        self.normalizer = TemporalNormalizer()
        self.sbert = SBertSimulator()
        self.sbert.build_embeddings(dataset.relations)

        # 预计算关系频率统计 (用于公式(3))
        self.relation_frequencies: Dict[str, int] = defaultdict(int)
        self.relation_times: Dict[str, List[float]] = defaultdict(list)
        for q in dataset.quadruples:
            self.relation_frequencies[q.relation] += 1
            self.relation_times[q.relation].append(q.time_val)

        self.total_frequency = sum(self.relation_frequencies.values())

    # --------------------------------------------------------
    # 公式(2): S1 - 语义相关性
    # --------------------------------------------------------
    def compute_semantic_relevance(self, target_rel: str,
                                   candidate_rel: str) -> float:
        """
        计算语义相关性 S1
        公式(2): S1 = cos(A,B)
        使用Sentence-BERT嵌入的余弦相似度
        """
        return self.sbert.similarity(target_rel, candidate_rel)

    # --------------------------------------------------------
    # 公式(3): S2 - 出现频率
    # --------------------------------------------------------
    def compute_occurrence_frequency(self, candidate_rel: str) -> float:
        """
        计算出现频率 S2
        公式(3): S2 = Count(rc) / ΣCount(rp)

        衡量候选关系在历史数据中的统计稳健性
        """
        if self.total_frequency == 0:
            return 0.0
        count_rc = self.relation_frequencies.get(candidate_rel, 0)
        return count_rc / self.total_frequency

    # --------------------------------------------------------
    # 公式(4): S3 - 时间相关性
    # --------------------------------------------------------
    def compute_temporal_correlation(self, target_rel: str,
                                     candidate_rel: str) -> float:
        """
        计算时间相关性 S3
        公式(4): S3 = 1/Δt
        Δt是候选关系与目标关系在历史路径中出现时间的平均间隔

        间隔越小 → S3越大 → 时间相关性越强
        """
        times_target = self.relation_times.get(target_rel, [])
        times_candidate = self.relation_times.get(candidate_rel, [])

        if not times_target or not times_candidate:
            return 0.01  # 极低的时间相关性

        mean_interval = self.normalizer.compute_mean_interval(
            times_target, times_candidate
        )

        if mean_interval == float("inf") or mean_interval == 0:
            return 0.01

        return 1.0 / mean_interval

    # --------------------------------------------------------
    # 公式(5): S = w1*S1 + w2*S2 + w3*S3
    # --------------------------------------------------------
    def compute_composite_score(self, target_rel: str,
                                 candidate_rel: str) -> float:
        """
        三维加权综合得分
        公式(5): S = w1*S1 + w2*S2 + w3*S3

        w1=0.5 (语义权重最高: 确保关系语义一致)
        w2=0.3 (频率权重中等: 统计稳健性)
        w3=0.2 (时间权重较低: 保持时序逻辑)
        """
        s1 = self.compute_semantic_relevance(target_rel, candidate_rel)
        s2 = self.compute_occurrence_frequency(candidate_rel)
        s3 = self.compute_temporal_correlation(target_rel, candidate_rel)

        # 对S3进行归一化 (因为S3值域可能很大)
        s3_normalized = min(s3, 0.5) * 2  # 缩放到[0,1]

        score = W_SEMANTIC * s1 + W_FREQUENCY * s2 + W_TEMPORAL * s3_normalized
        return score

    # --------------------------------------------------------
    # 公式(6): w(t) = exp(-λ·|t - T|)
    # --------------------------------------------------------
    def compute_time_decay_weight(self, t_edge: float, t_target: float) -> float:
        """
        时间衰减权重
        公式(6): w(t) = exp(-λ·|t - T|)

        边的时间戳越接近目标事件 → 权重越大
        """
        return self.normalizer.time_decay_weight(t_edge, t_target, LAMBDA_DECAY)

    # --------------------------------------------------------
    # 核心采样过程 (第4.1节)
    # --------------------------------------------------------
    def sample_paths(self, target_relation: str, target_time: str,
                     num_paths: int = 10, path_length: int = None) -> List[List[Quadruple]]:
        """
        LLM引导的时序路径采样

        Args:
            target_relation: 目标关系 rh (用于提取时序规则)
            target_time: 目标事件时间戳
            num_paths: 采样的路径数量
            path_length: 路径长度 (默认MAX_PATH_LENGTH)

        Returns:
            List[List[Quadruple]]: 每条路径是四元组序列
        """
        if path_length is None:
            path_length = MAX_PATH_LENGTH

        t_target = Quadruple._parse_time(target_time)
        sampled_paths = []

        # Step a: 采样规则头边 — O(1) 索引查询
        candidate_heads = [q for q in self.dataset.get_quads_by_relation(target_relation)
                           if q.time_val < t_target]

        if not candidate_heads:
            return sampled_paths

        selected_heads = np.random.choice(
            candidate_heads,
            size=min(num_paths, len(candidate_heads)),
            replace=False
        )

        for head_quad in selected_heads:
            path = [head_quad]
            current_entity = head_quad.subject
            current_time = head_quad.time_val

            for step in range(path_length - 1):
                # Step b: O(1) 索引查询当前实体的出边
                adjacent_quads = [q for q in self.dataset.get_quads_by_subject(current_entity)
                                  if q.time_val < t_target]

                if not adjacent_quads:
                    break

                # 限制候选数: 大数据集下实体可能有数百条出边, 采样 50 条即可
                if len(adjacent_quads) > 50:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(adjacent_quads), size=50, replace=False)
                    adjacent_quads = [adjacent_quads[i] for i in idx]

                scored_candidates = []
                for aq in adjacent_quads:
                    s_score = self.compute_composite_score(target_relation, aq.relation)
                    # 结合时间衰减权重 (公式(6))
                    time_weight = self.compute_time_decay_weight(aq.time_val, t_target)
                    combined = s_score * time_weight
                    scored_candidates.append((aq, combined))

                # 按得分降序排序
                scored_candidates.sort(key=lambda x: x[1], reverse=True)

                # 选择得分最高的边 (或带概率采样)
                if scored_candidates:
                    # 使用softmax概率采样top候选
                    top_k = min(5, len(scored_candidates))
                    top_candidates = scored_candidates[:top_k]
                    scores = np.array([s[1] for s in top_candidates])
                    probs = np.exp(scores) / np.sum(np.exp(scores))

                    chosen_idx = np.random.choice(len(top_candidates), p=probs)
                    chosen_quad, _ = top_candidates[chosen_idx]

                    # Step c: 符号约束 - 确保时序顺序
                    if chosen_quad.time_val >= current_time:
                        path.append(chosen_quad)
                        current_entity = chosen_quad.object
                        current_time = chosen_quad.time_val
                    else:
                        # 寻找满足时序约束的下一个候选
                        for aq, sc in scored_candidates:
                            if aq.time_val >= current_time:
                                path.append(aq)
                                current_entity = aq.object
                                current_time = aq.time_val
                                break

            if len(path) >= 2:
                sampled_paths.append(path)

        # Step e: LLM逻辑验证 (模拟)
        validated_paths = self._llm_validate_paths(sampled_paths, target_relation)

        return validated_paths

    # --------------------------------------------------------
    # LLM 路径验证 (真实 API + 模拟回退)
    # --------------------------------------------------------
    def _llm_validate_paths(self, paths: List[List[Quadruple]],
                            target_relation: str) -> List[List[Quadruple]]:
        """LLM驱动的路径逻辑验证 (第4.1节末尾)"""
        if not paths:
            return []

        # ---- 真实 LLM 调用 ----
        if is_real_llm_enabled() and get_ablation_mode() != "w/o G":
            llm_validated = self._llm_api_validate_paths(paths, target_relation)
            if llm_validated is not None:
                return llm_validated
            # API 失败 / 消融 → 回退

        # ---- 模拟 / 回退 ----
        scored_paths = []
        for path in paths:
            semantic_coherence = 0.0
            for i in range(len(path) - 1):
                sim = self.sbert.similarity(path[i].relation, path[i + 1].relation)
                semantic_coherence += sim
            semantic_coherence /= max(1, len(path) - 1)

            if len(path) >= 2:
                intervals = [path[i + 1].time_val - path[i].time_val
                             for i in range(len(path) - 1)]
                valid_intervals = [1.0 for dt in intervals if 0 < dt < 365]
                temporal_smoothness = np.mean(valid_intervals) if valid_intervals else 0.0
            else:
                temporal_smoothness = 0.0

            unique_entities = len(set(q.subject for q in path) |
                                  set(q.object for q in path))
            diversity = unique_entities / (2 * len(path))

            overall_quality = 0.5 * semantic_coherence + 0.3 * temporal_smoothness + 0.2 * diversity
            scored_paths.append((path, overall_quality))

        scored_paths.sort(key=lambda x: x[1], reverse=True)
        threshold = 0.3
        validated = [p for p, score in scored_paths if score >= threshold]
        return validated[:10]

    def _llm_api_validate_paths(self, paths, target_relation):
        """调用 LLM 验证路径逻辑。失败返回 None。"""
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        # 构建路径描述 (限制数量防止 token 溢出)
        path_lines = []
        for i, path in enumerate(paths[:10]):
            events = " → ".join(
                f"[{q.relation}]({q.subject}→{q.object},t={q.time_val:.0f})"
                for q in path[:4]
            )
            path_lines.append(f"  Path_{i}: {events}")
        path_text = "\n".join(path_lines)

        system = (
            "You are an expert in temporal knowledge graphs. "
            "Your task is to evaluate whether each sampled relational path "
            "is logically coherent with the target relation. "
            "Reply with ONLY a JSON object: "
            '{"valid_paths": [0, 2, 5]} listing indices of valid paths. '
            "No explanation."
        )

        user = (
            f"Target relation: \"{target_relation}\"\n\n"
            f"Candidate paths:\n{path_text}\n\n"
            f"For each path, judge whether the sequence of relations "
            f"forms a logically coherent chain leading to or supporting "
            f"\"{target_relation}\". Return the indices (0-based) of valid paths."
        )

        from ..utils import PerfTimer
        PerfTimer.start("llm_path_validate")
        result = llm_chat(system, user, max_tokens=256, temperature=0.0)
        PerfTimer.stop("llm_path_validate")
        if result is None:
            return None

        # 解析 JSON
        import re
        import json
        try:
            # 尝试直接解析
            data = json.loads(result)
            valid_idx = data.get("valid_paths", [])
        except json.JSONDecodeError:
            # 尝试从文本中提取数字列表
            nums = re.findall(r'\d+', result)
            valid_idx = [int(n) for n in nums if int(n) < len(paths)]

        if not valid_idx:
            return None
        return [paths[i] for i in valid_idx if i < len(paths)]

    # --------------------------------------------------------
    # 基于路径提取候选规则
    # --------------------------------------------------------
    def extract_rules_from_paths(self, paths: List[List[Quadruple]],
                                  target_relation: str) -> List["TemporalRule"]:
        """
        从采样路径中提取候选时序规则
        第4.1节: 长度为l的规则从长度为l+1的路径中推导

        Returns:
            List[TemporalRule]: 候选规则列表 (延迟导入避免循环)
        """
        from ..rules.rule_mining import TemporalRule

        rules = []
        for path in paths:
            if len(path) < 2:
                continue

            # 规则头: 路径的最后一个边的关系 (或目标关系)
            rule_head_rel = target_relation

            # 规则体: 路径中前面的边
            rule_body = [(q.relation,) for q in path[:-1]]
            # 简化: 存储关系序列
            body_rels = [q.relation for q in path[:-1]]

            # 规则时间约束: 路径的时间戳
            path_timestamps = [q.time_val for q in path]
            time_constraints = [(path_timestamps[i], path_timestamps[i + 1])
                                for i in range(len(path_timestamps) - 1)]

            rule = TemporalRule(
                head_relation=rule_head_rel,
                body_relations=body_rels,
                confidence=0.0,  # 稍后在第4.3节计算
                validity_period=30.0,  # 稍后在第4.2节确定
                source_path=path,
                time_constraints=time_constraints,
            )
            rules.append(rule)

        # 去重
        seen_signatures = set()
        unique_rules = []
        for rule in rules:
            sig = rule.signature()
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                unique_rules.append(rule)

        return unique_rules

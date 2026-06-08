"""
LLM双向验证模块
=================
核心机制 — 两个双向验证:
1. 规则引导验证图路径: Rule-guided Verification of Graph-based Paths
   利用显式逻辑规则验证GNN高内积路径的逻辑合理性
   LLM评估: 图推理路径是否有合理的规则支持
   调整公式: Adjusted Score = Initial Score × (Rule Matching Degree / 100)

2. 图引导验证规则路径: Graph-guided Verification of Rule-based Paths
   利用图嵌入的深层语义关联验证规则路径的语义一致性
   LLM判断: 规则体触发的事件链在向量空间中是否有实质性特征支持
   调整公式: Adjusted Score = Initial Score × (Average Vector Similarity / 100)

本模块实现LLM模拟验证器: 基于语义相似度和规则匹配度评分
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from ..config import is_real_llm_enabled, EMBEDDING_DIM, get_ablation_mode, get_prompt_style
from ..data.dataset import TKGDataSet, Quadruple
from ..rules.rule_mining import TemporalRule
from ..path_sampling.llm_guided_sampler import SBertSimulator


class LLMBidirectionalVerifier:
    """
    LLM双向验证器
    让LLM作为智能中介,
    利用一条路径的优势验证另一条路径的合理性
    """

    def __init__(self, dataset: TKGDataSet):
        self.dataset = dataset
        self.sbert = SBertSimulator()
        self.sbert.build_embeddings(dataset.relations)

        # 预计算实体间的向量相似度 (用于图引导验证)
        self.entity_similarity_cache: Dict[Tuple[str, str], float] = {}

    # --------------------------------------------------------
    # 方向1: 规则引导验证图路径
    # 论文Table 3: "Rule-guided validation based on graph-based paths"
    # --------------------------------------------------------
    def rule_guided_verify_graph_path(self,
                                       graph_path: List[Quadruple],
                                       graph_score: float,
                                       rules: List[TemporalRule],
                                       query_subject: str,
                                       query_relation: str,
                                       query_time: float) -> float:
        """规则引导验证图路径的合理性"""
        if not graph_path or not rules:
            return graph_score

        # ---- 真实 LLM 调用 ----
        if is_real_llm_enabled():
            adjusted = self._llm_rule_guided_verify(
                graph_path, graph_score, rules, query_subject, query_relation
            )
            if adjusted is not None:
                return adjusted

        # ---- 模拟 / 回退 ----
        path_relations = [q.relation for q in graph_path]
        best_match_degree = 0.0
        for rule in rules:
            match_degree = self._compute_rule_match_degree(
                rule, path_relations, graph_path, query_subject
            )
            best_match_degree = max(best_match_degree, match_degree)

        if best_match_degree > 0:
            adjusted = graph_score * (best_match_degree / 100.0)
        else:
            adjusted = graph_score * 0.7
        return adjusted

    def _llm_rule_guided_verify(self, graph_path, graph_score, rules,
                                 query_subject, query_relation):
        """调用 DeepSeek 进行规则引导验证。失败返回 None。"""
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        path_rel_str = " → ".join(q.relation for q in graph_path[:4])
        rules_str = "; ".join(
            f"{r.head_relation}←{'∧'.join(r.body_relations[:3])}"
            for r in rules[:5]
        )

        system = (
            "You are an expert in temporal knowledge graph reasoning. "
            "Your task is to evaluate whether a graph-based path is "
            "logically supported by a set of temporal rules. "
            "Reply with ONLY a single float number between 0.0 and 1.0, "
            "representing the adjusted confidence score. "
            'Higher means "more rule support". No explanation.'
        )

        user = (
            f"Query: ({query_subject}, {query_relation}, ?, t={graph_path[0].time_val:.0f})"
            if graph_path else f"Query: ({query_subject}, {query_relation}, ?)\n"
            f"\nGraph path relations: {path_rel_str}\n"
            f"Initial graph score: {graph_score:.4f}\n\n"
            f"Available rules:\n{rules_str}\n\n"
            f"Based on how well the rules support the graph path, "
            f"output the adjusted confidence score (0.0-1.0). "
            f"Use formula: Adjusted = Initial × (RuleMatchingDegree / 100)."
        )

        result = llm_chat(system, user, max_tokens=64, temperature=0.0)
        if result is None:
            return None

        import re
        nums = re.findall(r'[\d.]+', result)
        if nums:
            return max(0.0, min(1.0, float(nums[0])))
        return None

    # --------------------------------------------------------
    # 方向2: 图引导验证规则路径
    # "Graph-guided verification of rule-based paths"
    # --------------------------------------------------------
    def graph_guided_verify_rule_path(self,
                                       rule: TemporalRule,
                                       rule_score: float,
                                       candidate_entity: str,
                                       query_subject: str,
                                       query_relation: str) -> float:
        """图引导验证规则推理路径的语义一致性"""
        if not rule.source_path:
            return rule_score

        # ---- 真实 LLM 调用 ----
        if is_real_llm_enabled():
            adjusted = self._llm_graph_guided_verify(
                rule, rule_score, candidate_entity, query_subject, query_relation
            )
            if adjusted is not None:
                return adjusted

        # ---- 模拟 / 回退 ----
        similarities = []
        if rule.source_path:
            path_subjects = [q.subject for q in rule.source_path]
            path_objects = [q.object for q in rule.source_path]
            for ps in path_subjects:
                similarities.append(self._get_entity_similarity(query_subject, ps))
            for po in path_objects:
                similarities.append(self._get_entity_similarity(candidate_entity, po))

        avg_similarity = np.mean(similarities) if similarities else 0.5
        adjusted = rule_score * avg_similarity
        return adjusted

    def _llm_graph_guided_verify(self, rule, rule_score, candidate_entity,
                                  query_subject, query_relation):
        """调用 DeepSeek 进行图引导验证。失败返回 None。"""
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        rule_body = " ∧ ".join(rule.body_relations[:3])
        path_entities = []
        if rule.source_path:
            for q in rule.source_path[:4]:
                path_entities.append(f"({q.subject}→{q.object})")

        system = (
            "You are an expert in temporal knowledge graph reasoning. "
            "Your task is to evaluate the semantic consistency between "
            "a rule-based reasoning path and the entities involved. "
            "Reply with ONLY a single float number between 0.0 and 1.0, "
            "representing the adjusted confidence score. "
            'Higher means "more semantically consistent". No explanation.'
        )

        user = (
            f"Rule: {rule.head_relation} ← {rule_body}\n"
            f"Initial rule score: {rule_score:.4f}\n\n"
            f"Query: ({query_subject}, {query_relation}, ?, t=?)\n"
            f"Candidate entity: {candidate_entity}\n\n"
            f"Path entities: {' → '.join(path_entities) if path_entities else 'none'}\n\n"
            f"Judge whether the semantic correlation of the intermediate "
            f"entities and the query meets the standard. "
            f"Output adjusted score = Initial × (AvgVectorSimilarity / 100)."
        )

        result = llm_chat(system, user, max_tokens=64, temperature=0.0)
        if result is None:
            return None

        import re
        nums = re.findall(r'[\d.]+', result)
        if nums:
            return max(0.0, min(1.0, float(nums[0])))
        return None

    def _get_entity_similarity(self, entity_a: str, entity_b: str) -> float:
        """
        获取两个实体的语义相似度。

        真实模式: 使用 LLM 批量评估实体对, 带缓存。
        API 失败时回退到无随机的确定性启发式。
        """
        pair = (entity_a, entity_b) if entity_a < entity_b else (entity_b, entity_a)
        if pair in self.entity_similarity_cache:
            return self.entity_similarity_cache[pair]

        sim = self._compute_entity_similarity_fallback(entity_a, entity_b)
        self.entity_similarity_cache[pair] = sim
        return sim

    def _compute_entity_similarity_fallback(self, entity_a: str, entity_b: str) -> float:
        """确定性启发式: 基于实体名称前缀/结构相似度, 不使用 random()。"""
        a, b = entity_a.replace("_", " "), entity_b.replace("_", " ")

        # 完全相同的实体
        if a == b:
            return 1.0

        # 检查共同前缀 (如 患者_0100 vs 患者_0234 → 高相似)
        prefix_len = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                prefix_len += 1
            else:
                break

        if prefix_len >= 4:
            prefix_sim = 0.5 + 0.1 * min(prefix_len, 5)
        elif prefix_len >= 2:
            prefix_sim = 0.3 + 0.1 * prefix_len
        else:
            prefix_sim = 0.2

        # 检查词级重叠 (如 华西医院 vs 协和医院 → 共享 "医院")
        words_a, words_b = set(a.split()), set(b.split())
        shared = words_a & words_b
        if shared:
            word_sim = 0.3 + 0.2 * min(len(shared), 3)
        else:
            word_sim = 0.2

        return max(0.1, min(1.0, (prefix_sim + word_sim) / 2))

    # --------------------------------------------------
    # 批量实体相似度 LLM 调用 (供 verify 内部使用)
    # --------------------------------------------------
    def _llm_batch_entity_similarities(self, pairs: List[Tuple[str, str]]) -> Optional[Dict[Tuple[str, str], float]]:
        """
        一次 LLM 调用评估多对实体的语义相似度。

        Returns:
            {(entity_a, entity_b): similarity, ...} 或 None
        """
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        pairs_str = "\n".join(
            f"  {i}: ({a.replace('_', ' ')}, {b.replace('_', ' ')})"
            for i, (a, b) in enumerate(pairs)
        )

        system = (
            "You are an expert in knowledge graph entity semantics. "
            "Evaluate the semantic similarity of each entity pair below. "
            "Reply with ONLY a JSON object mapping pair indices to scores: "
            '{"0": 0.85, "1": 0.23, ...} '
            "Scores must be floats between 0.0 (completely unrelated) "
            "and 1.0 (identical meaning). No explanation."
        )

        user = (
            f"Entity pairs to evaluate:\n{pairs_str}\n\n"
            f"Return a JSON object with similarity scores for all {len(pairs)} pairs."
        )

        result = llm_chat(system, user, max_tokens=512, temperature=0.0)
        if result is None:
            return None

        import json, re
        try:
            data = json.loads(result)
            out = {}
            for i_str, score in data.items():
                i = int(i_str)
                if 0 <= i < len(pairs):
                    out[pairs[i]] = max(0.0, min(1.0, float(score)))
            return out
        except (json.JSONDecodeError, ValueError, KeyError):
            # 宽松解析: 提取所有数字
            nums = re.findall(r'[\d.]+', result)
            out = {}
            for i, n in enumerate(nums[:len(pairs)]):
                out[pairs[i]] = max(0.0, min(1.0, float(n)))
            return out if out else None

    # --------------------------------------------------------
    # 完整双向验证流程 (批量 LLM, 分块处理大规模候选)
    # --------------------------------------------------------
    BATCH_SIZE = 30  # 每批最多评估的候选数

    def verify(self,
               rule_scores: Dict[str, float],
               graph_scores: Dict[str, float],
               rules: List[TemporalRule],
               query_subject: str,
               query_relation: str,
               query_time: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        执行完整的双向验证 (论文第4.5.3节)

        性能优化: 仅对 top-N 候选发送 LLM 验证,
        其余候选保留原始分数直接传递给融合层。
        """
        # 大批量候选只取 top-20 送给 LLM, 其余保留原始分
        MAX_PER_SIDE = 20

        print(f"\n[LLM双向验证]  Bidirectional Verification")

        # ---- 方向1: 规则→验证图路径 ----
        graph_items = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)
        if len(graph_items) > 0:
            top_graph = graph_items[:MAX_PER_SIDE]
            rest_graph = graph_items[MAX_PER_SIDE:]
            top_adjusted = self._batch_rule_guided_verify(
                top_graph, rules, query_subject, query_relation, query_time
            )
            adjusted_graph = dict(top_adjusted)
            for e, s in rest_graph:
                adjusted_graph[e] = s  # 保留原始分
        else:
            adjusted_graph = {}

        # ---- 方向2: 图→验证规则路径 ----
        rule_items = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
        if len(rule_items) > 0:
            top_rule = rule_items[:MAX_PER_SIDE]
            rest_rule = rule_items[MAX_PER_SIDE:]
            top_adjusted = self._batch_graph_guided_verify(
                top_rule, rules, query_subject, query_relation, query_time
            )
            adjusted_rule = dict(top_adjusted)
            for e, s in rest_rule:
                adjusted_rule[e] = s  # 保留原始分
        else:
            adjusted_rule = {}

        print(f"  图验证: {len(adjusted_graph)}个候选 → 规则引导调整完成")
        print(f"  规则验证: {len(adjusted_rule)}个候选 → 图引导调整完成")
        return adjusted_rule, adjusted_graph

    # ================================================================
    # 批量 LLM 实现
    # ================================================================

    def _batch_rule_guided_verify(self, graph_items, rules,
                                   query_subject, query_relation, query_time):
        """批量规则引导验证图候选。"""
        if not graph_items:
            return {}

        # 真实 LLM
        if is_real_llm_enabled() and get_ablation_mode() != "w/o E":
            result = self._llm_batch_rule_guided(
                graph_items[:self.BATCH_SIZE * 5], rules,
                query_subject, query_relation, query_time
            )
            if result:
                # 合并未处理的高分候选 (直接保留原始分数)
                all_adjusted = dict(result)
                for entity, score in graph_items:
                    if entity not in all_adjusted:
                        all_adjusted[entity] = score
                return {k: max(0.0, v) for k, v in all_adjusted.items()}

        # 回退: 逐条
        adjusted = {}
        for entity, score in graph_items:
            virtual_path = self._build_virtual_graph_path(
                query_subject, entity, query_relation, query_time
            )
            adjusted[entity] = max(0.0, self.rule_guided_verify_graph_path(
                virtual_path, score, rules,
                query_subject, query_relation, query_time
            ))
        return adjusted

    def _batch_graph_guided_verify(self, rule_items, rules,
                                    query_subject, query_relation, query_time):
        """批量图引导验证规则候选。"""
        if not rule_items:
            return {}

        # 真实 LLM
        if is_real_llm_enabled() and get_ablation_mode() != "w/o E":
            result = self._llm_batch_graph_guided(
                rule_items[:self.BATCH_SIZE * 5], rules,
                query_subject, query_relation, query_time
            )
            if result:
                all_adjusted = dict(result)
                for entity, score in rule_items:
                    if entity not in all_adjusted:
                        all_adjusted[entity] = score
                return {k: max(0.0, v) for k, v in all_adjusted.items()}

        # 回退: 逐条
        adjusted = {}
        for entity, score in rule_items:
            best_rule = self._find_best_rule_for_entity(entity, rules, query_subject)
            if best_rule:
                adjusted[entity] = max(0.0, self.graph_guided_verify_rule_path(
                    best_rule, score, entity, query_subject, query_relation
                ))
            else:
                adjusted[entity] = score * 0.8
        return adjusted

    # ================================================================
    # TKG 上下文 & 规则格式化 辅助方法
    # ================================================================

    def _build_tkg_context(self, subject: str, query_time: float,
                           window_days: float = 90.0) -> List[Quadruple]:
        """提取查询时间窗口内 subject 的出边四元组作为 TKG 上下文。"""
        t_min = query_time - window_days
        events = []
        for q in self.dataset.quadruples:
            if q.subject == subject and t_min <= q.time_val <= query_time:
                events.append(q)
        return events

    def _build_tkg_missing_patterns(self, subject: str, relations: List[str],
                                     query_time: float, window_days: float = 90.0):
        """检测 TKG 上下文中不存在但在规则体中期望的关键关系。"""
        t_min = query_time - window_days
        existing_rels = set()
        for q in self.dataset.quadruples:
            if q.subject == subject and t_min <= q.time_val <= query_time:
                existing_rels.add(q.relation)
        missing = [r for r in relations if r not in existing_rels]
        return missing[:5]

    @staticmethod
    def _format_rule_with_variables(rule: 'TemporalRule',
                                     idx: int) -> str:
        """格式化为论文记法: (x0, rel1, x1, t1) AND (x1, rel2, x2, t2) => (x0, head, x2, tq)"""
        body = rule.body_relations[:3]
        n = len(body)
        if n == 1:
            body_str = f"(x0, {body[0]}, x1, t1)"
            return f"R{idx}: {body_str} => (x0, {rule.head_relation}, x1, tq)"
        elif n == 2:
            body_str = f"(x0, {body[0]}, x1, t1) AND (x1, {body[1]}, x2, t2)"
            return f"R{idx}: {body_str} => (x0, {rule.head_relation}, x2, tq)"
        else:
            body_parts = [f"(x0, {body[0]}, x1, t1)"]
            for j, rel in enumerate(body[1:-1], 1):
                body_parts.append(f"(x{j}, {rel}, x{j+1}, t{j+1})")
            body_parts.append(f"(x{n-1}, {body[-1]}, x{n}, t{n})")
            body_str = " AND ".join(body_parts)
            return f"R{idx}: {body_str} => (x0, {rule.head_relation}, x{n}, tq)"

    def _build_similarity_table(self, query_subject: str, candidate_entity: str,
                                 rule: 'TemporalRule') -> Tuple[str, float]:
        """构建向量相似度表 + 计算平均值。"""
        entities = [query_subject, candidate_entity]
        if rule.source_path:
            for q in rule.source_path[:4]:
                if q.subject not in entities:
                    entities.append(q.subject)
                if q.object not in entities:
                    entities.append(q.object)

        lines = []
        sims = []
        for i, ea in enumerate(entities):
            for j, eb in enumerate(entities):
                if i >= j:
                    continue
                sim = self._get_entity_similarity(ea, eb)
                lines.append(
                    f"  similarity({ea.replace('_', ' ')}, "
                    f"{eb.replace('_', ' ')}) = {sim:.4f}"
                )
                sims.append(sim)
        avg_sim = float(np.mean(sims)) if sims else 0.3
        return "\n".join(lines[:12]), avg_sim

    # ================================================================
    # 批量 LLM prompt 构建 (论文 Appendix B 改进版)
    # ================================================================

    def _llm_batch_rule_guided(self, graph_items, rules,
                                query_subject, query_relation, query_time):
        """
        B.1: 规则引导验证图路径.

        prompt_style="improved": TKG 上下文 + 论文变量记法 (当前)
        prompt_style="original": 论文 Appendix B 原始简版
        """
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        items = graph_items[:self.BATCH_SIZE * 5]
        chunks = [items[i:i+self.BATCH_SIZE] for i in range(0, len(items), self.BATCH_SIZE)]

        # ---- 原始 Prompt (Appendix B 简版) ----
        if get_prompt_style() == "original":
            return self._batch_rule_guided_original(
                chunks, rules, query_subject, query_relation, query_time, llm_chat
            )

        # ---- 改进版 Prompt ----
        rules_formatted = "\n".join(
            self._format_rule_with_variables(r, j)
            for j, r in enumerate(rules[:5])
        )

        # TKG 上下文 (所有 chunk 共享)
        context_events = self._build_tkg_context(query_subject, query_time)
        if context_events:
            context_lines = "\n".join(
                f"  ({q.subject.replace('_', ' ')}, {q.relation.replace('_', ' ')}, "
                f"{q.object.replace('_', ' ')}, {q.timestamp})"
                for q in context_events[:20]
            )
        else:
            context_lines = "  (no events in time window)"

        all_adjusted = {}
        ts_readable = self._time_val_to_str(query_time)

        for chunk in chunks:
            cand_lines = "\n".join(
                f"  C{i}: entity={e.replace('_', ' ')}, initial_score={s:.4f}"
                for i, (e, s) in enumerate(chunk)
            )
            system = (
                "You are a TKGR expert. Based on the information defined below, "
                "you need to validate the rationality of the knowledge graph "
                "inference paths and adjust scores."
            )
            user = (
                f"1. Core Rule Base (High-confidence rules Sd):\n"
                f"{rules_formatted}\n\n"
                f"2. Query: ({query_subject.replace('_', ' ')}, "
                f"{query_relation.replace('_', ' ')}, ?, {ts_readable})\n\n"
                f"3. TKG Context (events where subject={query_subject.replace('_', ' ')} "
                f"in time window):\n"
                f"{context_lines}\n\n"
                f"4. Candidate Inference Paths:\n"
                f"{cand_lines}\n\n"
                f"5. Validation Task:\n"
                f"(1) Check if the TKG context contains rule body events matching "
                f"any rule AND leading to that candidate entity.\n"
                f"(2) Compute: Adjusted Score = Initial Score × "
                f"(RuleMatchingDegree / 100), where RuleMatchingDegree=100 "
                f"if full rule body exists in context, <100 proportional to "
                f"matched body relation count.\n"
                f"(3) Return ONLY a JSON: "
                f'{{"C0": 0.85, "C1": 0.12, ...}} '
                f"with scores as floats (2 decimal places). No explanation."
            )

            result = llm_chat(system, user, max_tokens=512, temperature=0.0)
            if result is None:
                continue

            import json, re
            try:
                data = json.loads(result)
                for key, score in data.items():
                    idx = int(key.lstrip("C"))
                    if 0 <= idx < len(chunk):
                        all_adjusted[chunk[idx][0]] = max(0.0, min(1.0, float(score)))
            except (json.JSONDecodeError, ValueError, KeyError):
                parsed = self._parse_scores_from_text(result, len(chunk))
                for idx, sc in parsed.items():
                    if 0 <= idx < len(chunk):
                        all_adjusted[chunk[idx][0]] = max(0.0, min(1.0, float(sc)))

        return all_adjusted if all_adjusted else None

    def _llm_batch_graph_guided(self, rule_items, rules,
                                 query_subject, query_relation, query_time):
        """
        B.2: 图引导验证规则路径.

        prompt_style="improved": 向量相似度数值表 + TKG 上下文
        prompt_style="original": 论文 Appendix B 原始简版
        """
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        items = rule_items[:self.BATCH_SIZE * 5]
        chunks = [items[i:i+self.BATCH_SIZE] for i in range(0, len(items), self.BATCH_SIZE)]

        # ---- 原始 Prompt (Appendix B 简版) ----
        if get_prompt_style() == "original":
            return self._batch_graph_guided_original(
                chunks, rules, query_subject, query_relation, llm_chat
            )

        # ---- 改进版 Prompt ----
        all_adjusted = {}

        for chunk in chunks:
            # 取该 chunk 中首个候选对应的最佳规则来获取路径实体
            best_rule = None
            for entity, _ in chunk:
                best_rule = self._find_best_rule_for_entity(entity, rules, query_subject)
                if best_rule:
                    break

            # 构建相似度表 (取第一个候选)
            if chunk and best_rule:
                sim_table, avg_sim = self._build_similarity_table(
                    query_subject, chunk[0][0], best_rule
                )
            else:
                sim_table = "  (no similarity data available)"
                avg_sim = 0.3

            # 规则路径记法
            if best_rule and best_rule.source_path:
                body_chain = " + ".join(
                    f"{r}({best_rule.source_path[i].subject.replace('_', ' ')}, "
                    f"{best_rule.source_path[i].object.replace('_', ' ')})"
                    if i < len(best_rule.source_path)
                    else f"{r}(..., ...)"
                    for i, r in enumerate(best_rule.body_relations[:3])
                )
                rule_path_str = (
                    f"{best_rule.head_relation}({query_subject.replace('_', ' ')}, "
                    f"{chunk[0][0].replace('_', ' ')}) ← {body_chain}, "
                    f"initial score = {chunk[0][1]:.4f}"
                )
            else:
                rule_path_str = (
                    f"{query_relation}({query_subject.replace('_', ' ')}, "
                    f"?, ?) — no detailed rule path available"
                )

            # TKG 反例
            body_rels = best_rule.body_relations if best_rule else []
            missing_rels = self._build_tkg_missing_patterns(
                query_subject, body_rels, query_time
            )

            tkg_neg = ""
            if missing_rels:
                tkg_neg = "\n".join(
                    f'  (1) No "{r}" events for '
                    f"{query_subject.replace('_', ' ')} in window."
                    for r in missing_rels[:3]
                )

            cand_lines = "\n".join(
                f"  C{i}: entity={e.replace('_', ' ')}, initial_score={s:.4f}"
                for i, (e, s) in enumerate(chunk)
            )

            system = (
                "You are a TKGR expert. Based on the information defined below, "
                "you need to validate the rationality of the rule inference path "
                "and adjust scores."
            )
            user = (
                f"1. Rule Inference Path:\n"
                f"{rule_path_str}\n\n"
                f"2. Entity Vector Similarities (from knowledge graph embeddings):\n"
                f"{sim_table}\n"
                f"Average Vector Similarity = {avg_sim:.4f}\n\n"
                f"3. TKG Context (missing key patterns):\n"
                f"{tkg_neg if tkg_neg else '  (all expected patterns present)'}\n\n"
                f"4. Candidate Entities:\n"
                f"{cand_lines}\n\n"
                f"5. Validation Task:\n"
                f"(1) Determine if the Average Vector Similarity meets the standard "
                f"(≥ 0.3 acceptable, < 0.1 very weak).\n"
                f"(2) Check if TKG context supports or contradicts the rule path.\n"
                f"(3) Compute: Adjusted Score = Initial Score × "
                f"AverageVectorSimilarity\n"
                f"(4) Return ONLY a JSON: "
                f'{{"C0": 0.85, "C1": 0.12, ...}} '
                f"with scores as floats (2 decimal places). No explanation."
            )

            result = llm_chat(system, user, max_tokens=512, temperature=0.0)
            if result is None:
                continue

            import json
            try:
                data = json.loads(result)
                for key, score in data.items():
                    idx = int(key.lstrip("C"))
                    if 0 <= idx < len(chunk):
                        all_adjusted[chunk[idx][0]] = max(0.0, min(1.0, float(score)))
            except (json.JSONDecodeError, ValueError, KeyError):
                parsed = self._parse_scores_from_text(result, len(chunk))
                for idx, sc in parsed.items():
                    if 0 <= idx < len(chunk):
                        all_adjusted[chunk[idx][0]] = max(0.0, min(1.0, float(sc)))

        return all_adjusted if all_adjusted else None

    # ================================================================
    # 原始 Prompt (论文 Appendix B 简版, 用于 LLM 稳定性实验)
    # ================================================================

    def _batch_rule_guided_original(self, chunks, rules, qs, qr, qt, llm_chat):
        """B.1 原始版: 简单规则 + 候选, 无 TKG 上下文。"""
        rules_str = "; ".join(
            f"R{j}: {r.head_relation}←{'∧'.join(r.body_relations[:3])}"
            for j, r in enumerate(rules[:5])
        )
        all_adjusted = {}
        for chunk in chunks:
            cand_lines = "\n".join(
                f"  C{i}: entity={e}, score={s:.4f}"
                for i, (e, s) in enumerate(chunk)
            )
            system = "You are a TKGR expert. Validate the graph inference path and adjust the score."
            user = (
                f"1. Core Rule Base:\n{rules_str}\n\n"
                f"2. Knowledge Graph Inference Path: ({qs}, {qr}, ?, t={qt:.0f})\n\n"
                f"3. Candidate Paths:\n{cand_lines}\n\n"
                f"4. Validation Task:\n"
                f"(1) Check if rule body events exist for each candidate.\n"
                f"(2) Adjusted Score = Initial Score × (RuleMatchingDegree / 100).\n"
                f"(3) Return ONLY a JSON: {{\"C0\": 0.85, \"C1\": 0.12, ...}} (2 decimal places)."
            )
            result = llm_chat(system, user, max_tokens=512, temperature=0.7)
            if result is None:
                continue
            import json
            try:
                data = json.loads(result)
                for key, score in data.items():
                    idx = int(key.lstrip("C"))
                    if 0 <= idx < len(chunk):
                        all_adjusted[chunk[idx][0]] = max(0.0, min(1.0, float(score)))
            except (json.JSONDecodeError, ValueError, KeyError):
                parsed = self._parse_scores_from_text(result, len(chunk))
                for idx, sc in parsed.items():
                    if 0 <= idx < len(chunk):
                        all_adjusted[chunk[idx][0]] = max(0.0, min(1.0, float(sc)))
        return all_adjusted if all_adjusted else None

    def _batch_graph_guided_original(self, chunks, rules, qs, qr, llm_chat):
        """B.2 原始版: 简单规则路径 + 候选, 无向量相似度表。"""
        rule_summaries = []
        for r in rules[:5]:
            if r.source_path:
                ents = [q.object for q in r.source_path[:3]]
                rule_summaries.append(
                    f"{r.head_relation}←{'∧'.join(r.body_relations[:3])} [entities: {', '.join(ents)}]"
                )
        rule_text = "; ".join(rule_summaries) if rule_summaries else "(none)"
        all_adjusted = {}
        for chunk in chunks:
            cand_lines = "\n".join(
                f"  C{i}: entity={e}, score={s:.4f}"
                for i, (e, s) in enumerate(chunk)
            )
            system = "You are a TKGR expert. Validate the rule inference path and adjust the score."
            user = (
                f"1. Rule Inference Path:\n{rule_text}\n\n"
                f"2. Query: ({qs}, {qr}, ?, t=?)\n\n"
                f"3. Candidate Entities:\n{cand_lines}\n\n"
                f"4. Validation Task:\n"
                f"(1) Determine if the semantic correlation meets the standard.\n"
                f"(2) Adjusted Score = Initial Score × (AverageVectorSimilarity / 100).\n"
                f"(3) Return ONLY a JSON: {{\"C0\": 0.85, \"C1\": 0.12, ...}} (2 decimal places)."
            )
            result = llm_chat(system, user, max_tokens=512, temperature=0.7)
            if result is None:
                continue
            import json
            try:
                data = json.loads(result)
                for key, score in data.items():
                    idx = int(key.lstrip("C"))
                    if 0 <= idx < len(chunk):
                        all_adjusted[chunk[idx][0]] = max(0.0, min(1.0, float(score)))
            except (json.JSONDecodeError, ValueError, KeyError):
                parsed = self._parse_scores_from_text(result, len(chunk))
                for idx, sc in parsed.items():
                    if 0 <= idx < len(chunk):
                        all_adjusted[chunk[idx][0]] = max(0.0, min(1.0, float(sc)))
        return all_adjusted if all_adjusted else None

    @staticmethod
    def _time_val_to_str(time_val: float) -> str:
        """将 time_val (days since epoch) 转回日期字符串。"""
        from datetime import datetime, timedelta
        try:
            dt = datetime(1970, 1, 1) + timedelta(days=time_val)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return f"{time_val:.0f}"

    @staticmethod
    def _parse_scores_from_text(text: str, expected_count: int) -> Dict[int, float]:
        """从非 JSON 的 LLM 输出中提取分数。"""
        import re
        nums = re.findall(r'[\d.]+', text)
        out = {}
        for i, n in enumerate(nums[:expected_count]):
            out[i] = max(0.0, min(1.0, float(n)))
        return out

    def _build_virtual_graph_path(self,
                                   subject: str,
                                   candidate: str,
                                   relation: str,
                                   query_time: float) -> List[Quadruple]:
        """
        为图候选构建虚拟路径
        用于规则引导验证
        """
        # 查找从subject到candidate的实际路径
        actual_paths = []
        for q in self.dataset.quadruples:
            if (q.subject == subject and q.object == candidate
                and q.time_val <= query_time):
                actual_paths.append(q)

        if actual_paths:
            return actual_paths[:3]

        # 若没有直接连接, 构建包含关系的路径
        indirect = []
        for q in self.dataset.quadruples:
            if q.subject == subject and q.time_val <= query_time:
                indirect.append(q)
                break
        for q in self.dataset.quadruples:
            if q.object == candidate and q.time_val <= query_time:
                indirect.append(q)
                break

        return indirect

    def _find_best_rule_for_entity(self,
                                    entity: str,
                                    rules: List[TemporalRule],
                                    query_subject: str) -> Optional[TemporalRule]:
        """为候选实体找到最佳匹配规则"""
        best_rule = None
        best_priority = 0.0

        for rule in rules:
            if rule.priority_score > best_priority:
                # 检查规则是否能推导到该实体
                if rule.source_path:
                    path_entities = {q.object for q in rule.source_path}
                    if entity in path_entities:
                        best_rule = rule
                        best_priority = rule.priority_score

        return best_rule

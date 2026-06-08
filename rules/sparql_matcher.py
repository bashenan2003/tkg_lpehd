"""
SPARQL时序事件模式提取模块 (升级版)
=====================================
指定的时序事件匹配规则

本模块支持两种后端:
1. rdflib 原版SPARQL引擎 (推荐, W3C标准兼容)
2. 内存RDFTripleStore回退模式 (rdflib不可用时)

通过Neo4jStore的export_to_rdflib()获得标准RDF图后,
直接使用W3C标准SPARQL 1.1查询定义的6种时序事件模式。

事件模式:
- 外交声明的时序级联: Make_statement → Make_an_appeal_or_request → Make_a_visit
- 冲突升级模式: Accuse → Criticize_or_denounce → Protest_or_demonstrate
- 合作推进模式: Express_intent_to_cooperate → Engage_in_negotiation → Sign_formal_agreement
- 援助承诺与执行: Express_intent_to_provide_aid → Provide_economic_aid → Cooperate_on_economic_issues
- 调解与协议: Mediate → Engage_in_negotiation → Ratify_treaty
- 道歉-访问-合作链: Apologize → Make_a_visit → Engage_in_diplomatic_cooperation
"""

import os
import json
from typing import List, Dict, Tuple, Set, Optional

from collections import defaultdict

from ..config import SPARQL_PREFIX, DATA_DIR, OUTPUT_DIR, is_real_llm_enabled
from ..data.dataset import Quadruple

# rdflib (用于原版SPARQL)
try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import RDF, XSD
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False


class RDFTripleStore:
    """
    本地RDF三元组内存存储 (回退模式)
    使用内存字典模拟RDF图,
    支持类SPARQL的查询模式匹配
    """

    def __init__(self):
        self.triples: List[Dict] = []
        self.subject_index: Dict[str, List[int]] = {}
        self.relation_index: Dict[str, List[int]] = {}
        self.object_index: Dict[str, List[int]] = {}
        self.time_index: Dict[str, List[int]] = {}
        self.subj_rel_index: Dict[Tuple[str, str], List[int]] = {}  # O(1) 查找

    def add_quadruple(self, s: str, r: str, o: str, t: str):
        """添加四元组到RDF存储"""
        idx = len(self.triples)
        entry = {"subject": s, "relation": r, "object": o, "timestamp": t}
        self.triples.append(entry)

        self.subject_index.setdefault(s, []).append(idx)
        self.relation_index.setdefault(r, []).append(idx)
        self.object_index.setdefault(o, []).append(idx)
        self.time_index.setdefault(t, []).append(idx)
        self.subj_rel_index.setdefault((s, r), []).append(idx)

    def get_by_subj_rel(self, s: str, r: str) -> List[Dict]:
        """O(1) 索引查询: 获取某 subject 的某 relation 所有事件。"""
        indices = self.subj_rel_index.get((s, r), [])
        return [self.triples[i] for i in indices]

    def load_from_dataset(self, quadruples: List) -> int:
        """从数据集加载四元组"""
        for q in quadruples:
            if hasattr(q, 'to_tuple'):
                s, r, o, t = q.to_tuple()
            else:
                s, r, o, t = q
            self.add_quadruple(s, r, o, t)
        return len(self.triples)

    def query(self, conditions: Dict) -> List[Dict]:
        """
        简化的SPARQL查询接口
        支持: subject, relation, object, timestamp 的条件匹配
        支持时序约束: time_before, time_after, time_between
        """
        results = []

        for idx, triple in enumerate(self.triples):
            match = True
            for key, value in conditions.items():
                if key == "time_before" and triple["timestamp"] >= str(value):
                    match = False
                elif key == "time_after" and triple["timestamp"] <= str(value):
                    match = False
                elif key == "time_between":
                    t_min, t_max = value
                    if not (t_min <= triple["timestamp"] <= t_max):
                        match = False
                elif key in ("time_before", "time_after", "time_between"):
                    continue
                elif triple.get(key) != value:
                    match = False
            if match:
                results.append(triple)
        return results


class SPARQLTemporalMatcher:
    """
    SPARQL时序事件模式匹配器

    两种工作模式:
    1. rdflib原版SPARQL (传入rdf_graph参数)
    2. 内存回退模式 (传入rdf_store参数)

    SPARQL查询严格匹配定义的事件模式,
    使用W3C标准SPARQL 1.1语法。
    """

    # ICEWS 硬编码回退 (数据集不可用或无频繁模式时使用)
    FALLBACK_CHAINS = {
        "diplomatic_cascade": (["Make_statement", "Make_an_appeal_or_request", "Make_a_visit"], 60),
        "conflict_escalation": (["Accuse", "Criticize_or_denounce", "Protest_or_demonstrate"], 30),
        "cooperation_progression": (["Express_intent_to_cooperate", "Engage_in_negotiation", "Sign_formal_agreement"], 180),
        "aid_commitment": (["Express_intent_to_provide_aid", "Provide_economic_aid", "Cooperate_on_economic_issues"], 90),
        "mediation_treaty": (["Mediate", "Engage_in_negotiation", "Ratify_treaty"], 365),
        "apology_visit_cooperation": (["Apologize", "Make_a_visit", "Engage_in_diplomatic_cooperation"], 90),
    }

    def __init__(self,
                 rdf_graph: 'Graph' = None,
                 rdf_store: RDFTripleStore = None,
                 neo4j_store=None,
                 dataset=None):
        self.rdf_graph = rdf_graph
        self.store = rdf_store or RDFTripleStore()
        self.neo4j_store = neo4j_store
        self.matched_patterns: List[Dict] = []
        self.uses_native_sparql = rdf_graph is not None and RDFLIB_AVAILABLE

        # 动态发现时序模式 (有 dataset 时自动挖掘, 否则回退 ICEWS)
        if dataset is not None:
            self.temporal_chains = self._discover_temporal_chains(dataset)
        else:
            self.temporal_chains = dict(self.FALLBACK_CHAINS)

    # ================================================================
    # 动态时序模式发现 (替代硬编码 ICEWS 链)
    # ================================================================
    @staticmethod
    def _discover_temporal_chains(dataset, top_k: int = 6) -> Dict[str, Tuple[List[str], int]]:
        """
        从数据集中挖掘频繁的 3-关系时序链。

        策略:
        1. 扫描同一 (subject, object) 对上的事件序列, 按时间排序
        2. 统计连续 3-关系子序列的出现频次
        3. 取 top-k 高频链 + LLM 语义筛选

        Returns:
            {pattern_name: ([rel1, rel2, rel3], validity_days), ...}
        """
        quads = dataset.quadruples
        if len(quads) < 100:
            return dict(SPARQLTemporalMatcher.FALLBACK_CHAINS)

        # 按 (subject, object) 分组, 收集每个 pair 上按时间排序的关系序列
        pair_sequences = defaultdict(list)
        for q in quads:
            pair_sequences[(q.subject, q.object)].append((q.time_val, q.relation))

        # 对每个 pair, 按时间排序后提取连续三元组
        chain_counter = defaultdict(int)
        chain_time_spans = defaultdict(list)

        for (s, o), events in pair_sequences.items():
            if len(events) < 3:
                continue
            events.sort(key=lambda x: x[0])
            rels = [r for _, r in events]
            for i in range(len(rels) - 2):
                chain = (rels[i], rels[i + 1], rels[i + 2])
                chain_counter[chain] += 1
                span = events[i + 2][0] - events[i][0]
                chain_time_spans[chain].append(span)

        # 取 top-k 高频链
        top_chains = sorted(chain_counter.items(), key=lambda x: x[1], reverse=True)[:top_k]

        if not top_chains:
            return dict(SPARQLTemporalMatcher.FALLBACK_CHAINS)

        # LLM 语义筛选 (可用时)
        if is_real_llm_enabled() and top_chains:
            approved = SPARQLTemporalMatcher._llm_filter_chains(
                {c: cnt for c, cnt in top_chains}, dataset
            )
            if approved:
                top_chains = [(c, chain_counter[c]) for c in approved if c in chain_counter]

        # 构建 chains 字典
        chains = {}
        for i, (chain, count) in enumerate(top_chains):
            r1, r2, r3 = chain
            spans = chain_time_spans.get(chain, [30.0])
            avg_span = sum(spans) / len(spans)
            name = f"pattern_{i+1}_{r1.replace(' ', '_')}"
            validity = max(7, min(365, int(avg_span * 2)))
            chains[name] = ([r1, r2, r3], validity)

        return chains if chains else dict(SPARQLTemporalMatcher.FALLBACK_CHAINS)

    @staticmethod
    def _llm_filter_chains(chain_counts: Dict[Tuple[str, ...], int], dataset) -> Optional[List[Tuple[str, ...]]]:
        """调用 DeepSeek 筛选语义合理的时序链。失败返回 None (回退到纯频率)。"""
        try:
            from ..llm_client import llm_chat
        except ImportError:
            return None

        chains_str = "\n".join(
            f"  {i}: {r1} → {r2} → {r3} (frequency={cnt})"
            for i, ((r1, r2, r3), cnt) in enumerate(chain_counts.items())
        )
        system = (
            "You are an expert in temporal event pattern analysis. "
            "Evaluate whether each 3-relation chain forms a logically "
            "meaningful temporal sequence. Eliminate chains that are "
            "random noise or semantically incoherent. "
            'Reply with ONLY a JSON list of indices to KEEP: [0, 2, 5]. '
            "No explanation."
        )
        user = (
            f"Dataset domain context: the relations include terms like "
            f"{', '.join(list(dataset.relations)[:20])}\n\n"
            f"Candidate chains:\n{chains_str}\n\n"
            f"Which chains are semantically meaningful? Return indices to keep."
        )

        result = llm_chat(system, user, max_tokens=128, temperature=0.0)
        if result is None:
            return None

        import re, json
        try:
            keep_idx = json.loads(result)
        except json.JSONDecodeError:
            nums = re.findall(r'\d+', result)
            keep_idx = [int(n) for n in nums]

        items = list(chain_counts.keys())
        return [items[i] for i in keep_idx if 0 <= i < len(items)]

    def match_pattern(self, rel_chain: List[str], pattern_name: str,
                      validity_days: int, cross_entity: bool = False) -> List[Dict]:
        """通用模式匹配: 对任意 3-关系链执行 SPARQL 或内存匹配。"""
        r1, r2, r3 = rel_chain
        if self.uses_native_sparql:
            query = self._build_sparql(r1, r2, r3, cross_entity=cross_entity)
            return self._format_results_dynamic(pattern_name, rel_chain, validity_days,
                                                self.execute_native_sparql(query))
        return self._fallback_pattern(rel_chain, pattern_name, validity_days,
                                       cross_entity=cross_entity)

    def _format_results_dynamic(self, pattern_name, rel_chain, validity_days,
                                 bindings: List[Dict]) -> List[Dict]:
        """格式化 SPARQL 结果 (动态链版本)。"""
        r1, r2, r3 = rel_chain
        results = []
        for b in bindings:
            subj = b.get("subject", "").replace("http://tkg.lpehd.org/resource/", "")
            obj = b.get("object", "").replace("http://tkg.lpehd.org/resource/", "")
            inter = b.get("intermediate", "").replace("http://tkg.lpehd.org/resource/", "") if "intermediate" in b else None
            results.append({
                "pattern": pattern_name,
                "subject": subj, "object": obj, "intermediate": inter,
                "events": [(r1, str(b.get("t1", ""))),
                           (r2, str(b.get("t2", ""))),
                           (r3, str(b.get("t3", "")))],
                "validity_days": validity_days,
            })
        return results

    # ================================================================
    # 原版SPARQL查询方法 (W3C标准)
    # ================================================================
    def _build_sparql(self, rel1: str, rel2: str, rel3: str,
                      cross_entity: bool = False) -> str:
        """
        构建标准SPARQL查询

        模式: 查找同一实体对(s, o)之间按时序发生的三个事件
              t1 ≤ t2 ≤ t3

        Table 3 规则1:
          (x0, Make_statement, x1) ∧
          (x0, Make_an_appeal_or_request, x1) →
          (x0, Make_a_visit, x1), V(R)=60
        """
        prefix = """PREFIX tkg: <http://tkg.lpehd.org/ontology/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"""

        if cross_entity:
            # 跨实体模式 (道歉-访问-合作链)
            query = f"""
{prefix}

SELECT ?subject ?intermediate ?object ?t1 ?t2 ?t3 WHERE {{
  ?e1 tkg:subject ?subject ;
      tkg:relation tkg:{rel1} ;
      tkg:object ?object ;
      tkg:timestamp ?t1 .
  ?e2 tkg:subject ?subject ;
      tkg:relation tkg:{rel2} ;
      tkg:object ?intermediate ;
      tkg:timestamp ?t2 .
  ?e3 tkg:subject ?intermediate ;
      tkg:relation tkg:{rel3} ;
      tkg:object ?object ;
      tkg:timestamp ?t3 .
  FILTER(?t1 <= ?t2 && ?t2 <= ?t3)
}}
ORDER BY ?t1
LIMIT 100"""
        else:
            query = f"""
{prefix}

SELECT ?subject ?object ?t1 ?t2 ?t3 WHERE {{
  ?e1 tkg:subject ?subject ;
      tkg:relation tkg:{rel1} ;
      tkg:object ?object ;
      tkg:timestamp ?t1 .
  ?e2 tkg:subject ?subject ;
      tkg:relation tkg:{rel2} ;
      tkg:object ?object ;
      tkg:timestamp ?t2 .
  ?e3 tkg:subject ?subject ;
      tkg:relation tkg:{rel3} ;
      tkg:object ?object ;
      tkg:timestamp ?t3 .
  FILTER(?t1 <= ?t2 && ?t2 <= ?t3)
}}
ORDER BY ?t1
LIMIT 100"""
        return query

    def execute_native_sparql(self, query_str: str) -> List[Dict]:
        """执行原版SPARQL查询"""
        if not self.rdf_graph or not RDFLIB_AVAILABLE:
            return []

        try:
            result = self.rdf_graph.query(query_str)
        except Exception as e:
            print(f"  [SPARQL错误] {e}")
            return []

        bindings = []
        for row in result:
            binding = {}
            for var in result.vars:
                val = row[var]
                if val is None:
                    binding[str(var)] = None
                elif isinstance(val, (URIRef, Literal)):
                    binding[str(var)] = str(val)
                else:
                    binding[str(var)] = str(val)
            bindings.append(binding)
        return bindings

    # ================================================================
    # 6种事件模式匹配
    # ================================================================
    def pattern_diplomatic_cascade(self, entity: str = None,
                                    time_range: Tuple[str, str] = None) -> List[Dict]:
        """
        模式1: 外交声明时序级联
        Table 3, Rule 1: V(R)=60天

        (x0, Make_statement, x1) → (x0, Make_an_appeal_or_request, x1)
        → (x0, Make_a_visit, x1)
        """
        if self.uses_native_sparql:
            query = self._build_sparql(
                "Make_statement", "Make_an_appeal_or_request", "Make_a_visit"
            )
            return self._format_results("diplomatic_cascade", 60,
                                         self.execute_native_sparql(query))
        return self._fallback_pattern(
            ["Make_statement", "Make_an_appeal_or_request", "Make_a_visit"],
            "diplomatic_cascade", 60, entity, time_range
        )

    def pattern_conflict_escalation(self, entity: str = None) -> List[Dict]:
        """模式2: 冲突升级 (Accuse → Criticize → Protest)"""
        if self.uses_native_sparql:
            query = self._build_sparql("Accuse", "Criticize_or_denounce",
                                        "Protest_or_demonstrate")
            return self._format_results("conflict_escalation", 30,
                                         self.execute_native_sparql(query))
        return self._fallback_pattern(
            ["Accuse", "Criticize_or_denounce", "Protest_or_demonstrate"],
            "conflict_escalation", 30
        )

    def pattern_cooperation_progression(self) -> List[Dict]:
        """模式3: 合作推进 (Intent → Negotiate → Sign)"""
        if self.uses_native_sparql:
            query = self._build_sparql(
                "Express_intent_to_cooperate", "Engage_in_negotiation",
                "Sign_formal_agreement"
            )
            return self._format_results("cooperation_progression", 180,
                                         self.execute_native_sparql(query))
        return self._fallback_pattern(
            ["Express_intent_to_cooperate", "Engage_in_negotiation",
             "Sign_formal_agreement"],
            "cooperation_progression", 180
        )

    def pattern_aid_commitment(self) -> List[Dict]:
        """模式4: 援助承诺与执行 (Intent_Aid → Economic_Aid → Cooperate)"""
        if self.uses_native_sparql:
            query = self._build_sparql(
                "Express_intent_to_provide_aid", "Provide_economic_aid",
                "Cooperate_on_economic_issues"
            )
            return self._format_results("aid_commitment", 90,
                                         self.execute_native_sparql(query))
        return self._fallback_pattern(
            ["Express_intent_to_provide_aid", "Provide_economic_aid",
             "Cooperate_on_economic_issues"],
            "aid_commitment", 90
        )

    def pattern_mediation_treaty(self) -> List[Dict]:
        """模式5: 调解与协议 (Mediate → Negotiate → Ratify)"""
        if self.uses_native_sparql:
            query = self._build_sparql("Mediate", "Engage_in_negotiation",
                                        "Ratify_treaty")
            return self._format_results("mediation_treaty", 365,
                                         self.execute_native_sparql(query))
        return self._fallback_pattern(
            ["Mediate", "Engage_in_negotiation", "Ratify_treaty"],
            "mediation_treaty", 365
        )

    def pattern_apology_visit_cooperation(self) -> List[Dict]:
        """
        模式6: 道歉-访问-合作链 (Table 3, Rule 2): V(R)=15

        (x0, Apologize, x1) ∧ (x1, Make_a_visit, x2)
        → (x0, Make_a_visit, x2)
        """
        if self.uses_native_sparql:
            query = self._build_sparql(
                "Apologize", "Make_a_visit",
                "Engage_in_diplomatic_cooperation", cross_entity=True
            )
            return self._format_results("apology_visit_cooperation", 15,
                                         self.execute_native_sparql(query))
        return self._fallback_pattern(
            ["Apologize", "Make_a_visit", "Engage_in_diplomatic_cooperation"],
            "apology_visit_cooperation", 90,
            cross_entity=True
        )

    # ================================================================
    # 内存回退模式
    # ================================================================
    def _fallback_pattern(self, rel_chain: List[str], pattern_name: str,
                           validity_days: int, entity: str = None,
                           time_range: Tuple[str, str] = None,
                           cross_entity: bool = False) -> List[Dict]:
        """内存回退: O(1) 索引查找, 最多 100 条结果"""
        results = []
        r1, r2, r3 = rel_chain

        # Step 1: 遍历 r1 的 (subject, object) 对
        r1_events = self.store.query({"relation": r1})
        if entity:
            r1_events = [e for e in r1_events if e["subject"] == entity]

        seen_pairs = set()
        for e1 in r1_events:
            s, o1, t1 = e1["subject"], e1["object"], e1["timestamp"]

            # Step 2: 检查该 (subject, r2) 下是否有匹配事件 — O(1)
            r2_candidates = self.store.get_by_subj_rel(s, r2)
            r2_matches = [e for e in r2_candidates
                          if e["object"] == o1 and e["timestamp"] >= t1]

            for e2 in r2_matches:
                t2 = e2["timestamp"]
                inter = e2["object"]

                # Step 3: 检查 (subject, r3) — O(1)
                r3_candidates = self.store.get_by_subj_rel(s, r3)
                r3_matches = [e for e in r3_candidates
                              if e["object"] == o1 and e["timestamp"] >= t2]

                for e3 in r3_matches[:5]:  # 每组最多5条
                    key = (s, o1, t1, t2)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    results.append({
                        "pattern": pattern_name,
                        "subject": s, "object": o1,
                        "intermediate": inter if cross_entity else None,
                        "events": [(r1, t1), (r2, t2), (r3, e3["timestamp"])],
                        "validity_days": validity_days,
                    })
                    if len(results) >= 100:
                        return results

        return results

    def _format_results(self, pattern_name: str, validity_days: int,
                         bindings: List[Dict]) -> List[Dict]:
        """格式化SPARQL查询结果为标准结构"""
        results = []
        for b in bindings:
            subj = b.get("subject", "").replace(f"http://tkg.lpehd.org/resource/", "")
            obj = b.get("object", "").replace(f"http://tkg.lpehd.org/resource/", "")
            inter = b.get("intermediate", "").replace(f"http://tkg.lpehd.org/resource/", "") if "intermediate" in b else None
            t1 = b.get("t1", "")
            t2 = b.get("t2", "")
            t3 = b.get("t3", "")

            rel_map = {
                "diplomatic_cascade": ("Make_statement", "Make_an_appeal_or_request", "Make_a_visit"),
                "conflict_escalation": ("Accuse", "Criticize_or_denounce", "Protest_or_demonstrate"),
                "cooperation_progression": ("Express_intent_to_cooperate", "Engage_in_negotiation", "Sign_formal_agreement"),
                "aid_commitment": ("Express_intent_to_provide_aid", "Provide_economic_aid", "Cooperate_on_economic_issues"),
                "mediation_treaty": ("Mediate", "Engage_in_negotiation", "Ratify_treaty"),
                "apology_visit_cooperation": ("Apologize", "Make_a_visit", "Engage_in_diplomatic_cooperation"),
            }
            r1, r2, r3 = rel_map.get(pattern_name, ("", "", ""))

            results.append({
                "pattern": pattern_name,
                "subject": subj,
                "object": obj,
                "intermediate": inter,
                "events": [(r1, str(t1)), (r2, str(t2)), (r3, str(t3))],
                "validity_days": validity_days,
            })
        return results

    # ================================================================
    # 批量模式匹配
    # ================================================================
    def match_all_patterns(self) -> Dict[str, List[Dict]]:
        """
        执行时序事件模式匹配。

        大数据集 (>5000 RDF 三元组) 自动切换内存回退,
        避免 rdflib SPARQL 3-way JOIN 的 O(N³) 性能问题。
        """
        # rdflib SPARQL 在 3-way JOIN + FILTER 上是 O(N³), 大数据集直接走内存
        if self.uses_native_sparql:
            rdf_size = len(self.rdf_graph) if self.rdf_graph else 0
            if rdf_size > 500:
                self.uses_native_sparql = False

        backend_label = "rdflib原版SPARQL" if self.uses_native_sparql else "内存回退"
        print(f"[SPARQL匹配] 后端: {backend_label} (RDF triples={len(self.rdf_graph) if self.rdf_graph else 0})")

        all_matches = {}
        self.matched_patterns = []
        total = 0

        for pattern_name, (rel_chain, validity_days) in self.temporal_chains.items():
            cross = (pattern_name == "apology_visit_cooperation")
            matches = self.match_pattern(rel_chain, pattern_name, validity_days, cross_entity=cross)
            all_matches[pattern_name] = matches
            total += len(matches)
            self.matched_patterns.extend(matches)

        print(f"[SPARQL匹配] 共匹配 {total} 个时序事件模式:")
        for name, matches in all_matches.items():
            if matches:
                rels = self.temporal_chains.get(name, ([], 0))[0]
                chain_str = " → ".join(rels)
                print(f"  {name} [{chain_str}]: {len(matches)} 个")

        return all_matches

    # ================================================================
    # 导出
    # ================================================================
    def export_matches(self, output_dir: str = None):
        """导出SPARQL匹配结果"""
        if output_dir is None:
            output_dir = OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, "sparql_matches.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.matched_patterns, f, indent=2,
                      ensure_ascii=False, default=str)
        print(f"[SPARQL] 匹配结果已导出: {path}")


def build_sparql_store(dataset) -> RDFTripleStore:
    """从TKG数据集构建RDF三元组存储 (内存回退模式)"""
    store = RDFTripleStore()
    count = store.load_from_dataset(dataset.quadruples)
    print(f"[SPARQL] 内存RDF存储: {count} 条四元组")
    return store


def build_native_sparql_matcher(dataset) -> SPARQLTemporalMatcher:
    """
    构建原版SPARQL匹配器 (使用rdflib)

    将数据集转换为标准RDF图,
    使所有SPARQL查询符合W3C标准。
    """
    if not RDFLIB_AVAILABLE:
        print("[SPARQL] rdflib未安装, 回退到内存模式. 运行: pip install rdflib")
        store = build_sparql_store(dataset)
        return SPARQLTemporalMatcher(rdf_store=store, dataset=dataset)

    from rdflib import Graph, Namespace
    from rdflib.namespace import RDF

    from ..utils import safe_xsd_date_literal

    # 内存 store 始终构建 (全量数据, 快速匹配)
    mem_store = build_sparql_store(dataset)

    # RDF 图仅采样 2000 条 (大数据集全量构建太慢, 仅作小数据回退)
    import random
    quads_sample = list(dataset.quadruples)
    if len(quads_sample) > 2000:
        quads_sample = random.sample(quads_sample, 2000)

    g = Graph()
    TKG = Namespace("http://tkg.lpehd.org/ontology/")
    RES = Namespace("http://tkg.lpehd.org/resource/")
    g.bind("tkg", TKG)
    g.bind("xsd", Namespace("http://www.w3.org/2001/XMLSchema#"))

    for idx, q in enumerate(quads_sample):
        s, r, o, t = q.subject, q.relation, q.object, q.timestamp
        s_clean = s.replace('"', '').replace(' ', '_')
        o_clean = o.replace('"', '').replace(' ', '_')
        r_clean = r.replace('"', '').replace(' ', '_')
        event_uri = RES[f"Event_{idx}"]
        g.add((event_uri, RDF.type, TKG.Event))
        g.add((event_uri, TKG.subject, RES[s_clean]))
        g.add((event_uri, TKG.relation, TKG[r_clean]))
        g.add((event_uri, TKG.object, RES[o_clean]))
        g.add((event_uri, TKG.timestamp, safe_xsd_date_literal(t)))

    print(f"[SPARQL] rdflib RDF图: {len(g)}三元组 (采样{len(quads_sample)}事件), 内存: {len(mem_store.triples)}事件")
    return SPARQLTemporalMatcher(rdf_graph=g, rdf_store=mem_store, dataset=dataset)

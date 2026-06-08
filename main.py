#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================

主入口程序
流水线5个阶段:
  Phase 1 (4.1): LLM引导三维关系路径采样
  Phase 2 (4.2): LLM辅助时序有效性评估
  Phase 3 (4.3): 规则优先级排序
  Phase 4 (4.4): 自适应规则演化
  Phase 5 (4.5): LLM动态权重融合推理

使用方法:
  python main.py                      # 在当前目录直接运行
  python -m tkg_lpehd.main            # 从父目录以模块方式运行
  python main.py --dataset ICEWS14    # 指定数据集
  python main.py --skip-training      # 跳过GNN训练
  python main.py --neo4j              # 启用Neo4j存储
"""

import os
import sys

# ==================================================================
# 启动引导: 支持 `python main.py` 和 `python -m tkg_lpehd.main` 两种运行方式
# ==================================================================
if __name__ == "__main__" and __package__ is None:
    # 直接运行 main.py 时, 设置模块上下文重新启动
    _this_file = os.path.abspath(__file__)
    _pkg_dir = os.path.dirname(_this_file)   # tkg_lpehd/
    _parent = os.path.dirname(_pkg_dir)      # tkg_lpehd的父目录
    sys.path.insert(0, _parent)
    os.chdir(_parent)  # 切到父目录确保数据文件路径正确
    import runpy
    runpy.run_module("tkg_lpehd.main", run_name="__main__", alter_sys=True)
    sys.exit(0)

import time
import argparse

# ------------------------------------------------------------------
# 自动安装依赖检查
# ------------------------------------------------------------------
def _check_dependencies():
    """检查并安装必要的依赖库"""
    required = {
        "numpy": "numpy",
        "torch": "torch",
        "requests": "requests",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"[依赖] 缺少以下库: {missing}")
        print("[依赖] 正在自动安装...")
        import subprocess
        for pkg in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print("[依赖] 安装完成!")


_check_dependencies()

# ------------------------------------------------------------------
# 导入所有TKG-LPEHD模块
# ------------------------------------------------------------------
from .config import (
    DATASET_NAME, LAMBDA_DECAY, DEVICE, CASE_STUDY_QUERIES, print_config
)
from .data.dataset import TKGDataSet, load_or_generate_dataset
from .data.neo4j_store import Neo4jStore, create_neo4j_store, NEO4J_AVAILABLE
from .preprocessing.temporal_normalizer import TemporalNormalizer
from .preprocessing.quadruple_formatter import format_quad_for_llm
from .path_sampling.llm_guided_sampler import ThreeDimensionalPathSampler
from .path_sampling.validity_assessment import TemporalValidityAssessor
from .rules.rule_mining import TemporalRule, RulePriorityRanker
from .rules.rule_evolution import AdaptiveRuleEvolution
from .rules.sparql_matcher import (
    SPARQLTemporalMatcher, RDFTripleStore, build_sparql_store,
    build_native_sparql_matcher
)
from .reasoning.rule_reasoning import RuleReasoner
from .reasoning.graph_reasoning import GraphReasoner
from .reasoning.llm_verification import LLMBidirectionalVerifier
from .reasoning.dynamic_fusion import DynamicWeightFusion
from .evaluation.metrics import Evaluator, generate_test_queries
from .utils import set_seed, format_time, PerfTimer


# ==================================================================
# TKG-LPEHD 完整流水线
# ==================================================================
class TKGLPEHD_Pipeline:
    """
    完整推理流水线
    Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
    """

    def __init__(self, dataset_name: str = "ICEWS14", num_quads: int = 5000):
        self.dataset_name = dataset_name
        self.num_quads = num_quads
        self.dataset = None
        self.rules = []
        self.evaluator = Evaluator()
        self.results_cache = {}
        self.N_TARGET_RELS = 3
        self.N_PATHS_PER_REL = 3

        # 根据数据集选择案例研究查询 (大小写不敏感)
        cs_key = dataset_name.upper() if dataset_name.upper() in CASE_STUDY_QUERIES else dataset_name
        self.case_study_query = CASE_STUDY_QUERIES.get(
            cs_key,
            ("Barack_Obama", "Make_a_visit", "China", "2014-12-09")
        )

    # ================================================================
    # Step 0: 数据准备
    # ================================================================
    def step0_load_data(self):
        """
        数据加载与预处理
        数据集加载与格式化
        """
        print("\n" + "=" * 60)
        print("  Step 0: 数据加载与预处理")
        print("=" * 60)

        PerfTimer.start("data_load")
        self.dataset = load_or_generate_dataset(
            self.dataset_name, self.num_quads
        )
        PerfTimer.stop("data_load")

        # 打印数据摘要
        print(self.dataset.summary())

        # 示例: 格式化为自然语言 (用于LLM理解)
        if self.dataset.quadruples:
            sample = format_quad_for_llm(
                self.dataset.quadruples[0].to_tuple()
            )
            print(f"\n  示例LLM输入: {sample}")

        return self.dataset

    # ================================================================
    # Phase 1: LLM引导三维关系路径采样
    # ================================================================
    def phase1_path_sampling(self):
        """
        步骤:
        a) 采样规则头边
        b) 评估三维候选得分 S = w1*S1 + w2*S2 + w3*S3
        c) 符号时序约束
        d) 时间加权采样 w(t) = exp(-λ|t-T|)
        e) LLM逻辑验证
        """
        print("\n" + "=" * 60)
        print("  Phase 1: LLM引导三维关系路径采样")
        print("=" * 60)

        PerfTimer.start("sbert_load")
        sampler = ThreeDimensionalPathSampler(self.dataset)
        PerfTimer.stop("sbert_load")

        # 获取目标关系列表 (使用频率最高的关系)
        rel_freq = {}
        for q in self.dataset.quadruples:
            rel_freq[q.relation] = rel_freq.get(q.relation, 0) + 1
        # 大数据集优化: 3 个目标关系 + 3 条路径/关系 (原 5+5)
        self.N_TARGET_RELS = 3
        self.N_PATHS_PER_REL = 3
        target_relations = sorted(rel_freq.keys(), key=lambda r: rel_freq[r], reverse=True)[:self.N_TARGET_RELS]

        print(f"  目标关系 (Top-{self.N_TARGET_RELS}高频): {target_relations}")
        print(f"  三维融合: S = {0.5}·S1 + {0.3}·S2 + {0.2}·S3  [公式(5)]")
        print(f"  时间衰减: w(t) = exp(-{LAMBDA_DECAY}·|t-T|)  [公式(6)]")

        all_rules = []
        all_paths_for_validity = []
        target_rel_for_validity = None

        PerfTimer.start("path_sampling")
        for target_rel in target_relations:
            last_time = max(
                (q.timestamp for q in self.dataset.quadruples
                 if q.relation == target_rel), default="2020-01-01"
            )
            paths = sampler.sample_paths(
                target_relation=target_rel,
                target_time=last_time,
                num_paths=self.N_PATHS_PER_REL,
                path_length=3
            )
            print(f"  '{target_rel}': {len(paths)}条有效路径")

            # 从路径提取候选规则
            rules = sampler.extract_rules_from_paths(paths, target_rel)
            all_rules.extend(rules)

            if paths and not target_rel_for_validity:
                all_paths_for_validity = paths
                target_rel_for_validity = target_rel

        PerfTimer.stop("path_sampling")
        print(f"\n  总计: {len(all_rules)}条候选规则, "
              f"涉及{len(target_relations)}个目标关系")

        # 存储以供后续阶段使用
        self._phase1_output = {
            "sampler": sampler,
            "all_rules": all_rules,
            "paths_for_validity": all_paths_for_validity,
            "target_rel_for_validity": target_rel_for_validity,
        }
        return all_rules

    # ================================================================
    # Phase 2: LLM辅助时序有效性评估
    # ================================================================
    def phase2_validity_assessment(self, rules, paths, target_relation):
        """
        公式: Δti = t_target - t_final
        """
        print("\n" + "=" * 60)
        print("  Phase 2: LLM辅助时序有效性评估")
        print("=" * 60)

        assessor = TemporalValidityAssessor()

        # 确定目标时间
        if paths and paths[0]:
            target_time = paths[0][0].time_val
        else:
            target_time = 200.0  # 默认年中

        # 计算时间差
        time_diffs = assessor.compute_all_path_time_diffs(target_time, paths)
        print(f"  路径数: {len(paths)}")
        print(f"  时间差范围: min={min(time_diffs):.1f}, "
              f"max={max(time_diffs):.1f}, mean={sum(time_diffs)/len(time_diffs):.1f}")

        # 构建语义样本集
        sample_set = assessor.build_semantic_sample_set(paths, target_relation)
        print(f"  语义样本集T: {sample_set['num_paths']}条描述")

        # 确定有效期V
        validity_v = assessor.determine_validity_period(
            target_time, paths, target_relation
        )
        print(f"  LLM确定有效期 V = {validity_v:.1f}天")

        # 为规则设置有效期
        rules_with_validity = assessor.assess_rules(
            rules, target_time, target_relation, paths
        )

        valid_paths = assessor.filter_valid_temporal_paths(
            paths, validity_v, target_time
        )
        print(f"  有效期过滤后: {len(valid_paths)}条有效路径")

        print(f"  各规则有效期: ", end="")
        for r in rules_with_validity[:5]:
            print(f"[{r.head_relation}: {r.validity_period:.0f}d] ", end="")
        print()

        self._phase2_output = {
            "assessor": assessor,
            "validity_v": validity_v,
            "valid_paths": valid_paths,
        }
        return rules_with_validity

    # ================================================================
    # Phase 3: 规则优先级排序
    # ================================================================
    def phase3_rule_prioritization(self, rules):
        """
        公式: cp = Hb / B
        公式: C = B_valid / B_total
        P = w1*cp + w2*C
        """
        print("\n" + "=" * 60)
        print("  Phase 3: 规则优先级排序")
        print("=" * 60)

        ranker = RulePriorityRanker(self.dataset)

        # 计算置信度和优先级
        ranked_rules = ranker.filter_and_rank(rules)
        print(f"  高置信度规则 (cp ≥ {ranker.confidence_threshold}): "
              f"{len(ranked_rules)}/{len(rules)}")

        if ranked_rules:
            # 显示top规则
            print(f"\n  Top-5 优先级规则:")
            for i, rule in enumerate(ranked_rules[:5]):
                print(f"  {i+1}. {rule.signature()}")
                print(f"     cp={rule.confidence:.3f}, C={rule.validity_coverage:.3f}, "
                      f"P={rule.priority_score:.3f}, V={rule.validity_period:.0f}d")

            # 统计
            cp_values = [r.confidence for r in ranked_rules]
            p_values = [r.priority_score for r in ranked_rules]
            print(f"\n  统计: cp∈[{min(cp_values):.3f}, {max(cp_values):.3f}], "
                  f"P∈[{min(p_values):.3f}, {max(p_values):.3f}]")

        self._phase3_output = {"ranker": ranker}
        return ranked_rules

    # ================================================================
    # Phase 4: 自适应规则演化
    # ================================================================
    def phase4_rule_evolution(self, rules):
        """
        淘汰 P < P_th 且 cp < C_th 的规则
        """
        print("\n" + "=" * 60)
        print("  Phase 4: 自适应规则演化")
        print("=" * 60)

        evolver = AdaptiveRuleEvolution(self.dataset)

        # 使用当前数据执行演化
        current_data = self.dataset.test_quads or self.dataset.quadruples[-500:]
        evolved_rules = evolver.evolve(rules, current_data)

        # 打印演化摘要
        summary = evolver.get_evolution_summary()
        print(f"\n  演化摘要: {summary}")

        self._phase4_output = {"evolver": evolver}
        return evolved_rules

    # ================================================================
    # Phase 5: LLM动态权重融合推理
    # ================================================================
    def phase5_fusion_reasoning(self, rules, query_subject, query_relation, query_time,
                                  skip_training: bool = False):
        """
        规则推理 + 图推理 + LLM验证 + 动态融合

        子阶段:
        (公式: 规则推理
        (公式): 图推理 (GNN)
        LLM双向验证
        (公式): 动态权重融合
        """
        print("\n" + "=" * 60)
        print("  Phase 5: LLM动态权重融合推理")
        print("=" * 60)

        # ---- 规则推理  ----
        print("\n  规则推理")
        rule_reasoner = RuleReasoner(self.dataset)
        rule_scores = rule_reasoner.reason(
            rules, query_subject, query_relation, query_time
        )
        print(f"  规则推理候选: {len(rule_scores)}个实体")

        # ---- 图推理 ----
        print("\n 图推理")
        graph_reasoner = GraphReasoner(self.dataset)

        if not skip_training:
            graph_reasoner.train_model()
        else:
            print("  跳过训练 (使用预训练参数或随机初始化)")

        graph_scores = graph_reasoner.reason(
            query_subject, query_relation, query_time,
            auto_train=not skip_training
        )
        print(f"  图推理候选: {len(graph_scores)}个实体")

        # ---- LLM双向验证 ----
        print("\n  LLM双向验证")
        verifier = LLMBidirectionalVerifier(self.dataset)
        tq = float(self.dataset.quadruples[0].__class__._parse_time(query_time))

        adjusted_rule, adjusted_graph = verifier.verify(
            rule_scores, graph_scores, rules,
            query_subject, query_relation, tq
        )
        print(f"  验证后规则候选: {len(adjusted_rule)}个")
        print(f"  验证后图候选: {len(adjusted_graph)}个")

        # ---- 动态权重融合 ----
        print("\n  动态权重融合")
        fusion = DynamicWeightFusion(self.dataset)

        result = fusion.full_fusion_reasoning(
            rule_scores=adjusted_rule,
            graph_scores=adjusted_graph,
            query_subject=query_subject,
            query_relation=query_relation,
            query_time=query_time,
            top_k=10,
        )

        self._phase5_output = {
            "rule_reasoner": rule_reasoner,
            "graph_reasoner": graph_reasoner,
            "verifier": verifier,
            "fusion": fusion,
        }
        return result

    # ================================================================
    # SPARQL时序事件模式提取
    # ================================================================
    def run_sparql_matching(self, use_neo4j: bool = False,
                             neo4j_store=None):
        """
        SPARQL时序事件模式匹配
        指定的6种事件匹配模式

        数据流:
        - Neo4j模式: Dataset → Neo4j → export_to_rdflib() → rdflib RDF图 → 原版SPARQL
        - 内存模式: Dataset → rdflib RDF图 → 原版SPARQL
        """
        print("\n" + "=" * 60)
        print("  SPARQL时序事件模式提取 (论文指定事件模式)")
        print("=" * 60)

        if use_neo4j and neo4j_store is not None and NEO4J_AVAILABLE:
            # ================================================
            # Neo4j + 原版SPARQL 完整数据流
            # 1) 数据已存入Neo4j  2) → RDF映射  3) → SPARQL查询
            # ================================================
            print("[SPARQL] 数据流: Dataset → Neo4j属性图 → rdflib虚拟RDF → 原版SPARQL")
            print(f"[SPARQL] {neo4j_store.summary()}")

            # 确保RDF图已构建
            if neo4j_store.rdf_graph is None:
                neo4j_store.export_to_rdflib(self.dataset)

            # 使用Neo4j导出的RDF图执行原版SPARQL
            from .rules.sparql_matcher import build_sparql_store
            mem_store = build_sparql_store(self.dataset)
            matcher = SPARQLTemporalMatcher(
                rdf_graph=neo4j_store.rdf_graph,
                rdf_store=mem_store,
                neo4j_store=neo4j_store,
                dataset=self.dataset,
            )
            print("[SPARQL] 后端: Neo4j → rdflib 原版SPARQL (W3C标准)")
        else:
            # 内存回退模式
            if use_neo4j:
                print("[SPARQL] Neo4j不可用, 回退到内存模式")
            matcher = build_native_sparql_matcher(self.dataset)
            print("[SPARQL] 后端: 内存 rdflib 原版SPARQL")

        all_matches = matcher.match_all_patterns()
        matcher.export_matches()

        return all_matches

    # ================================================================
    # 完整流水线
    # ================================================================
    def run(self, skip_training: bool = False, run_eval: bool = True,
            use_neo4j: bool = False, neo4j_uri: str = None,
            neo4j_user: str = None, neo4j_password: str = None):
        """
        运行TKG-LPEHD完整流水线
        """
        start_time = time.time()
        print_config()

        # Step 0: 数据准备
        self.step0_load_data()

        # Neo4j 存储 (如果启用)
        neo4j_store = None
        if use_neo4j:
            print("\n" + "=" * 60)
            print("  Neo4j 时序数据存储")
            print("=" * 60)
            neo4j_store = Neo4jStore(
                uri=neo4j_uri, user=neo4j_user, password=neo4j_password
            )
            neo4j_store.ingest_dataset(self.dataset)
            self.neo4j_store = neo4j_store
            print(f"[Neo4j] 数据已存入Neo4j: {neo4j_store.stats['edges_created']}条边")

        # Phase 1: 路径采样
        all_rules = self.phase1_path_sampling()
        if not all_rules:
            print("[警告] 未生成任何规则, 请检查数据集")
            return

        # Phase 2: 有效性评估
        p1 = self._phase1_output
        PerfTimer.start("llm_validity")
        rules_with_validity = self.phase2_validity_assessment(
            all_rules,
            p1["paths_for_validity"],
            p1.get("target_rel_for_validity", self.case_study_query[1])
        )
        PerfTimer.stop("llm_validity")

        # Phase 3: 优先级排序
        PerfTimer.start("rule_induction")
        ranked_rules = self.phase3_rule_prioritization(rules_with_validity)
        PerfTimer.stop("rule_induction")
        if not ranked_rules:
            print("[警告] 所有规则未通过置信度阈值")
            return

        # Phase 4: 规则演化
        PerfTimer.start("llm_evolution")
        evolved_rules = self.phase4_rule_evolution(ranked_rules)
        PerfTimer.stop("llm_evolution")
        self.rules = evolved_rules
        print(f"\n  最终规则集Sr: {len(evolved_rules)}条高价值规则")

        # SPARQL模式匹配 (Neo4j → rdflib → 原版SPARQL 或 内存直连)
        PerfTimer.start("sparql_match")
        self.run_sparql_matching(
            use_neo4j=use_neo4j, neo4j_store=neo4j_store
        )
        PerfTimer.stop("sparql_match")

        # ============================================================
        # Phase 5: 案例研究
        # ============================================================
        cs_subject, cs_relation, cs_answer, cs_time = self.case_study_query

        print("\n" + "=" * 60)
        print("  案例研究:查询推理")
        print(f"  Query: ({cs_subject}, {cs_relation}, ?, {cs_time})")
        print("=" * 60)

        case_result = self.phase5_fusion_reasoning(
            evolved_rules,
            query_subject=cs_subject,
            query_relation=cs_relation,
            query_time=cs_time,
            skip_training=skip_training,
        )

        print(f"\n  Top-5 预测:")
        for i, (entity, score) in enumerate(case_result["ranked_candidates"][:5]):
            marker = f" ← 正确" if entity == cs_answer else ""
            print(f"    {i+1}. {entity}: {score:.4f}{marker}")

        # ============================================================
        # 评估
        # ============================================================
        if run_eval:
            print("\n" + "=" * 60)
            print("  模型评估")
            print("=" * 60)

            test_queries = generate_test_queries(self.dataset, num_queries=50)

            # 预构建评估组件 (只创建一次, 避免每查询重复创建 + 加载 SBERT)
            _fusion = DynamicWeightFusion(self.dataset)
            _rule_reasoner = RuleReasoner(self.dataset)
            _graph_reasoner = GraphReasoner(self.dataset)
            _verifier = LLMBidirectionalVerifier(self.dataset)

            if not skip_training and not _graph_reasoner.is_trained:
                _graph_reasoner.train_model(epochs=20)

            def predictor(s, r, t, top_k=10):
                """优化: 复用预构建组件 + 性能计时"""
                PerfTimer.start("rule_match_query")
                rs = _rule_reasoner.reason(evolved_rules, s, r, t)
                PerfTimer.stop("rule_match_query")

                PerfTimer.start("graph_reason_query")
                gs = _graph_reasoner.reason(s, r, t, auto_train=not skip_training)
                PerfTimer.stop("graph_reason_query")

                # 跳过 LLM 的情况: 无候选 / 单候选 / 仅单侧有候选
                if not rs and not gs:
                    return list(self.dataset.entities - {s})[:top_k]
                if not rs:
                    adj_rs, adj_gs = rs, gs
                elif not gs:
                    adj_rs, adj_gs = rs, gs
                else:
                    PerfTimer.start("llm_verify_query")
                    tq_val = float(self.dataset.quadruples[0].__class__._parse_time(t))
                    adj_rs, adj_gs = _verifier.verify(rs, gs, evolved_rules, s, r, tq_val)
                    PerfTimer.stop("llm_verify_query")

                PerfTimer.start("fusion_query")
                result = _fusion.full_fusion_reasoning(
                    adj_rs, adj_gs, s, r, t, top_k=top_k
                )
                PerfTimer.stop("fusion_query")
                return [e for e, _ in result["ranked_candidates"][:top_k]]

            eval_results = self.evaluator.run_evaluation(
                test_queries, predictor, top_k=10
            )

            # 打印结果
            print(f"\n  测试查询数: {len(test_queries)}")
            for metric, value in eval_results.items():
                print(f"  {metric}: {value:.4f}")

            self.evaluator.print_results(self.dataset_name)
            self.evaluator.export_results()

        # ---- 最终摘要 ----
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"  TKG-LPEHD 完整流水线执行完毕")
        print(f"  =" * 60)
        print(f"  数据集: {self.dataset_name}")
        print(f"  四元组数: {len(self.dataset.quadruples)}")
        print(f"  实体数: {self.dataset.num_entities}")
        print(f"  关系数: {self.dataset.num_relations}")
        print(f"  最终规则数: {len(evolved_rules)}")
        print(f"  总耗时: {format_time(elapsed)}")
        print(f"  =" * 60)
        print("=" * 70)

        # 性能计时报告 (Table 6)
        # print(PerfTimer.summary(
        #     self.dataset_name, len(self.dataset.quadruples),
        #     self.N_TARGET_RELS, self.N_PATHS_PER_REL
        # ))


# ==================================================================
# 命令行入口
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description="TKG-LPEHD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
python -m tkg_lpehd.main --dataset mhaes     # 新增的医疗防疫数据集
python -m tkg_lpehd.main --dataset ICEWS14    # 原有数据集
python -m tkg_lpehd.main --dataset ICEWS18
python -m tkg_lpehd.main --dataset ICEWS0515
python -m tkg_lpehd.main --dataset YAGO
        """,
    )
    parser.add_argument("--dataset", type=str, default="ICEWS14",
                        choices=["ICEWS14", "ICEWS0515", "ICEWS18", "YAGO", "mhaes"],
                        help="数据集名称 (默认: ICEWS14)")
    parser.add_argument("--retrain", action="store_true",
                        help="强制重新训练 GNN 模型 (忽略缓存)")
    parser.add_argument("--num-quads", type=int, default=5000,
                        help="生成的四元组数量 (默认: 5000)")
    parser.add_argument("--skip-training", action="store_true",
                        help="跳过GNN训练")
    parser.add_argument("--full", action="store_true",
                        help="运行完整评估")
    parser.add_argument("--case-study-only", action="store_true",
                        help="仅运行案例研究")
    parser.add_argument("--neo4j", action="store_true",default="True",
                        help="启用Neo4j存储 + 原版SPARQL查询")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687",
                        help="Neo4j Bolt URI (默认: bolt://localhost:7687)")
    parser.add_argument("--neo4j-user", type=str, default="neo4j",
                        help="Neo4j用户名 (默认: neo4j)")
    parser.add_argument("--neo4j-password", type=str, default="neo4j@2026",
                        help="Neo4j密码")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")

    args = parser.parse_args()

    # --retrain 强制重新训练
    if args.retrain:
        import tkg_lpehd.config as _cfg
        _cfg.RETRAIN_MODEL = True

    # 固定随机种子
    set_seed(args.seed)

    print("=" * 70)
    print("  TKG-LPEHD:")
    print("=" * 70)

    if args.case_study_only:
        # 仅案例研究
        dataset = load_or_generate_dataset(args.dataset, args.num_quads)

        # ---- Neo4j存储 + SPARQL (如启用) ----
        neo4j_store = None
        if args.neo4j:
            print("\n" + "=" * 60)
            print("  Neo4j时序数据存储 → 虚拟RDF映射 → 原版SPARQL")
            print("=" * 60)
            neo4j_store = Neo4jStore(
                uri=args.neo4j_uri, user=args.neo4j_user,
                password=args.neo4j_password
            )
            neo4j_store.ingest_dataset(dataset)
            print(f"[Neo4j] 数据已存入: {neo4j_store.stats['edges_created']}条边")

            # 虚拟RDF映射
            neo4j_store.export_to_rdflib(dataset)

            # 原版SPARQL查询
            print("\n[SPARQL] 通过Neo4j → rdflib执行原版SPARQL查询...")
            from .rules.sparql_matcher import build_sparql_store
            mem_store = build_sparql_store(dataset)
            matcher = SPARQLTemporalMatcher(
                rdf_graph=neo4j_store.rdf_graph, rdf_store=mem_store,
                neo4j_store=neo4j_store, dataset=dataset
            )
            matcher.match_all_patterns()
            matcher.export_matches()
        else:
            # 内存SPARQL (无Neo4j)
            matcher = build_native_sparql_matcher(dataset)
            matcher.match_all_patterns()

        # ---- Phases 1-5 ----
        cs_key = args.dataset.upper() if args.dataset.upper() in CASE_STUDY_QUERIES else args.dataset
        cs_subject, cs_relation, cs_answer, cs_time = CASE_STUDY_QUERIES[cs_key]

        sampler = ThreeDimensionalPathSampler(dataset)
        paths = sampler.sample_paths(cs_relation, cs_time, 5, 3)
        rules = sampler.extract_rules_from_paths(paths, cs_relation)

        assessor = TemporalValidityAssessor()
        target_time = dataset.quadruples[0].__class__._parse_time(cs_time)
        rules = assessor.assess_rules(rules, target_time, cs_relation, paths)

        ranker = RulePriorityRanker(dataset)
        ranked = ranker.filter_and_rank(rules)

        evolver = AdaptiveRuleEvolution(dataset)
        evolved = evolver.evolve(ranked, dataset.quadruples[-500:])

        pipeline = TKGLPEHD_Pipeline(args.dataset, args.num_quads)
        pipeline.dataset = dataset
        pipeline.rules = evolved

        result = pipeline.phase5_fusion_reasoning(
            evolved, cs_subject, cs_relation, cs_time,
            skip_training=args.skip_training
        )

        print(f"\n  案例研究结果 (Query: {cs_subject}, {cs_relation}, ?, {cs_time}):")
        for i, (entity, score) in enumerate(result["ranked_candidates"][:5]):
            marker = f" ← 正确" if entity == cs_answer else ""
            print(f"    {i+1}. {entity}: {score:.4f}{marker}")
    else:
        # 完整流水线
        pipeline = TKGLPEHD_Pipeline(args.dataset, args.num_quads)
        pipeline.run(
            skip_training=args.skip_training, run_eval=args.full,
            use_neo4j=args.neo4j, neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user, neo4j_password=args.neo4j_password
        )


if __name__ == "__main__":
    main()

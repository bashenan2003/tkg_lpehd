"""
LLM 稳定性实验运行器 (Table 8)
===============================
两阶段设计, 大幅减少真实 API 调用:
  阶段 A: Phase 1-4 用确定性算法 (无 LLM), 每个数据集跑 1 次, 缓存规则
  阶段 B: Phase 5 用真实 LLM, 每个 Provider 切换后评估 15 个查询

API 调用: 3 数据集 × 5 Provider × 15 查询 × 2 方向 = 450 次 (原 1500+)
预估耗时: 5-15 分钟 (DeepSeek) / 15-30 分钟 (GPT-4)

使用: python -m tkg_lpehd.ablation_study
"""

import os
import sys
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from .config import DATASET_PATHS, OUTPUT_DIR, DEVICE
from .data.dataset import TKGDataSet, Quadruple
from .path_sampling.llm_guided_sampler import ThreeDimensionalPathSampler
from .path_sampling.validity_assessment import TemporalValidityAssessor
from .rules.rule_mining import RulePriorityRanker
from .rules.rule_evolution import AdaptiveRuleEvolution
from .reasoning.rule_reasoning import RuleReasoner
from .reasoning.graph_reasoning import GraphReasoner
from .reasoning.llm_verification import LLMBidirectionalVerifier
from .reasoning.dynamic_fusion import DynamicWeightFusion
from .evaluation.metrics import Evaluator, generate_test_queries
from .utils import set_seed
from .llm_client import switch_provider, get_active_provider_info

# ==================================================================
# 配置
# ==================================================================
DATASETS = ["ICEWS14", "ICEWS18", "mhaes"]

# Table 8 的 5 种 LLM 配置
STABILITY_CONFIGS = [
    ("Baseline (GPT4+Original+Stoch)", "gpt4",      "original"),
    ("Weaker LLM (Llama2 7B)",         "llama2_7b",  "original"),
    ("Weaker LLM (GPT3.5 Turbo)",      "gpt35_turbo","original"),
    ("Prompt Paraphrasing",            "prompt_paraphrasing", "original"),
    ("Deterministic Decoding",         "deterministic",       "original"),
]

NUM_QUERIES = 15   # 少量查询, 确保真实 LLM 能在合理时间内完成
TOP_K = 10
SKIP_TRAINING = True
CACHE_DIR = os.path.join(OUTPUT_DIR, "stability_cache")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================================
# 阶段 A: 确定性规则提取 (0 次 API 调用)
# ==================================================================
def phase_a_extract_rules(dataset: TKGDataSet) -> List:
    """
    Phase 1-4, 全部使用确定性算法 (0 次 API 调用)。

    通过 TKG_SIMULATE=1 强制所有模块走回退路径:
    - Phase 1 路径验证 → 启发式评分
    - Phase 2 有效期    → 固定默认值
    - Phase 4 规则演化  → 阈值淘汰 + 算术调整

    返回 evolved_rules (所有 Provider 共享)。
    """
    os.environ["TKG_SIMULATE"] = "1"

    sampler = ThreeDimensionalPathSampler(dataset)

    # 采样 10K 四元组加速 Phase 1 (大数据集下够用)
    all_quads = list(dataset.quadruples)
    if len(all_quads) > 10000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_quads), size=10000, replace=False)
        all_quads = [all_quads[i] for i in idx]

    # 临时替换, 完成后恢复
    orig_quads = dataset.quadruples
    orig_indices = (dataset.quads_by_subject, dataset.quads_by_relation)

    dataset.quadruples = all_quads
    # 重建小索引
    from collections import defaultdict
    dataset.quads_by_subject = defaultdict(list)
    dataset.quads_by_relation = defaultdict(list)
    for q in all_quads:
        dataset.quads_by_subject[q.subject].append(q)
        dataset.quads_by_relation[q.relation].append(q)

    rel_freq = {}
    for q in all_quads:
        rel_freq[q.relation] = rel_freq.get(q.relation, 0) + 1
    target_relations = sorted(rel_freq, key=rel_freq.get, reverse=True)[:5]

    all_rules, all_paths, first_rel = [], [], None
    for tr in target_relations:
        times = [q.timestamp for q in all_quads if q.relation == tr]
        last_time = max(times) if times else "2020-01-01"
        paths = sampler.sample_paths(tr, last_time, num_paths=5, path_length=3)
        rules = sampler.extract_rules_from_paths(paths, tr)
        all_rules.extend(rules)
        if paths and not first_rel:
            all_paths, first_rel = paths, tr

    # 恢复完整数据
    dataset.quadruples = orig_quads
    dataset.quads_by_subject = orig_indices[0]

    if not all_rules:
        return []

    if first_rel and all_paths:
        target_time = all_paths[0][0].time_val if all_paths[0] else 200.0
        all_rules = TemporalValidityAssessor().assess_rules(
            all_rules, target_time, first_rel, all_paths
        )

    ranked = RulePriorityRanker(dataset).filter_and_rank(all_rules)
    if not ranked:
        return []

    test_data = dataset.test_quads if dataset.test_quads else list(dataset.quadruples)[-1000:]
    evolved = AdaptiveRuleEvolution(dataset).evolve(ranked, test_data)
    return evolved


# ==================================================================
# 阶段 B: 真实 LLM 评估 (每次查询 2 次 API 调用)
# ==================================================================
def phase_b_evaluate(dataset: TKGDataSet,
                     evolved_rules: List,
                     test_queries: List[Tuple[str, str, str, str]],
                     top_k: int) -> Dict[str, float]:
    """
    Phase 5: 真实 LLM 双向验证 — 每次 verifier.verify() 走真实 API。

    TKG_SIMULATE 已清除, is_real_llm_enabled() 返回 True。
    """
    os.environ.pop("TKG_SIMULATE", None)

    evaluator = Evaluator()

    def predictor(s, r, t, top_k=top_k):
        rule_r = RuleReasoner(dataset)
        graph_r = GraphReasoner(dataset)

        rs = rule_r.reason(evolved_rules, s, r, t, priority_threshold=0.3)
        gs = graph_r.reason(s, r, t, auto_train=not SKIP_TRAINING)

        if not rs and not gs:
            return list(dataset.entities - {s})[:top_k]

        verifier = LLMBidirectionalVerifier(dataset)
        tq = float(Quadruple._parse_time(t))
        adj_rs, adj_gs = verifier.verify(rs, gs, evolved_rules, s, r, tq)

        fusion = DynamicWeightFusion(dataset)
        result = fusion.full_fusion_reasoning(adj_rs, adj_gs, s, r, t, top_k=top_k)
        return [e for e, _ in result["ranked_candidates"][:top_k]]

    return evaluator.run_evaluation(test_queries, predictor, top_k=top_k)


# ==================================================================
# 缓存规则
# ==================================================================
def cache_path(ds_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{ds_name}_rules.pkl")

def save_rules(ds_name: str, rules: List):
    with open(cache_path(ds_name), "wb") as f:
        pickle.dump(rules, f)

def load_rules(ds_name: str) -> Optional[List]:
    p = cache_path(ds_name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


# ==================================================================
# 主流程
# ==================================================================
def main():
    set_seed(42)
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    t0_total = time.time()

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        # ==== 阶段 A: 确定性规则提取 (0 API) ====
        evolved = load_rules(ds_name)
        if evolved is not None:
            print(f"  [阶段A] 从缓存加载 {len(evolved)} 条规则")
        else:
            ds = TKGDataSet(name=ds_name)
            ds.load(DATASET_PATHS[ds_name.upper()])
            print(f"  数据: {len(ds.quadruples)} quads, {ds.num_entities} entities")
            print(f"  [阶段A] 确定性规则提取 (0 API)...")
            t0 = time.time()
            evolved = phase_a_extract_rules(ds)
            save_rules(ds_name, evolved)
            print(f"  [阶段A] {len(evolved)} 条规则, {time.time()-t0:.0f}s")

        if not evolved:
            print(f"  [跳过] 无可用规则")
            ds_results = {}
            for label, _, _ in STABILITY_CONFIGS:
                ds_results[label] = {"MRR": 0.0, "Hit@1": 0.0, "Hit@10": 0.0}
            all_results[ds_name] = ds_results
            continue

        # 加载数据 + 查询
        ds = TKGDataSet(name=ds_name)
        ds.load(DATASET_PATHS[ds_name.upper()])
        test_queries = generate_test_queries(ds, NUM_QUERIES)
        print(f"  [阶段B] {len(test_queries)} 个查询, {len(STABILITY_CONFIGS)} 个 LLM 配置")
        api_est = len(test_queries) * len(STABILITY_CONFIGS) * 2
        print(f"  [阶段B] 预计 {api_est} 次真实 API 调用")

        # ==== 阶段 B: 真实 LLM 评估 (每个 Provider) ====
        ds_results = {}
        for label, provider, prompt_style in STABILITY_CONFIGS:
            print(f"\n  [{label}] ", end="", flush=True)

            # 设置 Provider 和 Prompt 风格
            os.environ["TKG_PROMPT_STYLE"] = prompt_style
            switch_provider(provider)
            info = get_active_provider_info()
            if not info.get("available", False):
                print(f"SKIP (provider unavailable)")
                ds_results[label] = {"MRR": 0.0, "Hit@1": 0.0, "Hit@10": 0.0}
                continue
            print(f"provider={info['provider']} ", end="", flush=True)

            t0 = time.time()
            try:
                m = phase_b_evaluate(ds, evolved, test_queries, TOP_K)
            except Exception as e:
                print(f"ERROR: {e}")
                m = {"MRR": 0.0, "Hit@1": 0.0, "Hit@10": 0.0}
            sec = time.time() - t0

            ds_results[label] = m
            mrr = m["MRR"] * 100
            h1 = m["Hit@1"] * 100
            h10 = m["Hit@10"] * 100
            print(f"MRR={mrr:.1f} H@1={h1:.1f} H@10={h10:.1f} ({sec:.0f}s)")

        all_results[ds_name] = ds_results

    # ============================================================
    # Table 8
    # ============================================================
    total_min = (time.time() - t0_total) / 60
    print(f"\n\n{'='*80}")
    print(f"  Table 8: LLM Stability Analysis Performance Metrics")
    print(f"  (total {total_min:.1f} min)")
    print(f"{'='*80}")
    hdr = f"{'Setting':<42}"
    for ds in DATASETS:
        hdr += f"  {ds:>27}"
    print(hdr)
    sub = f"{'':42}"
    for _ in DATASETS:
        sub += f"  {'MRR   H@1   H@10':>27}"
    print(sub)
    print("-" * 118)

    for label, _, _ in STABILITY_CONFIGS:
        row = f"{label:<42}"
        for ds in DATASETS:
            m = all_results[ds].get(label, {"MRR": 0, "Hit@1": 0, "Hit@10": 0})
            row += f"  {m['MRR']*100:5.1f}  {m['Hit@1']*100:5.1f}  {m['Hit@10']*100:5.1f}"
        print(row)

    out = os.path.join(OUTPUT_DIR, "stability_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n -> {out}")


if __name__ == "__main__":
    main()

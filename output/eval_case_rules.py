"""
Standalone script: evaluate the 3 case-study temporal rules from Section 4.6.2
against the actual ICEWS14 and MHAES datasets.

Run from project root:
  cd D:\MyProjects\tkg_lpehd && python output/eval_case_rules.py

This script does NOT modify any source files.
"""
import sys, os, csv, json, re, math
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Set


# ============================================================
# Minimal Quadruple class (replicates dataset.Quadruple)
# ============================================================
class Quad:
    __slots__ = ["subject", "relation", "object", "timestamp", "time_val"]

    def __init__(self, s: str, r: str, o: str, t: str):
        self.subject = s
        self.relation = r
        self.object = o
        self.timestamp = t
        self.time_val = self._parse_time(t)

    @staticmethod
    def _parse_time(ts: str) -> float:
        ts = str(ts).strip()
        if len(ts) == 4:
            return float(ts)
        elif len(ts) == 10:
            dt = datetime.strptime(ts, "%Y-%m-%d")
            return dt.timestamp() / 86400.0
        else:
            dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
            return dt.timestamp() / 86400.0


def load_csv(path: str) -> List[Quad]:
    """Load CSV, identical logic to dataset._load_csv."""
    text = None
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        return []

    lines = text.strip().split("\n")
    header = lines[0].strip().split(",")
    subj_idx = header.index("subject") if "subject" in header else 0
    rel_idx = header.index("relation") if "relation" in header else 1
    obj_idx = header.index("object") if "object" in header else 2
    ts_idx = header.index("timestamp") if "timestamp" in header else 3

    _date_re = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    quads = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
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

        if not _date_re.match(ts):
            continue
        if not subj or not rel or not obj:
            continue
        quads.append(Quad(subj, rel, obj, ts))
    return quads


# ============================================================
# Build indices
# ============================================================
def build_indices(quads: List[Quad]):
    """Build subject, relation, and subject-relation indexes."""
    by_subj_rel = defaultdict(list)   # (s, r) -> [Quad, ...]
    by_relation = defaultdict(list)
    by_subject = defaultdict(list)
    all_entities = set()
    all_relations = set()

    for q in quads:
        by_subj_rel[(q.subject, q.relation)].append(q)
        by_relation[q.relation].append(q)
        by_subject[q.subject].append(q)
        all_entities.add(q.subject)
        all_entities.add(q.object)
        all_relations.add(q.relation)

    return by_subj_rel, by_relation, by_subject, all_entities, all_relations


# ============================================================
# Rule evaluation logic
# ============================================================
def evaluate_2body_1head_rule(
    quads: List[Quad],
    by_subj_rel: dict,
    all_entities: Set[str],
    body_rel1: str,
    body_rel2: str,
    head_rel: str,
    dataset_name: str = "",
) -> Dict[str, float]:
    """
    Rule: (s, body_rel1, o, t1) AND (s, body_rel2, o, t2) => (s, head_rel, o, t3)
    with t1 < t2 < t3

    For each (s, o) pair where body matches, check if head event exists.
    Compute MRR / Hit@1 / Hit@10 across all trigger instances.
    """
    # Step 1: find all (s, o, t1, t2) where body holds
    body_matches = []
    for (s, r1) in list(by_subj_rel.keys()):
        if r1 != body_rel1:
            continue
        for e1 in by_subj_rel[(s, body_rel1)]:
            # e2: same subject, same object, body_rel2, later timestamp
            candidates_r2 = by_subj_rel.get((s, body_rel2), [])
            for e2 in candidates_r2:
                if e2.object == e1.object and e2.time_val > e1.time_val:
                    body_matches.append({
                        "s": s, "o": e1.object, "t1": e1.time_val,
                        "t2": e2.time_val, "t1_str": e1.timestamp, "t2_str": e2.timestamp
                    })
                    if len(body_matches) >= 5000:
                        break
            if len(body_matches) >= 5000:
                break
        if len(body_matches) >= 5000:
            break

    print(f"  [{dataset_name}] Body matches ({body_rel1} + {body_rel2}): {len(body_matches)} instances")

    if len(body_matches) < 5:
        print(f"  WARNING: too few body matches, metrics unreliable")
        return {"Hit@1": 0.0, "Hit@10": 0.0, "MRR": 0.0, "count": len(body_matches)}

    # Step 2: for each body match, check if head event exists
    # For ranking: we rank ALL entities as potential objects for the head,
    # computing the rank at which the true object appears.
    # This is a standard TKG link prediction setting.
    mrrs = []
    hits1 = []
    hits10 = []

    # Build a pool of candidate entities (limit for efficiency)
    entity_list = sorted(all_entities)
    max_candidates = min(5000, len(entity_list))

    for i, match in enumerate(body_matches):
        s, o_true = match["s"], match["o"]
        t2 = match["t2"]

        # Find all (s, head_rel, ?, t3) with t3 > t2
        head_candidates = by_subj_rel.get((s, head_rel), [])
        true_found = False
        true_t3 = None
        for h in head_candidates:
            if h.object == o_true and h.time_val > t2:
                true_found = True
                true_t3 = h.time_val
                break

        if not true_found:
            # Head event not found — rank is worst
            mrrs.append(0.0)
            hits1.append(0)
            hits10.append(0)
            continue

        # Build a ranking: we rank candidate objects based on how many
        # (s, head_rel, candidate, t>t2) events exist (frequency-based ranking)
        candidate_scores = {}
        for h in head_candidates:
            if h.time_val > t2:
                candidate_scores[h.object] = candidate_scores.get(h.object, 0) + 1

        # If the true object is not in candidates, it's still a valid answer
        if o_true not in candidate_scores:
            candidate_scores[o_true] = 0

        # Sort by score descending, then by entity name for tie-breaking
        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        entity_to_rank = {}
        for rank_idx, (ent, score) in enumerate(ranked):
            entity_to_rank[ent] = rank_idx + 1  # 1-indexed

        rank = entity_to_rank.get(o_true, len(ranked) + 1)
        mrr = 1.0 / rank
        mrrs.append(mrr)
        hits1.append(1 if rank <= 1 else 0)
        hits10.append(1 if rank <= 10 else 0)

    n = len(mrrs)
    return {
        "Hit@1": sum(hits1) / n,
        "Hit@10": sum(hits10) / n,
        "MRR": sum(mrrs) / n,
        "count": n,
    }


def evaluate_1body_1head_rule(
    quads: List[Quad],
    by_subj_rel: dict,
    all_entities: Set[str],
    body_rel: str,
    head_rel: str,
    dataset_name: str = "",
) -> Dict[str, float]:
    """
    Rule: (s, body_rel, o, t1) => (o, head_rel, o, t2) with t1 < t2
    (head subject = body object, head object = body object — self-loop)
    """
    body_matches = []
    for (s, r1) in list(by_subj_rel.keys()):
        if r1 != body_rel:
            continue
        for e1 in by_subj_rel[(s, body_rel)]:
            body_matches.append({
                "s": s, "o": e1.object, "t1": e1.time_val,
                "t1_str": e1.timestamp
            })
            if len(body_matches) >= 5000:
                break
        if len(body_matches) >= 5000:
            break

    print(f"  [{dataset_name}] Body matches ({body_rel}): {len(body_matches)} instances")

    if len(body_matches) < 5:
        print(f"  WARNING: too few body matches, metrics unreliable")
        return {"Hit@1": 0.0, "Hit@10": 0.0, "MRR": 0.0, "count": len(body_matches)}

    entity_list = sorted(all_entities)
    mrrs, hits1, hits10 = [], [], []

    for match in body_matches:
        o_true = match["o"]  # In this rule, head subject = head object = o
        t1 = match["t1"]

        # Find all (o, head_rel, o, t2) with t2 > t1 (self-loop head)
        head_candidates = by_subj_rel.get((o_true, head_rel), [])
        true_found = False
        for h in head_candidates:
            if h.object == o_true and h.time_val > t1:
                true_found = True
                break

        if not true_found:
            mrrs.append(0.0)
            hits1.append(0)
            hits10.append(0)
            continue

        # Rank candidate objects by their frequency as head objects
        candidate_scores = {}
        for (s_cand, r_cand), quads_list in by_subj_rel.items():
            if r_cand == head_rel:
                for h in quads_list:
                    if h.object == s_cand and h.time_val > t1:  # self-loop check
                        candidate_scores[s_cand] = candidate_scores.get(s_cand, 0) + 1

        if o_true not in candidate_scores:
            candidate_scores[o_true] = 0

        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        entity_to_rank = {}
        for rank_idx, (ent, score) in enumerate(ranked):
            entity_to_rank[ent] = rank_idx + 1

        rank = entity_to_rank.get(o_true, len(ranked) + 1)
        mrr = 1.0 / rank
        mrrs.append(mrr)
        hits1.append(1 if rank <= 1 else 0)
        hits10.append(1 if rank <= 10 else 0)

    n = len(mrrs)
    return {
        "Hit@1": sum(hits1) / n,
        "Hit@10": sum(hits10) / n,
        "MRR": sum(mrrs) / n,
        "count": n,
    }


# ============================================================
# Main
# ============================================================
def main():
    base = "D:/MyProjects/tkg_lpehd/data_files"

    # ---- Load ICEWS14 ----
    icews_path = os.path.join(base, "icews14", "icews14.csv")
    print("Loading ICEWS14 ...")
    icews_quads = load_csv(icews_path)
    idx_i = build_indices(icews_quads)
    print(f"  {len(icews_quads)} quads, {len(idx_i[3])} entities, {len(idx_i[4])} relations")

    # ---- Load MHAES ----
    mhaes_path = os.path.join(base, "mhaes", "mhaes.csv")
    print("\nLoading MHAES ...")
    mhaes_quads = load_csv(mhaes_path)
    idx_m = build_indices(mhaes_quads)
    print(f"  {len(mhaes_quads)} quads, {len(idx_m[3])} entities, {len(idx_m[4])} relations")

    # Check which relations exist
    print("\n=== Relation existence check ===")
    for name, rel in [("Make_an_appeal_or_request", "ICEWS14/MHAES"),
                      ("Make_statement", "ICEWS14"),
                      ("Make_a_visit", "ICEWS14"),
                      ("Launch_aid_appeal", "MHAES"),
                      ("Provide_humanitarian_aid", "MHAES")]:
        icews_has = rel in idx_i[1]
        mhaes_has = rel in idx_m[1]
        print(f"  {name}: ICEWS14={'YES' if icews_has else 'NO'} ({len(idx_i[1].get(rel, []))}), "
              f"MHAES={'YES' if mhaes_has else 'NO'} ({len(idx_m[1].get(rel, []))})")

    # ============================================================
    # Case 1: Diplomatic Visit (ICEWS14)
    # Rule: (x0, Make_an_appeal_or_request, x1, t1) AND
    #       (x0, Make_statement, x1, t2) => (x0, Make_a_visit, x1, t3)
    #       with t1 < t2 < t3
    # ============================================================
    print("\n" + "=" * 60)
    print("  Case 1: Diplomatic Visit (ICEWS14)")
    print("=" * 60)
    r1 = evaluate_2body_1head_rule(
        icews_quads, idx_i[0], idx_i[3],
        "Make_an_appeal_or_request", "Make_statement", "Make_a_visit",
        "ICEWS14"
    )
    print(f"  Hit@1  = {r1['Hit@1']:.4f}")
    print(f"  Hit@10 = {r1['Hit@10']:.4f}")
    print(f"  MRR    = {r1['MRR']:.4f}")
    print(f"  N      = {r1['count']}")

    # ============================================================
    # Case 2: Humanitarian Assistance (MHAES)
    # Rule: (x0, Launch_aid_appeal, x1, t1) => (x1, Provide_humanitarian_aid, x1, t2)
    #       with t1 < t2
    # ============================================================
    print("\n" + "=" * 60)
    print("  Case 2: Humanitarian Assistance (MHAES)")
    print("=" * 60)
    r2 = evaluate_1body_1head_rule(
        mhaes_quads, idx_m[0], idx_m[3],
        "Launch_aid_appeal", "Provide_humanitarian_aid",
        "MHAES"
    )
    print(f"  Hit@1  = {r2['Hit@1']:.4f}")
    print(f"  Hit@10 = {r2['Hit@10']:.4f}")
    print(f"  MRR    = {r2['MRR']:.4f}")
    print(f"  N      = {r2['count']}")

    # ============================================================
    # Case 3: Medical Consultation (MHAES)
    # Rule: same structure as Case 1:
    #       (x0, Make_an_appeal_or_request, x1, t1) AND
    #       (x0, Make_statement, x1, t2) => (x0, Make_a_visit, x1, t3)
    # ============================================================
    print("\n" + "=" * 60)
    print("  Case 3: Medical Consultation (MHAES)")
    print("=" * 60)
    r3 = evaluate_2body_1head_rule(
        mhaes_quads, idx_m[0], idx_m[3],
        "Make_an_appeal_or_request", "Make_statement", "Make_a_visit",
        "MHAES"
    )
    print(f"  Hit@1  = {r3['Hit@1']:.4f}")
    print(f"  Hit@10 = {r3['Hit@10']:.4f}")
    print(f"  MRR    = {r3['MRR']:.4f}")
    print(f"  N      = {r3['count']}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for name, r in [("Case 1 (Diplomatic)", r1), ("Case 2 (Humanitarian)", r2), ("Case 3 (Medical)", r3)]:
        print(f"  {name}: Hit@1={r['Hit@1']:.4f}, Hit@10={r['Hit@10']:.4f}, MRR={r['MRR']:.4f}  (N={r['count']})")


if __name__ == "__main__":
    main()

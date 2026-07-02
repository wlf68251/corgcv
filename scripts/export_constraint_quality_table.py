# scripts/15_export_constraint_quality_table.py
# -*- coding: utf-8 -*-

"""
导出每条候选互斥约束的质量分析表。

输出：
1. outputs/evaluation_tables/table_6_x_constraint_quality_detail.csv
2. outputs/evaluation_tables/table_6_x_constraint_quality_detail.md

运行：
    python scripts/15_export_constraint_quality_table.py
"""

import os
import re
import time
import pandas as pd


def detect_project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    parent = os.path.abspath(os.path.join(here, ".."))
    if os.path.exists(os.path.join(parent, "data", "processed")):
        return parent
    if os.path.exists(os.path.join(here, "data", "processed")):
        return here
    return parent


PROJECT_ROOT = detect_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "evaluation_tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

CONSTRAINTS_FINAL = os.path.join(PROCESSED_DIR, "mutual_exclusion_constraints_final.csv")
CONSTRAINTS_SCORED = os.path.join(PROCESSED_DIR, "mutual_exclusion_scored.csv")
CONSTRAINTS_LLM = os.path.join(PROCESSED_DIR, "mutual_exclusion_llm_candidates.csv")
CONFLICTS_IN = os.path.join(PROCESSED_DIR, "detected_conflicts.csv")
EDGES_AFTER_IN = os.path.join(PROCESSED_DIR, "relation_edges_after_check.csv")

OUT_CSV = os.path.join(OUTPUT_DIR, "table_6_x_constraint_quality_detail.csv")
OUT_MD = os.path.join(OUTPUT_DIR, "table_6_x_constraint_quality_detail.md")
LOG_OUT = os.path.join(LOG_DIR, "15_export_constraint_quality_table.log")


RELATION_NAME = {
    "01": "公开声明",
    "02": "呼吁请求",
    "03": "表达合作意向",
    "04": "协商磋商",
    "05": "外交合作",
    "06": "实质合作",
    "07": "提供援助",
    "08": "让步妥协",
    "09": "调查",
    "10": "要求",
    "11": "反对批评",
    "12": "拒绝",
    "13": "威胁",
    "14": "抗议",
    "15": "军事姿态",
    "16": "削减关系",
    "17": "强制行动",
    "18": "军事攻击",
    "19": "军事冲突",
    "20": "非常规暴力",
}


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def log(msg: str) -> None:
    ensure_dirs()
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(text)
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def get_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def normalize_rel(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit() and len(s) == 1:
        s = "0" + s
    return s


def to_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def pct(x):
    return f"{x * 100:.2f}%"


def read_constraints() -> tuple[pd.DataFrame, str]:
    for path in [CONSTRAINTS_FINAL, CONSTRAINTS_SCORED, CONSTRAINTS_LLM]:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            log(f"[OK] 读取约束文件: {path}, rows={len(df)}")
            return df, path
    raise FileNotFoundError("未找到 mutual_exclusion_constraints_final.csv / mutual_exclusion_scored.csv / mutual_exclusion_llm_candidates.csv")


def normalize_constraints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cid_col = get_col(df, ["constraint_id", "id"])
    key_col = get_col(df, ["constraint_key", "key", "constraint_name"])
    ra_col = get_col(df, ["relation_a", "rel_a"])
    rb_col = get_col(df, ["relation_b", "rel_b"])

    if not ra_col or not rb_col:
        raise ValueError(f"约束文件缺少 relation_a/relation_b 字段，当前字段: {list(df.columns)}")

    if cid_col:
        df["_constraint_id"] = df[cid_col].astype(str)
    else:
        df["_constraint_id"] = [f"C_{i+1:04d}" for i in range(len(df))]

    if key_col:
        df["_constraint_key"] = df[key_col].astype(str)
    else:
        df["_constraint_key"] = df["_constraint_id"]

    df["_relation_a"] = df[ra_col].apply(normalize_rel)
    df["_relation_b"] = df[rb_col].apply(normalize_rel)

    df["_relation_pair"] = df.apply(
        lambda r: f"{r['_relation_a']}_{r['_relation_b']}",
        axis=1,
    )
    df["_relation_semantic"] = df.apply(
        lambda r: f"{RELATION_NAME.get(r['_relation_a'], r['_relation_a'])} - {RELATION_NAME.get(r['_relation_b'], r['_relation_b'])}",
        axis=1,
    )

    comparable_col = get_col(df, ["comparable_count", "patecon_instantiation_count", "instantiation_count"])
    support_col = get_col(df, ["patecon_support_count", "support_count", "non_overlap_count"])
    violation_col = get_col(df, ["violation_count", "overlap_count", "negative_count"])
    confidence_col = get_col(df, ["patecon_confidence", "structural_confidence", "confidence"])
    violation_rate_col = get_col(df, ["violation_rate"])
    swc_col = get_col(df, ["source_weighted_confidence", "weighted_confidence", "weighted_conf"])
    auto_col = get_col(df, ["auto_filter_status", "filter_status"])
    final_col = get_col(df, ["final_status", "use_status"])
    use_col = get_col(df, ["use_for_update", "enabled_for_update", "is_enabled"])

    df["_comparable_count"] = to_num(df[comparable_col], 0).astype(int) if comparable_col else 0
    df["_support_count"] = to_num(df[support_col], 0).astype(int) if support_col else 0
    df["_violation_count"] = to_num(df[violation_col], 0).astype(int) if violation_col else 0

    if not comparable_col:
        df["_comparable_count"] = df["_support_count"] + df["_violation_count"]

    if confidence_col:
        df["_patecon_confidence"] = to_num(df[confidence_col], 0)
    else:
        df["_patecon_confidence"] = df.apply(
            lambda r: safe_div(r["_support_count"], r["_comparable_count"]),
            axis=1,
        )

    if violation_rate_col:
        df["_violation_rate"] = to_num(df[violation_rate_col], 0)
    else:
        df["_violation_rate"] = df.apply(
            lambda r: safe_div(r["_violation_count"], r["_comparable_count"]),
            axis=1,
        )

    df["_source_weighted_confidence"] = to_num(df[swc_col], 0) if swc_col else 0.0
    df["_auto_filter_status"] = df[auto_col].fillna("").astype(str) if auto_col else ""
    df["_final_status"] = df[final_col].fillna("").astype(str) if final_col else ""

    if use_col:
        df["_use_for_update"] = df[use_col].fillna("").astype(str)
    else:
        df["_use_for_update"] = df["_final_status"].apply(
            lambda x: "是" if x in {"enabled_for_update", "enabled_for_mark_mixed", "mark_mixed"} else "否"
        )

    return df


def count_conflicts_by_constraint(constraints: pd.DataFrame) -> dict:
    result = {cid: 0 for cid in constraints["_constraint_id"]}
    if not os.path.exists(CONFLICTS_IN):
        log("[WARN] 未找到 detected_conflicts.csv，跳过冲突数量统计。")
        return result

    conflicts = pd.read_csv(CONFLICTS_IN, low_memory=False)
    log(f"[OK] 读取冲突文件: {CONFLICTS_IN}, rows={len(conflicts)}")

    id_cols = [c for c in conflicts.columns if "constraint" in c.lower()]
    ra_col = get_col(conflicts, ["relation_a", "rel_a"])
    rb_col = get_col(conflicts, ["relation_b", "rel_b"])

    for _, c in constraints.iterrows():
        cid = str(c["_constraint_id"])
        ckey = str(c["_constraint_key"])
        ra = str(c["_relation_a"])
        rb = str(c["_relation_b"])
        pair_set = {f"{ra}_{rb}", f"{rb}_{ra}"}

        count = 0

        # 优先按 constraint_id / constraint_key 匹配
        for col in id_cols:
            s = conflicts[col].fillna("").astype(str)
            count = max(
                count,
                int(s.str.contains(re.escape(cid), regex=True).sum()),
                int(s.str.contains(re.escape(ckey), regex=True).sum()),
            )

        # 如果没有约束字段，则按 relation_a/relation_b 近似匹配
        if count == 0 and ra_col and rb_col:
            tmp = conflicts.copy()
            tmp["_ra"] = tmp[ra_col].apply(normalize_rel)
            tmp["_rb"] = tmp[rb_col].apply(normalize_rel)
            tmp["_pair"] = tmp.apply(lambda r: f"{r['_ra']}_{r['_rb']}", axis=1)
            count = int(tmp["_pair"].isin(pair_set).sum())

        result[cid] = count

    return result


def count_edges_by_constraint(constraints: pd.DataFrame) -> dict:
    result = {cid: {"triggered_edges": 0, "review_edges": 0, "mark_mixed_edges": 0} for cid in constraints["_constraint_id"]}

    if not os.path.exists(EDGES_AFTER_IN):
        log("[WARN] 未找到 relation_edges_after_check.csv，跳过关系边影响统计。")
        return result

    edges = pd.read_csv(EDGES_AFTER_IN, low_memory=False)
    log(f"[OK] 读取更新后关系边: {EDGES_AFTER_IN}, rows={len(edges)}")

    action_col = get_col(edges, ["update_action", "after_check_status", "status"])
    constraint_cols = [c for c in edges.columns if "constraint" in c.lower()]

    if not constraint_cols:
        log("[WARN] relation_edges_after_check.csv 中未找到 constraint 相关字段，无法精确统计每条约束影响边数。")
        return result

    for _, c in constraints.iterrows():
        cid = str(c["_constraint_id"])
        ckey = str(c["_constraint_key"])

        mask = pd.Series(False, index=edges.index)

        for col in constraint_cols:
            s = edges[col].fillna("").astype(str)
            mask = mask | s.str.contains(re.escape(cid), regex=True) | s.str.contains(re.escape(ckey), regex=True)

        sub = edges[mask]
        if action_col:
            actions = sub[action_col].fillna("").astype(str)
            review_edges = int((actions == "review").sum())
            mark_mixed_edges = int((actions == "mark_mixed").sum())
        else:
            review_edges = 0
            mark_mixed_edges = 0

        result[cid] = {
            "triggered_edges": int(len(sub)),
            "review_edges": review_edges,
            "mark_mixed_edges": mark_mixed_edges,
        }

    return result


def main():
    ensure_dirs()
    log("========== 导出候选约束质量明细表 ==========")

    constraints_raw, path = read_constraints()
    constraints = normalize_constraints(constraints_raw)

    conflict_counts = count_conflicts_by_constraint(constraints)
    edge_counts = count_edges_by_constraint(constraints)

    rows = []
    for _, r in constraints.iterrows():
        cid = r["_constraint_id"]
        rows.append({
            "约束编号": cid,
            "约束键": r["_constraint_key"],
            "关系对": r["_relation_pair"],
            "关系语义": r["_relation_semantic"],
            "可判断实例数": int(r["_comparable_count"]),
            "支持实例数": int(r["_support_count"]),
            "违反实例数": int(r["_violation_count"]),
            "结构置信度": round(float(r["_patecon_confidence"]), 4),
            "违反比例": round(float(r["_violation_rate"]), 4),
            "来源加权置信度": round(float(r["_source_weighted_confidence"]), 4),
            "自动筛选状态": r["_auto_filter_status"],
            "最终状态": r["_final_status"],
            "是否进入更新": r["_use_for_update"],
            "触发冲突记录数": conflict_counts.get(cid, 0),
            "影响关系边数": edge_counts.get(cid, {}).get("triggered_edges", 0),
            "review边数": edge_counts.get(cid, {}).get("review_edges", 0),
            "mark_mixed边数": edge_counts.get(cid, {}).get("mark_mixed_edges", 0),
        })

    out = pd.DataFrame(rows)

    # 为 PPT 和论文阅读排序：优先可用约束，再按可判断实例数排序
    status_order = {
        "pass": 1,
        "kept": 1,
        "strong_soft": 2,
        "mixed_candidate": 3,
        "review": 4,
        "weak_review": 5,
        "weak": 6,
    }
    out["_sort"] = out["自动筛选状态"].map(status_order).fillna(9)
    out = out.sort_values(["_sort", "可判断实例数"], ascending=[True, False]).drop(columns=["_sort"])

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(out.to_markdown(index=False))

    log(f"[OK] 已输出 CSV: {OUT_CSV}")
    log(f"[OK] 已输出 Markdown: {OUT_MD}")

    print("\n========== 候选约束质量明细表 ==========")
    print(out.to_string(index=False))
    print(f"\n[OK] CSV: {OUT_CSV}")
    print(f"[OK] Markdown: {OUT_MD}")
    print(f"[OK] Log: {LOG_OUT}")


if __name__ == "__main__":
    main()
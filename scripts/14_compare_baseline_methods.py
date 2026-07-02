# scripts/14_compare_baseline_methods.py
# -*- coding: utf-8 -*-

"""
对比方法指标统计脚本。

对比方法：
1. NoConstraint：不使用约束，全部 keep。
2. PaTeCon-Direct：同一主体、客体、月份下不能出现两种及以上关系；
   每组只保留最高置信度关系，其余标记为 filtered。
3. LLM-only：直接使用 LLM 生成的全部候选互斥约束；
   同一主体、客体、月份下若 relation_a 和 relation_b 同时出现，则保留高置信度边，过滤低置信度边。
4. SSTC-Fusion：读取已有 relation_edges_after_check.csv 的真实结果。

输出：
    outputs/evaluation_tables/table_6_x_method_comparison_metrics.csv
    outputs/evaluation_tables/table_6_x_method_action_distribution.csv
    outputs/evaluation_tables/table_6_x_method_evidence_group_actions.csv
    logs/14_compare_baseline_methods.log

运行：
    python scripts/14_compare_baseline_methods.py
"""

import os
import time
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


# =========================
# 路径配置
# =========================

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

EDGES_BEFORE_IN = os.path.join(PROCESSED_DIR, "relation_edges_before_check.csv")
EDGES_AFTER_IN = os.path.join(PROCESSED_DIR, "relation_edges_after_check.csv")

LLM_CANDIDATES_IN = os.path.join(PROCESSED_DIR, "mutual_exclusion_llm_candidates.csv")
CONSTRAINTS_FINAL_IN = os.path.join(PROCESSED_DIR, "mutual_exclusion_constraints_final.csv")

COMPARE_METRICS_OUT = os.path.join(OUTPUT_DIR, "table_6_x_method_comparison_metrics.csv")
ACTION_DIST_OUT = os.path.join(OUTPUT_DIR, "table_6_x_method_action_distribution.csv")
GROUP_ACTIONS_OUT = os.path.join(OUTPUT_DIR, "table_6_x_method_evidence_group_actions.csv")
LOG_OUT = os.path.join(LOG_DIR, "14_compare_baseline_methods.log")


# =========================
# 阈值配置
# =========================

HIGH_CONF_THRESHOLD = 0.70
LOW_CONF_THRESHOLD = 0.50

# 对比方法中，filtered 表示该方法会直接过滤/隐藏该派生关系边。
# 注意：这里只是模拟对比，不写回真实图谱。
FILTER_ACTIONS = {"filtered", "hide", "hide_low_evidence", "downgrade", "removed"}


# =========================
# 基础工具函数
# =========================

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def log(msg: str) -> None:
    ensure_dirs()
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(text)
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def get_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def to_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def to_bool(x: Any) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "是", "触发"}


def normalize_relation_type(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit() and len(s) == 1:
        s = "0" + s
    return s


def normalize_action(x: Any) -> str:
    if pd.isna(x):
        return "keep"
    s = str(x).strip()
    if not s:
        return "keep"
    return s


# =========================
# 数据读取与规范化
# =========================

def read_edges_before() -> pd.DataFrame:
    if not os.path.exists(EDGES_BEFORE_IN):
        raise FileNotFoundError(f"未找到输入文件: {EDGES_BEFORE_IN}")

    df = pd.read_csv(EDGES_BEFORE_IN, low_memory=False)
    log(f"[OK] 读取校验前关系边: {EDGES_BEFORE_IN}, rows={len(df)}")
    return normalize_edges_common(df, source_name="before")


def read_edges_after() -> pd.DataFrame:
    if not os.path.exists(EDGES_AFTER_IN):
        log(f"[WARN] 未找到 SSTC-Fusion 更新结果文件: {EDGES_AFTER_IN}")
        return pd.DataFrame()

    df = pd.read_csv(EDGES_AFTER_IN, low_memory=False)
    log(f"[OK] 读取 SSTC-Fusion 更新后关系边: {EDGES_AFTER_IN}, rows={len(df)}")
    return normalize_edges_common(df, source_name="after")


def normalize_edges_common(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    edge_col = get_col(df, ["edge_id", "id"])
    if not edge_col:
        df["edge_id"] = [f"EDGE_{i:09d}" for i in range(len(df))]
    elif edge_col != "edge_id":
        df["edge_id"] = df[edge_col].astype(str)

    subj_col = get_col(df, ["subject_org_id", "S", "subject_id"])
    obj_col = get_col(df, ["object_org_id", "O", "object_id"])
    rel_col = get_col(df, ["relation_type", "P", "relation_code"])
    month_col = get_col(df, ["event_month", "month", "t", "time"])

    missing = []
    for name, col in [
        ("subject_org_id", subj_col),
        ("object_org_id", obj_col),
        ("relation_type", rel_col),
        ("event_month/month", month_col),
    ]:
        if not col:
            missing.append(name)

    if missing:
        raise ValueError(
            f"{source_name} 关系边文件缺少必要字段: {missing}，"
            f"当前字段: {list(df.columns)}"
        )

    df["_subject"] = df[subj_col].astype(str)
    df["_object"] = df[obj_col].astype(str)
    df["_relation"] = df[rel_col].apply(normalize_relation_type)
    df["_month"] = df[month_col].astype(str)

    conf_col = get_col(df, ["final_confidence", "confidence", "score"])
    if conf_col:
        df["_confidence"] = to_num(df[conf_col], 0.0)
    else:
        df["_confidence"] = 0.0
        log(f"[WARN] {source_name} 文件未找到 confidence/final_confidence，默认置 0。")

    source_col = get_col(df, ["source_datasets", "sources", "source_dataset", "source"])
    if source_col:
        df["_sources"] = df[source_col].fillna("").astype(str)
    else:
        df["_sources"] = ""

    icews_col = get_col(df, ["icews_event_count", "icews_count"])
    gdelt_col = get_col(df, ["gdelt_event_count", "gdelt_count"])

    if icews_col:
        df["_icews_count"] = to_num(df[icews_col], 0)
    else:
        df["_icews_count"] = df["_sources"].str.upper().str.contains("ICEWS").astype(int)

    if gdelt_col:
        df["_gdelt_count"] = to_num(df[gdelt_col], 0)
    else:
        df["_gdelt_count"] = df["_sources"].str.upper().str.contains("GDELT").astype(int)

    df["_is_cross_source"] = (df["_icews_count"] > 0) & (df["_gdelt_count"] > 0)
    df["_is_high_evidence"] = df["_is_cross_source"] | (df["_confidence"] >= HIGH_CONF_THRESHOLD)
    df["_is_low_evidence"] = (~df["_is_cross_source"]) & (df["_confidence"] < LOW_CONF_THRESHOLD)

    return df


def read_llm_constraints() -> pd.DataFrame:
    if os.path.exists(LLM_CANDIDATES_IN):
        path = LLM_CANDIDATES_IN
    elif os.path.exists(CONSTRAINTS_FINAL_IN):
        path = CONSTRAINTS_FINAL_IN
    else:
        raise FileNotFoundError(
            f"未找到 LLM 候选约束文件: {LLM_CANDIDATES_IN} 或 {CONSTRAINTS_FINAL_IN}"
        )

    df = pd.read_csv(path, low_memory=False)
    log(f"[OK] 读取 LLM 候选约束: {path}, rows={len(df)}")

    rel_a_col = get_col(df, ["relation_a", "rel_a"])
    rel_b_col = get_col(df, ["relation_b", "rel_b"])

    if not rel_a_col or not rel_b_col:
        raise ValueError(
            f"LLM 候选约束文件缺少 relation_a/relation_b 字段，当前字段: {list(df.columns)}"
        )

    df["_relation_a"] = df[rel_a_col].apply(normalize_relation_type)
    df["_relation_b"] = df[rel_b_col].apply(normalize_relation_type)

    pairs = []
    for _, row in df.iterrows():
        a = row["_relation_a"]
        b = row["_relation_b"]
        if not a or not b or a == b:
            continue
        pair = tuple(sorted([a, b]))
        pairs.append(pair)

    pairs = sorted(set(pairs))
    log(f"[STAT] LLM-only 使用候选互斥关系对数量: {len(pairs)}")
    log(f"[STAT] LLM-only 关系对: {pairs}")

    return pd.DataFrame(pairs, columns=["relation_a", "relation_b"])


# =========================
# 对比方法模拟
# =========================

def init_sim_result(edges: pd.DataFrame, method_name: str) -> pd.DataFrame:
    df = edges.copy()
    df["_method"] = method_name
    df["_sim_action"] = "keep"
    df["_sim_triggered"] = False
    df["_sim_reason"] = ""
    return df


def simulate_no_constraint(edges: pd.DataFrame) -> pd.DataFrame:
    df = init_sim_result(edges, "NoConstraint")
    return df


def choose_keep_edge(group: pd.DataFrame) -> str:
    """
    在同一个冲突组中保留一个置信度最高的关系边。
    若置信度相同，则保留 event_count 较高者；
    若仍相同，则保留 edge_id 字典序最小者。
    """
    df = group.copy()

    event_col = get_col(df, ["event_count"])
    if event_col:
        df["_event_count_tmp"] = to_num(df[event_col], 0)
    else:
        df["_event_count_tmp"] = 0

    df = df.sort_values(
        by=["_confidence", "_event_count_tmp", "edge_id"],
        ascending=[False, False, True],
    )
    return str(df.iloc[0]["edge_id"])


def simulate_patecon_direct(edges: pd.DataFrame) -> pd.DataFrame:
    """
    PaTeCon-Direct 加速版：
    同一主体、客体、月份下不能出现两种及以上关系。
    每组只保留置信度最高的一条边，其余 filtered。
    """
    df = init_sim_result(edges, "PaTeCon-Direct")

    group_cols = ["_subject", "_object", "_month"]

    # 统计每组关系类型数量
    rel_count = (
        df.groupby(group_cols)["_relation"]
        .nunique()
        .reset_index(name="_rel_nunique")
    )

    df = df.merge(rel_count, on=group_cols, how="left")

    # 只处理多关系组
    conflict_mask = df["_rel_nunique"] > 1
    df.loc[conflict_mask, "_sim_triggered"] = True
    df.loc[conflict_mask, "_sim_reason"] = "same_subject_object_month_multi_relation"

    if not conflict_mask.any():
        df = df.drop(columns=["_rel_nunique"])
        return df

    # 置信度最高的边保留；其余 filtered
    event_col = get_col(df, ["event_count"])
    if event_col:
        df["_event_count_tmp"] = to_num(df[event_col], 0)
    else:
        df["_event_count_tmp"] = 0

    # 对每组排序，取第一条作为保留边
    sort_cols = group_cols + ["_confidence", "_event_count_tmp", "edge_id"]
    df_sorted = df.sort_values(
        by=sort_cols,
        ascending=[True, True, True, False, False, True],
    )

    keep_ids = (
        df_sorted[df_sorted["_rel_nunique"] > 1]
        .groupby(group_cols, sort=False)
        .head(1)["edge_id"]
        .astype(str)
        .tolist()
    )

    keep_set = set(keep_ids)

    filter_mask = conflict_mask & (~df["edge_id"].astype(str).isin(keep_set))
    df.loc[filter_mask, "_sim_action"] = "filtered"

    df = df.drop(columns=["_rel_nunique", "_event_count_tmp"], errors="ignore")
    return df

def build_llm_pair_set(constraints: pd.DataFrame) -> Set[Tuple[str, str]]:
    pairs = set()
    for _, row in constraints.iterrows():
        a = normalize_relation_type(row["relation_a"])
        b = normalize_relation_type(row["relation_b"])
        if a and b and a != b:
            pairs.add(tuple(sorted([a, b])))
    return pairs


def simulate_llm_only(edges: pd.DataFrame, constraints: pd.DataFrame) -> pd.DataFrame:
    """
    LLM-only 加速版：
    直接使用全部 LLM 候选互斥关系对。
    若同一主体、客体、月份下出现候选互斥关系对，则触发；
    每个触发关系对保留置信度最高边，其余 filtered。
    """
    df = init_sim_result(edges, "LLM-only")
    pair_set = build_llm_pair_set(constraints)

    if not pair_set:
        log("[WARN] LLM-only 没有可用互斥关系对。")
        return df

    group_cols = ["_subject", "_object", "_month"]

    # 只保留可能参与 LLM 互斥的关系类型，减少计算量
    llm_relations = set()
    for a, b in pair_set:
        llm_relations.add(a)
        llm_relations.add(b)

    cand = df[df["_relation"].isin(llm_relations)].copy()

    if cand.empty:
        return df

    # 给每个 group 内的关系做自连接，找出同组中同时出现的候选关系对
    left = cand[
        ["edge_id", "_subject", "_object", "_month", "_relation", "_confidence"]
    ].copy()
    right = left.copy()

    left = left.rename(columns={
        "edge_id": "edge_id_a",
        "_relation": "relation_a",
        "_confidence": "confidence_a",
    })

    right = right.rename(columns={
        "edge_id": "edge_id_b",
        "_relation": "relation_b",
        "_confidence": "confidence_b",
    })

    merged = left.merge(
        right,
        on=group_cols,
        how="inner",
    )

    # 去掉自己和自己匹配，并避免重复组合
    merged = merged[merged["edge_id_a"].astype(str) < merged["edge_id_b"].astype(str)].copy()

    if merged.empty:
        return df

    merged["pair"] = merged.apply(
        lambda r: tuple(sorted([normalize_relation_type(r["relation_a"]), normalize_relation_type(r["relation_b"])])),
        axis=1,
    )

    merged = merged[merged["pair"].isin(pair_set)].copy()

    if merged.empty:
        return df

    # 所有触发到的边
    triggered_ids = set(merged["edge_id_a"].astype(str)).union(
        set(merged["edge_id_b"].astype(str))
    )

    # 对每个触发 pair，过滤置信度较低的一侧
    filter_ids = set()

    for _, r in merged.iterrows():
        edge_a = str(r["edge_id_a"])
        edge_b = str(r["edge_id_b"])
        conf_a = float(r["confidence_a"])
        conf_b = float(r["confidence_b"])

        if conf_a > conf_b:
            filter_ids.add(edge_b)
        elif conf_b > conf_a:
            filter_ids.add(edge_a)
        else:
            # 置信度相同，保留 edge_id 字典序较小的边
            filter_ids.add(max(edge_a, edge_b))

    df.loc[df["edge_id"].astype(str).isin(triggered_ids), "_sim_triggered"] = True
    df.loc[df["edge_id"].astype(str).isin(triggered_ids), "_sim_reason"] = "llm_candidate_pair_triggered"
    df.loc[df["edge_id"].astype(str).isin(filter_ids), "_sim_action"] = "filtered"

    return df

def simulate_sstc_fusion(edges_before: pd.DataFrame, edges_after: pd.DataFrame) -> pd.DataFrame:
    """
    SSTC-Fusion 修正版：
    使用 relation_edges_before_check.csv 作为统一证据分组基础；
    只从 relation_edges_after_check.csv 合并 update_action 和 is_mutex_triggered。
    这样所有方法的高证据/低证据划分口径一致。
    """
    if edges_after.empty:
        return pd.DataFrame()

    df = init_sim_result(edges_before, "SSTC-Fusion")

    after = edges_after.copy()

    action_col = get_col(after, ["update_action", "after_check_status", "status"])
    triggered_col = get_col(after, ["is_mutex_triggered", "mutex_triggered", "triggered"])

    keep_cols = ["edge_id"]

    if action_col:
        after["_after_action"] = after[action_col].apply(normalize_action)
        keep_cols.append("_after_action")
    else:
        after["_after_action"] = "keep"
        keep_cols.append("_after_action")

    if triggered_col:
        after["_after_triggered"] = after[triggered_col].apply(to_bool)
        keep_cols.append("_after_triggered")
    else:
        after["_after_triggered"] = after["_after_action"].ne("keep")
        keep_cols.append("_after_triggered")

    after_small = after[keep_cols].drop_duplicates(subset=["edge_id"])

    df = df.merge(
        after_small,
        on="edge_id",
        how="left",
    )

    df["_sim_action"] = df["_after_action"].fillna("keep")
    df["_sim_triggered"] = df["_after_triggered"].fillna(False).astype(bool)
    df["_sim_reason"] = "sstc_fusion_actual_output"

    df = df.drop(columns=["_after_action", "_after_triggered"], errors="ignore")

    return df

# =========================
# 指标统计
# =========================

def compute_metrics_for_method(df: pd.DataFrame, method_name: str) -> Dict[str, Any]:
    total = len(df)

    action = df["_sim_action"].fillna("keep").astype(str)

    n_keep = int((action == "keep").sum())
    n_triggered = int(df["_sim_triggered"].sum())
    n_changed = int(action.ne("keep").sum())

    n_filtered = int(action.isin(FILTER_ACTIONS).sum())
    n_review = int((action == "review").sum())
    n_mark_mixed = int((action == "mark_mixed").sum())

    high = df[df["_is_high_evidence"]]
    n_high = len(high)
    n_high_filtered = int(high["_sim_action"].isin(FILTER_ACTIONS).sum())
    n_high_protected = n_high - n_high_filtered

    low_triggered = df[df["_is_low_evidence"] & df["_sim_triggered"]]
    n_low_triggered = len(low_triggered)

    low_processed_actions = {
        "filtered",
        "review",
        "mark_mixed",
        "downgrade",
        "hide",
        "hide_low_evidence",
        "keep_with_conflict_tag",
    }
    n_low_processed = int(low_triggered["_sim_action"].isin(low_processed_actions).sum())

    return {
        "方法": method_name,
        "关系边总数": total,
        "触发关系边数": n_triggered,
        "触发冲突覆盖率": pct(safe_div(n_triggered, total)),
        "状态变化关系边数": n_changed,
        "状态变化关系边占比": pct(safe_div(n_changed, total)),
        "keep数量": n_keep,
        "更新影响控制率": pct(safe_div(n_keep, total)),
        "filtered数量": n_filtered,
        "review数量": n_review,
        "mark_mixed数量": n_mark_mixed,
        "高证据关系数": n_high,
        "高证据被过滤数": n_high_filtered,
        "高证据关系保护率": pct(safe_div(n_high_protected, n_high)),
        "低证据触发关系数": n_low_triggered,
        "低证据触发后处理数": n_low_processed,
        "低证据异常关系处理率": pct(safe_div(n_low_processed, n_low_triggered)),
        "原始证据保留率": "100.00%",
    }


def compute_action_distribution(all_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        all_df.groupby(["_method", "_sim_action"])
        .size()
        .reset_index(name="数量")
        .rename(columns={"_method": "方法", "_sim_action": "动作"})
    )

    total_by_method = table.groupby("方法")["数量"].transform("sum")
    table["占比"] = [
        pct(safe_div(n, t)) for n, t in zip(table["数量"], total_by_method)
    ]

    return table


def compute_evidence_group_actions(all_df: pd.DataFrame) -> pd.DataFrame:
    df = all_df.copy()

    def group_name(row: pd.Series) -> str:
        if row["_is_high_evidence"]:
            return "高证据关系边"
        if row["_is_low_evidence"]:
            return "低证据关系边"
        return "中等证据关系边"

    df["_evidence_group"] = df.apply(group_name, axis=1)

    table = (
        df.groupby(["_method", "_evidence_group", "_sim_action"])
        .size()
        .reset_index(name="数量")
        .rename(
            columns={
                "_method": "方法",
                "_evidence_group": "证据强度分组",
                "_sim_action": "动作",
            }
        )
    )

    total_by_group = table.groupby(["方法", "证据强度分组"])["数量"].transform("sum")
    table["组内占比"] = [
        pct(safe_div(n, t)) for n, t in zip(table["数量"], total_by_group)
    ]

    return table


def main() -> None:
    ensure_dirs()

    log("========== 对比方法指标统计 ==========")
    log(f"[INFO] Project root: {PROJECT_ROOT}")

    edges_before = read_edges_before()
    llm_constraints = read_llm_constraints()

    results = []

    no_constraint = simulate_no_constraint(edges_before)
    results.append(no_constraint)

    patecon_direct = simulate_patecon_direct(edges_before)
    results.append(patecon_direct)

    llm_only = simulate_llm_only(edges_before, llm_constraints)
    results.append(llm_only)

    edges_after = read_edges_after()
    sstc = simulate_sstc_fusion(edges_before, edges_after)
    if not sstc.empty:
        results.append(sstc)

    all_df = pd.concat(results, ignore_index=True)

    metrics_rows = []
    for method_name, g in all_df.groupby("_method", sort=False):
        metrics_rows.append(compute_metrics_for_method(g, method_name))

    metrics_df = pd.DataFrame(metrics_rows)
    action_dist = compute_action_distribution(all_df)
    group_actions = compute_evidence_group_actions(all_df)

    metrics_df.to_csv(COMPARE_METRICS_OUT, index=False, encoding="utf-8-sig")
    action_dist.to_csv(ACTION_DIST_OUT, index=False, encoding="utf-8-sig")
    group_actions.to_csv(GROUP_ACTIONS_OUT, index=False, encoding="utf-8-sig")

    log(f"[OK] 已输出方法对比指标表: {COMPARE_METRICS_OUT}")
    log(f"[OK] 已输出方法动作分布表: {ACTION_DIST_OUT}")
    log(f"[OK] 已输出证据分组动作表: {GROUP_ACTIONS_OUT}")

    print("\n========== 方法对比指标 ==========")
    print(metrics_df.to_string(index=False))

    print("\n========== 方法动作分布 ==========")
    print(action_dist.to_string(index=False))

    print("\n========== 证据强度分组动作分布 ==========")
    print(group_actions.to_string(index=False))

    print(f"\n[OK] 方法对比指标: {COMPARE_METRICS_OUT}")
    print(f"[OK] 动作分布: {ACTION_DIST_OUT}")
    print(f"[OK] 证据分组动作: {GROUP_ACTIONS_OUT}")
    print(f"[OK] 日志: {LOG_OUT}")


if __name__ == "__main__":
    main()
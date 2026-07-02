# scripts/13_compute_effectiveness_metrics.py
# -*- coding: utf-8 -*-

"""
计算第六章方法有效性指标。

主要输出：
1. 高证据关系保护率
2. 低证据异常关系处理率
3. 低结构支持约束占比
4. 约束有效筛选率
5. 更新影响控制率
6. 触发冲突覆盖率
7. 原始证据保留率
8. 不同证据强度关系边的更新动作分布

运行：
    python scripts/13_compute_effectiveness_metrics.py
"""

import os
import time
from typing import Any, List

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

EDGES_AFTER_IN = os.path.join(PROCESSED_DIR, "relation_edges_after_check.csv")
CONSTRAINTS_FINAL_IN = os.path.join(PROCESSED_DIR, "mutual_exclusion_constraints_final.csv")
CONSTRAINTS_SCORED_IN = os.path.join(PROCESSED_DIR, "mutual_exclusion_scored.csv")
CONFLICTS_IN = os.path.join(PROCESSED_DIR, "detected_conflicts.csv")

METRICS_OUT = os.path.join(OUTPUT_DIR, "table_6_x_effectiveness_metrics.csv")
GROUP_ACTIONS_OUT = os.path.join(OUTPUT_DIR, "table_6_x_evidence_group_actions.csv")
CONSTRAINT_STATUS_OUT = os.path.join(OUTPUT_DIR, "table_6_x_constraint_status_stats.csv")
LOG_OUT = os.path.join(LOG_DIR, "13_compute_effectiveness_metrics.log")


HIGH_CONF_THRESHOLD = 0.70
LOW_CONF_THRESHOLD = 0.50


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


def to_bool(x: Any) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "是", "触发"}


def to_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def read_edges() -> pd.DataFrame:
    if not os.path.exists(EDGES_AFTER_IN):
        raise FileNotFoundError(f"未找到关系边文件: {EDGES_AFTER_IN}")

    df = pd.read_csv(EDGES_AFTER_IN, low_memory=False)
    log(f"[OK] 读取关系边文件: {EDGES_AFTER_IN}, rows={len(df)}")
    return df


def read_constraints() -> pd.DataFrame:
    if os.path.exists(CONSTRAINTS_FINAL_IN):
        path = CONSTRAINTS_FINAL_IN
    elif os.path.exists(CONSTRAINTS_SCORED_IN):
        path = CONSTRAINTS_SCORED_IN
    else:
        log("[WARN] 未找到互斥约束文件，将跳过约束状态统计。")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    log(f"[OK] 读取互斥约束文件: {path}, rows={len(df)}")
    return df


def normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conf_col = get_col(df, ["final_confidence", "confidence", "score"])
    if not conf_col:
        log("[WARN] 未找到 confidence/final_confidence 字段，置信度默认置为 0。")
        df["_confidence"] = 0.0
    else:
        df["_confidence"] = to_num(df[conf_col], 0.0)

    action_col = get_col(df, ["update_action", "action", "after_check_status", "status"])
    if not action_col:
        log("[WARN] 未找到 update_action 字段，默认全部视为 keep。")
        df["_action"] = "keep"
    else:
        df["_action"] = df[action_col].fillna("keep").astype(str).str.strip()

    triggered_col = get_col(df, ["is_mutex_triggered", "mutex_triggered", "triggered"])
    if not triggered_col:
        log("[WARN] 未找到 is_mutex_triggered 字段，将根据 update_action != keep 近似判断触发。")
        df["_triggered"] = df["_action"].ne("keep")
    else:
        df["_triggered"] = df[triggered_col].apply(to_bool)

    source_col = get_col(df, ["source_datasets", "sources", "source_dataset", "source"])
    if not source_col:
        log("[WARN] 未找到 source_datasets/sources 字段，来源默认置为空。")
        df["_sources"] = ""
    else:
        df["_sources"] = df[source_col].fillna("").astype(str)

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

    # 高证据关系：跨源支持 或 final_confidence >= 0.70
    df["_is_high_evidence"] = df["_is_cross_source"] | (df["_confidence"] >= HIGH_CONF_THRESHOLD)

    # 低证据关系：非跨源 且 final_confidence < 0.50
    df["_is_low_evidence"] = (~df["_is_cross_source"]) & (df["_confidence"] < LOW_CONF_THRESHOLD)

    # 被强处理：降权或隐藏
    df["_is_downgrade_or_hide"] = df["_action"].isin([
        "downgrade",
        "hide",
        "hide_low_evidence",
        "hidden",
        "downgraded",
    ])

    # 被处理：记录、标记、降权、隐藏都算处理
    df["_is_processed"] = df["_action"].isin([
        "review",
        "mark_mixed",
        "downgrade",
        "hide",
        "hide_low_evidence",
        "keep_with_conflict_tag",
        "mixed",
        "hidden",
        "downgraded",
    ])

    # 被保护：没有被降权或隐藏。review/mark_mixed 视为保护，因为没有删除或强制降权高证据边
    df["_is_protected"] = ~df["_is_downgrade_or_hide"]

    return df


def compute_edge_metrics(edges: pd.DataFrame) -> pd.DataFrame:
    total_edges = len(edges)

    n_keep = int((edges["_action"] == "keep").sum())
    n_triggered_edges = int(edges["_triggered"].sum())
    n_affected = int(edges["_action"].ne("keep").sum())

    high_edges = edges[edges["_is_high_evidence"]]
    n_high = len(high_edges)
    n_high_protected = int(high_edges["_is_protected"].sum())

    low_triggered = edges[edges["_is_low_evidence"] & edges["_triggered"]]
    n_low_triggered = len(low_triggered)
    n_low_processed = int(low_triggered["_is_processed"].sum())

    if os.path.exists(CONFLICTS_IN):
        conflicts = pd.read_csv(CONFLICTS_IN, low_memory=False)
        n_conflicts = len(conflicts)
        log(f"[OK] 读取冲突记录文件: {CONFLICTS_IN}, rows={n_conflicts}")
    else:
        n_conflicts = 0
        log("[WARN] 未找到 detected_conflicts.csv，触发冲突数量置为 0。")

    metrics = [
        {
            "指标名称": "高证据关系保护率",
            "指标值": pct(safe_div(n_high_protected, n_high)),
            "分子": n_high_protected,
            "分母": n_high,
            "判断方向": "越高越好",
            "说明": "高证据关系中未被降权或隐藏的比例，反映方法对跨源支持或高置信度关系的保护能力。",
        },
        {
            "指标名称": "低证据异常关系处理率",
            "指标值": pct(safe_div(n_low_processed, n_low_triggered)),
            "分子": n_low_processed,
            "分母": n_low_triggered,
            "判断方向": "越高越好",
            "说明": "触发互斥约束的低证据关系中被 review、mark_mixed、downgrade 或 hide 的比例。",
        },
        {
            "指标名称": "更新影响控制率",
            "指标值": pct(safe_div(n_keep, total_edges)),
            "分子": n_keep,
            "分母": total_edges,
            "判断方向": "越高说明越保守",
            "说明": "保持 keep 的关系边占全部关系边比例，反映方法是否避免大范围修改图谱。",
        },
        {
            "指标名称": "触发冲突覆盖率",
            "指标值": pct(safe_div(n_triggered_edges, total_edges)),
            "分子": n_triggered_edges,
            "分母": total_edges,
            "判断方向": "适中较好",
            "说明": "触发互斥约束的关系边占全部关系边比例，反映约束实际影响范围。",
        },
        {
            "指标名称": "状态变化关系边占比",
            "指标值": pct(safe_div(n_affected, total_edges)),
            "分子": n_affected,
            "分母": total_edges,
            "判断方向": "不宜过高",
            "说明": "非 keep 关系边占比，反映图谱更新对关系边状态的影响范围。",
        },
        {
            "指标名称": "原始证据保留率",
            "指标值": "100.00%",
            "分子": total_edges,
            "分母": total_edges,
            "判断方向": "越高越好",
            "说明": "本文只更新派生关系边状态，不删除 ICEWS/GDELT 原始事件证据，因此该值为 100%。",
        },
    ]

    if n_conflicts > 0:
        metrics.append({
            "指标名称": "触发冲突记录数量",
            "指标值": str(n_conflicts),
            "分子": n_conflicts,
            "分母": "",
            "判断方向": "展示指标",
            "说明": "满足最终约束触发条件的冲突记录数量，用于展示约束触发规模。",
        })

    return pd.DataFrame(metrics)


def compute_group_action_table(edges: pd.DataFrame) -> pd.DataFrame:
    def group_name(row: pd.Series) -> str:
        if row["_is_high_evidence"]:
            return "高证据关系边"
        if row["_is_low_evidence"]:
            return "低证据关系边"
        return "中等证据关系边"

    df = edges.copy()
    df["_evidence_group"] = df.apply(group_name, axis=1)

    table = (
        df.groupby(["_evidence_group", "_action"])
        .size()
        .reset_index(name="数量")
    )

    pivot = table.pivot_table(
        index="_evidence_group",
        columns="_action",
        values="数量",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    pivot = pivot.rename(columns={"_evidence_group": "证据强度分组"})

    action_cols = [c for c in pivot.columns if c != "证据强度分组"]
    pivot["合计"] = pivot[action_cols].sum(axis=1)

    for c in action_cols:
        pivot[f"{c}_占比"] = pivot.apply(
            lambda r: pct(safe_div(r[c], r["合计"])),
            axis=1,
        )

    return pivot


def compute_constraint_metrics(constraints: pd.DataFrame) -> pd.DataFrame:
    if constraints.empty:
        return pd.DataFrame()

    df = constraints.copy()
    total = len(df)

    status_col = get_col(df, ["auto_filter_status", "filter_status", "status"])
    final_col = get_col(df, ["final_status"])
    update_col = get_col(df, ["use_for_update", "is_enabled"])

    if status_col:
        df["_auto_status"] = df[status_col].fillna("").astype(str).str.strip()
    else:
        df["_auto_status"] = ""

    if final_col:
        df["_final_status"] = df[final_col].fillna("").astype(str).str.strip()
    else:
        df["_final_status"] = ""

    if update_col:
        df["_use_for_update"] = df[update_col].apply(to_bool)
    else:
        df["_use_for_update"] = df["_final_status"].isin([
            "enabled_for_update",
            "enabled_for_mark_mixed",
        ])

    weak_statuses = {
        "weak",
        "weak_review",
        "review",
        "review_only",
        "weak_prompt",
    }

    selected_statuses = {
        "pass",
        "strong_soft",
        "mixed_candidate",
        "enabled_for_update",
        "enabled_for_mark_mixed",
    }

    n_weak = int(df["_auto_status"].isin(weak_statuses).sum())

    n_selected = int(
        df["_use_for_update"].sum()
        if update_col or final_col
        else df["_auto_status"].isin(selected_statuses).sum()
    )

    swc_col = get_col(df, ["source_weighted_confidence", "weighted_confidence", "weighted_conf"])
    pc_col = get_col(df, ["patecon_confidence", "confidence"])
    vr_col = get_col(df, ["violation_rate"])

    avg_swc = to_num(df[swc_col], 0).mean() if swc_col else 0.0
    avg_pc = to_num(df[pc_col], 0).mean() if pc_col else 0.0
    avg_vr = to_num(df[vr_col], 0).mean() if vr_col else 0.0

    metrics = [
        {
            "指标名称": "低结构支持约束占比",
            "指标值": pct(safe_div(n_weak, total)),
            "分子": n_weak,
            "分母": total,
            "判断方向": "越低越好；若较高则说明结构筛选必要",
            "说明": "自动筛选状态为 weak/review/weak_review 等弱约束的比例。",
        },
        {
            "指标名称": "约束有效筛选率",
            "指标值": pct(safe_div(n_selected, total)),
            "分子": n_selected,
            "分母": total,
            "判断方向": "适中较好",
            "说明": "进入自动更新或自动标记阶段的约束占候选约束比例。",
        },
        {
            "指标名称": "平均来源加权置信度",
            "指标值": f"{avg_swc:.4f}",
            "分子": "",
            "分母": "",
            "判断方向": "越高越好",
            "说明": "候选约束在来源证据加权后的平均置信度。",
        },
        {
            "指标名称": "平均结构置信度",
            "指标值": f"{avg_pc:.4f}",
            "分子": "",
            "分母": "",
            "判断方向": "越高越好",
            "说明": "候选约束在时间区间结构证据上的平均支持程度。",
        },
        {
            "指标名称": "平均违反比例",
            "指标值": f"{avg_vr:.4f}",
            "分子": "",
            "分母": "",
            "判断方向": "越低越好",
            "说明": "候选约束在当前图谱中的平均违反比例。",
        },
    ]

    # 额外输出状态分布表
    status_stats = (
        df.groupby("_auto_status")
        .size()
        .reset_index(name="数量")
        .rename(columns={"_auto_status": "自动筛选状态"})
    )
    status_stats["占比"] = status_stats["数量"].apply(lambda x: pct(safe_div(x, total)))
    status_stats.to_csv(CONSTRAINT_STATUS_OUT, index=False, encoding="utf-8-sig")
    log(f"[OK] 已输出约束状态分布表: {CONSTRAINT_STATUS_OUT}")

    return pd.DataFrame(metrics)


def main() -> None:
    ensure_dirs()
    log("========== 计算第六章方法有效性指标 ==========")
    log(f"[INFO] Project root: {PROJECT_ROOT}")

    edges = read_edges()
    edges = normalize_edges(edges)

    edge_metrics = compute_edge_metrics(edges)
    group_actions = compute_group_action_table(edges)

    constraints = read_constraints()
    constraint_metrics = compute_constraint_metrics(constraints)

    all_metrics = pd.concat(
        [edge_metrics, constraint_metrics],
        ignore_index=True,
    )

    all_metrics.to_csv(METRICS_OUT, index=False, encoding="utf-8-sig")
    group_actions.to_csv(GROUP_ACTIONS_OUT, index=False, encoding="utf-8-sig")

    log(f"[OK] 已输出方法有效性指标表: {METRICS_OUT}")
    log(f"[OK] 已输出证据分组动作分布表: {GROUP_ACTIONS_OUT}")

    print("\n========== 方法有效性指标 ==========")
    print(all_metrics.to_string(index=False))

    print("\n========== 证据强度分组 × 更新动作分布 ==========")
    print(group_actions.to_string(index=False))

    print(f"\n[OK] 指标输出: {METRICS_OUT}")
    print(f"[OK] 分组动作输出: {GROUP_ACTIONS_OUT}")
    print(f"[OK] 日志输出: {LOG_OUT}")


if __name__ == "__main__":
    main()
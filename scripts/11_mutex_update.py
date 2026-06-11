# scripts/11_update_graph.py
# -*- coding: utf-8 -*-

"""
第 11 步：根据最终互斥约束更新派生关系边，并生成可视化图数据。

新版逻辑：
1. 读取第 10 步输出 mutual_exclusion_scored.csv。
2. 可选读取人工复核表 mutual_exclusion_manual_review.csv。
3. 融合 LLM、PaTeCon-style confidence 和人工复核结果，生成 mutual_exclusion_constraints_final.csv。
4. 根据最终约束，在 same_subject_object + month 范围内检测互斥触发。
5. 不删除原始 ICEWS/GDELT 事件证据，只更新派生关系边：
   - enabled_for_update: 可 downgrade / mark_mixed / hide_low_evidence
   - enabled_for_mark_mixed: 只标记 mixed，不隐藏
   - review_only: 只输出复核记录，不改边
   - disabled: 不使用
6. 输出：
   - relation_edges_after_check.csv
   - mutual_exclusion_conflicts.csv
   - update_decisions.csv
   - mutual_exclusion_constraints_final.csv
   - timeline_graph_data.json
   - outputs/case_studies/*.csv

运行：
    python scripts/11_update_graph.py

如果脚本放在项目根目录：
    python ./11_update_graph.py
"""

import os
import json
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


def detect_project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(os.path.join(here, "data", "processed")):
        return here
    parent = os.path.abspath(os.path.join(here, ".."))
    if os.path.exists(os.path.join(parent, "data", "processed")):
        return parent
    return parent


PROJECT_ROOT = detect_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EVAL_DIR = os.path.join(OUTPUTS_DIR, "evaluation_tables")
CASE_DIR = os.path.join(OUTPUTS_DIR, "case_studies")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

EDGES_IN = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")
SCORED_CONSTRAINTS_IN = os.path.join(PROCESSED_DIR, "mutual_exclusion_scored.csv")
VIOLATION_INSTANCES_IN = os.path.join(PROCESSED_DIR, "mutual_exclusion_violation_instances.csv")

DEFAULT_MANUAL_REVIEW_IN = os.path.join(EVAL_DIR, "mutual_exclusion_manual_review.csv")

FINAL_CONSTRAINTS_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_constraints_final.csv")
EDGES_AFTER_OUT = os.path.join(PROCESSED_DIR, "relation_edges_after_check.csv")
CONFLICTS_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_conflicts.csv")
DECISIONS_OUT = os.path.join(PROCESSED_DIR, "update_decisions.csv")
TIMELINE_JSON_OUT = os.path.join(PROCESSED_DIR, "timeline_graph_data.json")

LOG_OUT = os.path.join(LOG_DIR, "11_update_graph.log")


RELATION_ZH = {
    "01": "公开声明",
    "02": "呼吁请求",
    "03": "表达合作意愿",
    "04": "协商磋商",
    "05": "外交合作",
    "06": "实质合作",
    "07": "提供援助",
    "08": "让步妥协",
    "09": "调查询问",
    "10": "要求施压",
    "11": "反对批评",
    "12": "拒绝合作",
    "13": "威胁警告",
    "14": "抗议冲突",
    "15": "非常规暴力",
    "16": "制裁断交",
    "17": "强制行动",
    "18": "军事攻击",
    "19": "军事冲突",
    "20": "战争暴力",
    "verbal_cooperation": "言语合作",
    "material_cooperation": "实质合作",
    "verbal_conflict": "言语冲突",
    "material_conflict": "实质冲突",
    "mixed_relation": "混合关系",
}


ACTION_SEVERITY = {
    "keep": 0,
    "review": 1,
    "mark_mixed": 2,
    "downgrade": 3,
    "hide_low_evidence": 4,
    "hide": 5,
}


def log(msg: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(text)
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(CASE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def normalize_relation_type(x: Any) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    if x.isdigit() and len(x) == 1:
        x = "0" + x
    return x


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def get_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def read_csv_required(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到{name}: {path}")
    return pd.read_csv(path, low_memory=False)


def detect_edge_columns(edges: pd.DataFrame) -> Dict[str, str]:
    cols = {
        "edge_id": get_col(edges, ["edge_id", "id"]),
        "subject": get_col(edges, ["subject_org_id", "subject_id", "source_org_id"]),
        "object": get_col(edges, ["object_org_id", "object_id", "target_org_id"]),
        "subject_name": get_col(edges, ["subject_name", "source_name", "Actor1Name"]),
        "object_name": get_col(edges, ["object_name", "target_name", "Actor2Name"]),
        "relation": get_col(edges, ["relation_type", "relation_code", "event_root_code", "cameo_root", "cameo_code"]),
        "month": get_col(edges, ["month", "event_month", "event_date", "date"]),
        "confidence": get_col(edges, ["confidence", "final_confidence", "score"]),
        "sources": get_col(edges, ["sources", "source_datasets", "dataset"]),
        "icews_count": get_col(edges, ["icews_event_count", "icews_count"]),
        "gdelt_count": get_col(edges, ["gdelt_event_count", "gdelt_count"]),
        "num_mentions": get_col(edges, ["num_mentions", "NumMentions"]),
        "num_articles": get_col(edges, ["num_articles", "NumArticles"]),
        "is_derived": get_col(edges, ["is_derived_relation", "derived", "is_derived"]),
    }

    required = ["subject", "object", "relation", "month"]
    missing = [k for k in required if not cols[k]]
    if missing:
        raise ValueError(f"relation_edges_scored.csv 缺少必要字段: {missing}，当前字段: {list(edges.columns)}")

    return cols


def normalize_month(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace("/", "-").replace(".", "-")
    digits = "".join(ch for ch in s if ch.isdigit())

    if len(s) >= 7 and s[4:5] == "-":
        return s[:7]

    if len(digits) >= 6:
        return f"{digits[:4]}-{digits[4:6]}"

    return s[:7]


def edge_source_profile(row: pd.Series, cols: Dict[str, str]) -> Dict[str, Any]:
    sources_text = str(row.get(cols["sources"], "") if cols["sources"] else "").upper()

    icews_count = safe_float(row.get(cols["icews_count"], 0), 0.0) if cols["icews_count"] else 0.0
    gdelt_count = safe_float(row.get(cols["gdelt_count"], 0), 0.0) if cols["gdelt_count"] else 0.0

    has_icews = icews_count > 0 or "ICEWS" in sources_text
    has_gdelt = gdelt_count > 0 or "GDELT" in sources_text

    confidence = safe_float(row.get(cols["confidence"], 0.0), 0.0) if cols["confidence"] else 0.0
    num_mentions = safe_float(row.get(cols["num_mentions"], 0), 0.0) if cols["num_mentions"] else 0.0
    num_articles = safe_float(row.get(cols["num_articles"], 0), 0.0) if cols["num_articles"] else 0.0

    is_low_evidence = False
    if has_gdelt and not has_icews:
        if confidence < 0.5 or (num_mentions <= 1 and num_articles <= 1):
            is_low_evidence = True

    return {
        "has_icews": has_icews,
        "has_gdelt": has_gdelt,
        "cross_source": has_icews and has_gdelt,
        "low_evidence": is_low_evidence,
        "confidence": confidence,
        "sources_text": sources_text,
    }


def is_derived_edge(row: pd.Series, cols: Dict[str, str]) -> bool:
    col = cols.get("is_derived", "")
    if not col:
        return True

    val = row.get(col, True)

    if isinstance(val, bool):
        return val

    s = str(val).strip().lower()
    if s in {"false", "0", "no", "n", "raw", "original"}:
        return False
    return True


def load_manual_review(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        log(f"[INFO] 未找到人工复核表，使用自动规则生成最终约束: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    log(f"[OK] 读取人工复核表: {path}, rows={len(df)}")
    return df


def pick_manual_decision(row: pd.Series) -> str:
    for c in ["final_status", "manual_decision", "review_decision", "decision", "use_status"]:
        if c in row.index:
            v = str(row.get(c, "")).strip()
            if v:
                return v
    return ""


def auto_final_status(row: pd.Series) -> str:
    """
    根据第 10 步输出的 auto_filter_status、candidate_level、expected_action 等生成最终默认状态。

    注意：
    - pass 是严格通过，可用于更新；
    - strong_soft / mixed_candidate 更适合 mark_mixed；
    - review 默认只进入复核；
    - weak / disabled 不采用。
    """
    status = str(row.get("auto_filter_status", "")).strip()
    level = str(row.get("candidate_level", "")).strip()
    expected_action = str(row.get("llm_expected_action", row.get("expected_action", ""))).strip()

    confidence = safe_float(row.get("patecon_confidence", 0.0), 0.0)
    weighted_conf = safe_float(row.get("source_weighted_confidence", 0.0), 0.0)
    violation_rate = safe_float(row.get("violation_rate", 1.0), 1.0)
    comparable = safe_int(row.get("comparable_count", row.get("patecon_instantiation_count", 0)), 0)

    if status == "pass":
        return "enabled_for_update"

    if status == "strong_soft":
        return "enabled_for_mark_mixed"

    if status == "mixed_candidate":
        return "enabled_for_mark_mixed"

    # 兼容旧版 10.py：没有 strong_soft 时，可把接近阈值的强候选作为强软约束
    if (
        status == "review"
        and level == "strong_candidate"
        and comparable >= 500
        and confidence >= 0.74
        and weighted_conf >= 0.735
        and violation_rate <= 0.26
    ):
        return "enabled_for_mark_mixed"

    if status in {"review", "weak_review"}:
        if expected_action == "mark_mixed" and comparable >= 500 and confidence >= 0.72 and violation_rate <= 0.29:
            return "enabled_for_mark_mixed"
        return "review_only"

    return "disabled"


def normalize_manual_status(v: str) -> str:
    s = str(v or "").strip()

    mapping = {
        "pass": "enabled_for_update",
        "enable": "enabled_for_update",
        "enabled": "enabled_for_update",
        "enabled_for_update": "enabled_for_update",

        "strong_soft": "enabled_for_mark_mixed",
        "mixed_candidate": "enabled_for_mark_mixed",
        "mark_mixed": "enabled_for_mark_mixed",
        "enabled_for_mark_mixed": "enabled_for_mark_mixed",

        "review": "review_only",
        "review_only": "review_only",
        "manual_review": "review_only",

        "weak": "disabled",
        "disable": "disabled",
        "disabled": "disabled",
        "drop": "disabled",
        "delete": "disabled",
    }

    return mapping.get(s, s)


def build_final_constraints(scored: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    final_df = scored.copy()

    if "constraint_key" not in final_df.columns:
        final_df["constraint_key"] = (
            final_df["relation_a"].astype(str).map(normalize_relation_type)
            + "_"
            + final_df["relation_b"].astype(str).map(normalize_relation_type)
            + "_"
            + final_df["forbidden_temporal_relation"].astype(str)
        )

    final_df["auto_final_status"] = final_df.apply(auto_final_status, axis=1)
    final_df["manual_final_status"] = ""
    final_df["manual_score"] = ""
    final_df["manual_comment"] = ""

    if not manual.empty:
        manual = manual.copy()

        if "constraint_key" not in manual.columns and {"relation_a", "relation_b", "forbidden_temporal_relation"}.issubset(manual.columns):
            manual["constraint_key"] = (
                manual["relation_a"].astype(str).map(normalize_relation_type)
                + "_"
                + manual["relation_b"].astype(str).map(normalize_relation_type)
                + "_"
                + manual["forbidden_temporal_relation"].astype(str)
            )

        manual_lookup = {}

        for _, r in manual.iterrows():
            key = ""
            if "constraint_id" in manual.columns and str(r.get("constraint_id", "")).strip():
                key = ("constraint_id", str(r.get("constraint_id")).strip())
            elif "constraint_key" in manual.columns and str(r.get("constraint_key", "")).strip():
                key = ("constraint_key", str(r.get("constraint_key")).strip())

            if key:
                manual_lookup[key] = r

        for idx, r in final_df.iterrows():
            mr = None

            cid = str(r.get("constraint_id", "")).strip()
            ckey = str(r.get("constraint_key", "")).strip()

            if ("constraint_id", cid) in manual_lookup:
                mr = manual_lookup[("constraint_id", cid)]
            elif ("constraint_key", ckey) in manual_lookup:
                mr = manual_lookup[("constraint_key", ckey)]

            if mr is not None:
                decision = normalize_manual_status(pick_manual_decision(mr))
                if decision:
                    final_df.at[idx, "manual_final_status"] = decision

                for c in ["manual_score", "score", "review_score"]:
                    if c in mr.index and str(mr.get(c, "")).strip():
                        final_df.at[idx, "manual_score"] = mr.get(c, "")
                        break

                for c in ["manual_comment", "comment", "review_comment", "reason"]:
                    if c in mr.index and str(mr.get(c, "")).strip():
                        final_df.at[idx, "manual_comment"] = mr.get(c, "")
                        break

    final_df["final_status"] = final_df.apply(
        lambda r: r["manual_final_status"] if str(r.get("manual_final_status", "")).strip() else r["auto_final_status"],
        axis=1,
    )

    final_df["is_enabled"] = final_df["final_status"].isin({
        "enabled_for_update",
        "enabled_for_mark_mixed",
        "review_only",
    })

    final_df["use_for_update"] = final_df["final_status"].isin({
        "enabled_for_update",
        "enabled_for_mark_mixed",
    })

    final_df["use_for_review"] = final_df["final_status"].isin({
        "enabled_for_update",
        "enabled_for_mark_mixed",
        "review_only",
    })

    return final_df


def prepare_edges(edges: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    df = edges.copy()

    if not cols["edge_id"]:
        df["edge_id"] = [f"E_{i:09d}" for i in range(1, len(df) + 1)]
        cols["edge_id"] = "edge_id"

    df["_subject"] = df[cols["subject"]].astype(str)
    df["_object"] = df[cols["object"]].astype(str)
    df["_relation"] = df[cols["relation"]].apply(normalize_relation_type)
    df["_month"] = df[cols["month"]].apply(normalize_month)

    if cols["confidence"]:
        df["_original_confidence"] = df[cols["confidence"]].apply(lambda x: safe_float(x, 0.0))
    else:
        df["_original_confidence"] = 0.0

    df["original_confidence"] = df["_original_confidence"]
    df["final_confidence"] = df["_original_confidence"]

    df["is_mutex_triggered"] = False
    df["after_check_status"] = "keep"
    df["update_action"] = "keep"
    df["update_reason"] = ""
    df["triggered_constraint_ids"] = ""
    df["triggered_constraint_keys"] = ""
    df["triggered_mutex_relations"] = ""

    df["_current_severity"] = 0

    return df


def append_text(old: Any, new: str) -> str:
    old_s = str(old or "").strip()
    new_s = str(new or "").strip()
    if not new_s:
        return old_s
    if not old_s:
        return new_s

    parts = [p for p in old_s.split(";") if p]
    if new_s not in parts:
        parts.append(new_s)
    return ";".join(parts)


def choose_edge_action(
    final_status: str,
    expected_action: str,
    edge_row: pd.Series,
    cols: Dict[str, str],
) -> str:
    """
    根据最终约束状态、LLM 建议动作和边证据质量决定实际更新动作。
    """
    if final_status == "review_only":
        return "review"

    if final_status == "enabled_for_mark_mixed":
        return "mark_mixed"

    if final_status != "enabled_for_update":
        return "review"

    profile = edge_source_profile(edge_row, cols)
    derived = is_derived_edge(edge_row, cols)

    if not derived:
        return "review"

    # 保护 ICEWS + GDELT 共同支持的高证据边
    if profile["cross_source"]:
        return "mark_mixed"

    expected_action = str(expected_action or "").strip()

    if expected_action == "hide":
        if profile["low_evidence"]:
            return "hide_low_evidence"
        return "mark_mixed"

    if expected_action == "downgrade":
        return "downgrade"

    if expected_action == "mark_mixed":
        return "mark_mixed"

    if expected_action == "mark_mixed_or_downgrade":
        if profile["low_evidence"]:
            return "downgrade"
        return "mark_mixed"

    return "mark_mixed"


def apply_action_to_edge(
    edges: pd.DataFrame,
    idx: int,
    action: str,
    constraint_id: str,
    constraint_key: str,
    mutex_relation: str,
    reason: str,
    downgrade_factor: float,
) -> None:
    old_action = str(edges.at[idx, "update_action"])
    old_sev = ACTION_SEVERITY.get(old_action, 0)
    new_sev = ACTION_SEVERITY.get(action, 0)

    edges.at[idx, "is_mutex_triggered"] = True
    edges.at[idx, "triggered_constraint_ids"] = append_text(edges.at[idx, "triggered_constraint_ids"], constraint_id)
    edges.at[idx, "triggered_constraint_keys"] = append_text(edges.at[idx, "triggered_constraint_keys"], constraint_key)
    edges.at[idx, "triggered_mutex_relations"] = append_text(edges.at[idx, "triggered_mutex_relations"], mutex_relation)
    edges.at[idx, "update_reason"] = append_text(edges.at[idx, "update_reason"], reason)

    if new_sev < old_sev:
        return

    edges.at[idx, "_current_severity"] = new_sev
    edges.at[idx, "update_action"] = action

    if action == "review":
        edges.at[idx, "after_check_status"] = "review"

    elif action == "mark_mixed":
        edges.at[idx, "after_check_status"] = "mark_mixed"

    elif action == "downgrade":
        edges.at[idx, "after_check_status"] = "downgraded"
        old_conf = safe_float(edges.at[idx, "final_confidence"], 0.0)
        edges.at[idx, "final_confidence"] = round(old_conf * downgrade_factor, 6)

    elif action == "hide_low_evidence":
        edges.at[idx, "after_check_status"] = "hidden"
        old_conf = safe_float(edges.at[idx, "final_confidence"], 0.0)
        edges.at[idx, "final_confidence"] = round(old_conf * min(downgrade_factor, 0.5), 6)

    elif action == "hide":
        edges.at[idx, "after_check_status"] = "hidden"
        old_conf = safe_float(edges.at[idx, "final_confidence"], 0.0)
        edges.at[idx, "final_confidence"] = round(old_conf * min(downgrade_factor, 0.5), 6)


def update_graph_edges(
    edges: pd.DataFrame,
    cols: Dict[str, str],
    final_constraints: pd.DataFrame,
    downgrade_factor: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges = prepare_edges(edges, cols)

    usable = final_constraints[final_constraints["use_for_review"] == True].copy()

    if usable.empty:
        log("[WARN] 没有可用于 review/update 的最终互斥约束。")
        return edges, pd.DataFrame(), pd.DataFrame()

    relation_pairs = []

    for _, c in usable.iterrows():
        relation_a = normalize_relation_type(c.get("relation_a", ""))
        relation_b = normalize_relation_type(c.get("relation_b", ""))
        if not relation_a or not relation_b or relation_a == relation_b:
            continue

        relation_pairs.append({
            "constraint_id": str(c.get("constraint_id", "")),
            "constraint_key": str(c.get("constraint_key", f"{relation_a}_{relation_b}_overlap")),
            "constraint_name": str(c.get("constraint_name", "")),
            "relation_a": relation_a,
            "relation_b": relation_b,
            "final_status": str(c.get("final_status", "review_only")),
            "expected_action": str(c.get("llm_expected_action", c.get("expected_action", "review"))),
            "patecon_confidence": safe_float(c.get("patecon_confidence", 0.0), 0.0),
            "source_weighted_confidence": safe_float(c.get("source_weighted_confidence", 0.0), 0.0),
            "violation_rate": safe_float(c.get("violation_rate", 0.0), 0.0),
            "auto_filter_status": str(c.get("auto_filter_status", "")),
            "candidate_level": str(c.get("candidate_level", "")),
            "reason": str(c.get("llm_reason", c.get("reason", ""))),
        })

    if not relation_pairs:
        log("[WARN] 可用约束中没有合法 relation_a/relation_b。")
        return edges, pd.DataFrame(), pd.DataFrame()

    conflicts = []
    decisions = []
    conflict_id = 1

    grouped = edges.groupby(["_subject", "_object", "_month"], sort=False)

    for (s, o, month), g in grouped:
        if not month:
            continue

        rel_set = set(g["_relation"].astype(str).tolist())

        for c in relation_pairs:
            ra = c["relation_a"]
            rb = c["relation_b"]

            if ra not in rel_set or rb not in rel_set:
                continue

            ga = g[g["_relation"] == ra]
            gb = g[g["_relation"] == rb]

            if ga.empty or gb.empty:
                continue

            constraint_id = c["constraint_id"]
            constraint_key = c["constraint_key"]
            final_status = c["final_status"]
            expected_action = c["expected_action"]
            mutex_relation = f"{ra}-{rb}"

            edge_ids_a = ga[cols["edge_id"]].astype(str).tolist()
            edge_ids_b = gb[cols["edge_id"]].astype(str).tolist()

            actions_for_group = []

            for idx in list(ga.index) + list(gb.index):
                action = choose_edge_action(
                    final_status=final_status,
                    expected_action=expected_action,
                    edge_row=edges.loc[idx],
                    cols=cols,
                )

                if final_status == "review_only":
                    action = "review"

                actions_for_group.append(action)

                apply_action_to_edge(
                    edges=edges,
                    idx=idx,
                    action=action,
                    constraint_id=constraint_id,
                    constraint_key=constraint_key,
                    mutex_relation=mutex_relation,
                    reason=c["reason"],
                    downgrade_factor=downgrade_factor,
                )

                decisions.append({
                    "decision_id": f"D_{len(decisions) + 1:09d}",
                    "constraint_id": constraint_id,
                    "constraint_key": constraint_key,
                    "subject_org_id": s,
                    "object_org_id": o,
                    "month": month,
                    "edge_id": str(edges.at[idx, cols["edge_id"]]),
                    "relation_type": str(edges.at[idx, "_relation"]),
                    "final_status": final_status,
                    "update_action": action,
                    "original_confidence": edges.at[idx, "original_confidence"],
                    "final_confidence": edges.at[idx, "final_confidence"],
                    "reason": c["reason"],
                    "patecon_confidence": c["patecon_confidence"],
                    "source_weighted_confidence": c["source_weighted_confidence"],
                    "violation_rate": c["violation_rate"],
                    "auto_filter_status": c["auto_filter_status"],
                    "candidate_level": c["candidate_level"],
                })

            conflict_action = max(actions_for_group, key=lambda x: ACTION_SEVERITY.get(x, 0)) if actions_for_group else "review"

            conflicts.append({
                "conflict_id": f"C_{conflict_id:09d}",
                "constraint_id": constraint_id,
                "constraint_key": constraint_key,
                "constraint_name": c["constraint_name"],
                "subject_org_id": s,
                "object_org_id": o,
                "month": month,
                "relation_a": ra,
                "relation_b": rb,
                "edge_ids_a": ";".join(edge_ids_a),
                "edge_ids_b": ";".join(edge_ids_b),
                "final_status": final_status,
                "conflict_action": conflict_action,
                "expected_action": expected_action,
                "patecon_confidence": c["patecon_confidence"],
                "source_weighted_confidence": c["source_weighted_confidence"],
                "violation_rate": c["violation_rate"],
                "auto_filter_status": c["auto_filter_status"],
                "candidate_level": c["candidate_level"],
                "reason": c["reason"],
            })

            conflict_id += 1

    edges = edges.drop(columns=["_current_severity"], errors="ignore")

    conflicts_df = pd.DataFrame(conflicts)
    decisions_df = pd.DataFrame(decisions)

    return edges, conflicts_df, decisions_df


def build_timeline_graph_data(
    edges_after: pd.DataFrame,
    cols: Dict[str, str],
    out_path: str,
    max_edges_per_month: int = 500,
) -> Dict[str, Any]:
    df = edges_after.copy()

    months = sorted([m for m in df["_month"].dropna().astype(str).unique().tolist() if m])
    graphs = {}

    for month in months:
        g = df[df["_month"] == month].copy()

        if g.empty:
            graphs[month] = {"nodes": [], "links": []}
            continue

        # 不要先按筛选删边，只限制数量
        if "final_confidence" in g.columns:
            g = g.sort_values(
                ["is_mutex_triggered", "final_confidence"],
                ascending=[False, False],
            )

        if max_edges_per_month and len(g) > max_edges_per_month:
            g = g.head(max_edges_per_month).copy()

        node_degree = {}
        node_name_map = {}

        for _, r in g.iterrows():
            s_id = str(r["_subject"])
            o_id = str(r["_object"])

            s_name = str(r.get("_subject_name", s_id))
            o_name = str(r.get("_object_name", o_id))

            node_degree[s_id] = node_degree.get(s_id, 0) + 1
            node_degree[o_id] = node_degree.get(o_id, 0) + 1

            node_name_map[s_id] = s_name if s_name and s_name.lower() != "nan" else s_id
            node_name_map[o_id] = o_name if o_name and o_name.lower() != "nan" else o_id

        nodes = []
        for node_id, degree in sorted(node_degree.items(), key=lambda x: (-x[1], x[0])):
            display_name = node_name_map.get(node_id, node_id)

            nodes.append({
                "id": node_id,          # 连接用，必须是 ORG_ID
                "name": display_name,   # 显示用，真实名称
                "org_id": node_id,
                "category": "组织机构",
                "symbolSize": max(14, min(46, 10 + degree * 2)),
                "value": degree,
            })

        links = []

        for _, r in g.iterrows():
            source_id = str(r["_subject"])
            target_id = str(r["_object"])

            source_name = str(r.get("_subject_name", source_id))
            target_name = str(r.get("_object_name", target_id))

            relation = str(r["_relation"])
            relation_zh = relation_label(relation)

            action = str(r.get("update_action", "keep"))
            status = str(r.get("after_check_status", "keep"))

            links.append({
                "source": source_id,     # 连接用，必须对应 node.id
                "target": target_id,

                "source_name": source_name,
                "target_name": target_name,
                "source_org_id": source_id,
                "target_org_id": target_id,

                "edge_id": str(r.get(cols["edge_id"], "")),
                "relation": relation,
                "relation_code": relation,
                "relation_label": relation_zh,
                "edge_label": relation_zh,

                "confidence": safe_float(r.get("final_confidence", 0.0), 0.0),
                "original_confidence": safe_float(r.get("original_confidence", 0.0), 0.0),

                "status": status,
                "status_label": str(r.get("after_check_status_label", status)),
                "update_action": action,
                "update_action_label": str(r.get("update_action_label", action)),

                "sources": str(r.get(cols["sources"], "")) if cols.get("sources") else "",
                "triggered_constraint_ids": str(r.get("triggered_constraint_ids", "")),
                "triggered_constraint_keys": str(r.get("triggered_constraint_keys", "")),
                "triggered_mutex_relations": str(r.get("triggered_mutex_relations", "")),
                "triggered_mutex_relations_label": str(r.get("triggered_mutex_relations_label", "")),
                "update_reason": str(r.get("update_reason", "")),
                "is_mutex_triggered": bool(r.get("is_mutex_triggered", False)),
            })

        graphs[month] = {
            "nodes": nodes,
            "links": links,
        }

    data = {
        "months": months,
        "graphs": graphs,
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "max_edges_per_month": max_edges_per_month,
            "node_display": "node.id 保留 ORG_ID，node.name 显示真实名称",
            "edge_display": "edge_label 显示中文关系含义",
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

def save_case_studies(conflicts: pd.DataFrame, decisions: pd.DataFrame) -> None:
    os.makedirs(CASE_DIR, exist_ok=True)

    if not conflicts.empty:
        conflicts[conflicts["conflict_action"] == "mark_mixed"].head(100).to_csv(
            os.path.join(CASE_DIR, "case_mutex_mark_mixed.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        conflicts[conflicts["conflict_action"].isin(["hide", "hide_low_evidence"])].head(100).to_csv(
            os.path.join(CASE_DIR, "case_mutex_hide_low_evidence.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        conflicts[conflicts["conflict_action"] == "review"].head(100).to_csv(
            os.path.join(CASE_DIR, "case_mutex_review.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if not decisions.empty:
        decisions[decisions["update_action"] == "downgrade"].head(100).to_csv(
            os.path.join(CASE_DIR, "case_mutex_downgrade.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        decisions[decisions["update_action"] == "mark_mixed"].head(100).to_csv(
            os.path.join(CASE_DIR, "case_mutex_keep_cross_source.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_review_path", type=str, default=DEFAULT_MANUAL_REVIEW_IN)
    parser.add_argument("--downgrade_factor", type=float, default=0.8)
    parser.add_argument("--max_edges_per_month", type=int, default=400)
    args = parser.parse_args()

    ensure_dirs()

    log("========== 第 11 步：根据最终互斥约束更新派生关系边 ==========")

    edges = read_csv_required(EDGES_IN, "关系边文件 relation_edges_scored.csv")
    cols = detect_edge_columns(edges)
    log(f"[OK] 读取关系边: {EDGES_IN}, rows={len(edges)}")

    scored = read_csv_required(SCORED_CONSTRAINTS_IN, "第 10 步约束评分文件 mutual_exclusion_scored.csv")
    log(f"[OK] 读取互斥约束评分: {SCORED_CONSTRAINTS_IN}, rows={len(scored)}")

    manual = load_manual_review(args.manual_review_path)

    final_constraints = build_final_constraints(scored, manual)
    final_constraints.to_csv(FINAL_CONSTRAINTS_OUT, index=False, encoding="utf-8-sig")
    log(f"[OK] 已生成最终互斥约束: {FINAL_CONSTRAINTS_OUT}, rows={len(final_constraints)}")

    status_counts = final_constraints["final_status"].value_counts(dropna=False).to_dict()
    log(f"[STAT] final_status 分布: {status_counts}")

    edges_after, conflicts, decisions = update_graph_edges(
        edges=edges,
        cols=cols,
        final_constraints=final_constraints,
        downgrade_factor=args.downgrade_factor,
    )

    edges_after.to_csv(EDGES_AFTER_OUT, index=False, encoding="utf-8-sig")
    conflicts.to_csv(CONFLICTS_OUT, index=False, encoding="utf-8-sig")
    decisions.to_csv(DECISIONS_OUT, index=False, encoding="utf-8-sig")

    log(f"[OK] 已生成更新后关系边: {EDGES_AFTER_OUT}, rows={len(edges_after)}")
    log(f"[OK] 已生成互斥冲突表: {CONFLICTS_OUT}, rows={len(conflicts)}")
    log(f"[OK] 已生成更新决策表: {DECISIONS_OUT}, rows={len(decisions)}")

    action_counts = edges_after["update_action"].value_counts(dropna=False).to_dict()
    log(f"[STAT] update_action 分布: {action_counts}")

    build_timeline_graph_data(
        edges_after=edges_after,
        cols=cols,
        out_path=TIMELINE_JSON_OUT,
        max_edges_per_month=args.max_edges_per_month,
    )
    log(f"[OK] 已生成时间轴图谱数据: {TIMELINE_JSON_OUT}")

    save_case_studies(conflicts, decisions)
    log(f"[OK] 已生成案例文件目录: {CASE_DIR}")

    log("========== 第 11 步完成 ==========")


if __name__ == "__main__":
    main()
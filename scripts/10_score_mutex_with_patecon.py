# 10_score_mutex_with_patecon.py
# -*- coding: utf-8 -*-

"""
第 10 步：为 LLM 候选互斥约束计算 PaTeCon-style 数据置信度。

注意：
本脚本是“新方法主流程”的第 10 步，不是 PaTeCon-only baseline。
它不会调用 PaTeCon-master/Constraint_Mining.py，也不会让 PaTeCon 自由挖掘全部频繁时序规则。

本脚本只读取第 09 步生成的 mutual_exclusion_llm_candidates.csv，
并在 org_relation_intervals.tsv 上逐条计算候选互斥约束的数据置信度。

输入：
    data/processed/org_relation_intervals.tsv
    data/processed/mutual_exclusion_llm_candidates.csv
    data/processed/relation_edges_scored.csv

输出：
    data/processed/mutual_exclusion_scored.csv
    data/processed/mutual_exclusion_trigger_instances.csv
    data/processed/mutual_exclusion_violation_instances.csv
    logs/10_score_mutex_with_patecon.log

运行：
    python ./10_score_mutex_with_patecon.py

如果放在 scripts/ 目录下，也可以运行：
    python scripts/10_score_mutex_with_patecon.py
"""

import os
import re
import time
import argparse
from typing import Any, Dict, List, Tuple

import pandas as pd


# ============================================================
# 路径设置
# ============================================================

def detect_project_root() -> str:
    """
    兼容两种放置方式：
    1. 项目根目录/10_score_mutex_with_patecon.py
    2. 项目根目录/scripts/10_score_mutex_with_patecon.py
    """
    here = os.path.abspath(os.path.dirname(__file__))

    if os.path.exists(os.path.join(here, "data", "processed")):
        return here

    parent = os.path.abspath(os.path.join(here, ".."))
    if os.path.exists(os.path.join(parent, "data", "processed")):
        return parent

    return here


PROJECT_ROOT = detect_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

INTERVALS_IN = os.path.join(PROCESSED_DIR, "org_relation_intervals.tsv")
CANDIDATES_IN = os.path.join(PROCESSED_DIR, "mutual_exclusion_llm_candidates.csv")
EDGES_IN = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")

SCORED_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_scored.csv")
TRIGGER_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_trigger_instances.csv")
VIOLATION_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_violation_instances.csv")

LOG_OUT = os.path.join(LOG_DIR, "10_score_mutex_with_patecon.log")


# ============================================================
# 基础工具函数
# ============================================================

def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def log(msg: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(text)
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def normalize_relation_type(x: Any) -> str:
    """
    将关系类型统一为字符串。
    例如：
        1   -> 01
        1.0 -> 01
        06  -> 06
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if not s:
        return ""

    if s.endswith(".0"):
        s = s[:-2]

    if s.isdigit() and len(s) == 1:
        s = "0" + s

    return s


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


# ============================================================
# 时间解析
# ============================================================

def parse_month(x: Any) -> pd.Period:
    """
    将各种可能的时间格式统一解析为 pandas Period(月粒度)。

    支持：
        2023-03
        2023-03-01
        202303
        20230301
        2023030
        2023/03/01
        2023.03.01
    """
    if pd.isna(x):
        return pd.NaT

    s = str(x).strip()

    if not s or s.lower() in {"nan", "nat", "none", "null"}:
        return pd.NaT

    if s.endswith(".0"):
        s = s[:-2]

    s = s.replace("/", "-").replace(".", "-")

    # YYYY-MM 或 YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{1,2})(?:-\d{1,2})?$", s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
        return pd.NaT

    digits = re.sub(r"\D", "", s)

    # YYYYMMDD
    if len(digits) >= 8:
        year = int(digits[:4])
        month = int(digits[4:6])
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
        return pd.NaT

    # YYYYMM
    if len(digits) == 6:
        year = int(digits[:4])
        month = int(digits[4:6])
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
        return pd.NaT

    # 异常 7 位，例如 2023030，按年月修复
    if len(digits) == 7:
        year = int(digits[:4])
        month = int(digits[4:6])
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
        return pd.NaT

    return pd.NaT


def read_intervals(path: str) -> pd.DataFrame:
    """
    读取 org_relation_intervals.tsv。

    期望格式：
        S, P, O, start_time, end_time

    输出字段：
        subject_org_id
        relation_type
        object_org_id
        start_time
        end_time
        start_period
        end_period
        interval_id
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到区间文件: {path}")

    df = pd.read_csv(path, sep="\t", header=None, dtype=str)

    if df.shape[1] < 5:
        raise ValueError("org_relation_intervals.tsv 至少需要 5 列: S, P, O, start_time, end_time")

    df = df.iloc[:, :5].copy()
    df.columns = [
        "subject_org_id",
        "relation_type",
        "object_org_id",
        "start_time",
        "end_time",
    ]

    df["subject_org_id"] = df["subject_org_id"].astype(str)
    df["object_org_id"] = df["object_org_id"].astype(str)
    df["relation_type"] = df["relation_type"].apply(normalize_relation_type)

    df["start_period"] = df["start_time"].apply(parse_month)
    df["end_period"] = df["end_time"].apply(parse_month)

    bad_time = df["start_period"].isna() | df["end_period"].isna()
    if bad_time.any():
        bad_path = os.path.join(PROCESSED_DIR, "bad_interval_time_rows.csv")
        df.loc[bad_time].to_csv(bad_path, index=False, encoding="utf-8-sig")
        log(f"[WARN] 发现无法解析的时间区间 {bad_time.sum()} 条，已保存到: {bad_path}")
        df = df.loc[~bad_time].copy()

    bad_order = df["end_period"] < df["start_period"]
    if bad_order.any():
        log(f"[WARN] 发现 end_time < start_time 的区间 {bad_order.sum()} 条，已交换起止时间。")

        old_start_period = df.loc[bad_order, "start_period"].copy()
        old_start_time = df.loc[bad_order, "start_time"].copy()

        df.loc[bad_order, "start_period"] = df.loc[bad_order, "end_period"]
        df.loc[bad_order, "start_time"] = df.loc[bad_order, "end_time"]

        df.loc[bad_order, "end_period"] = old_start_period
        df.loc[bad_order, "end_time"] = old_start_time

    df["start_time"] = df["start_period"].astype(str)
    df["end_time"] = df["end_period"].astype(str)

    df["interval_id"] = ["I_%09d" % i for i in range(1, len(df) + 1)]

    log(f"[OK] 有效关系区间 rows={len(df)}")

    return df


# ============================================================
# 来源加权
# ============================================================

def edge_source_weight(row: pd.Series) -> float:
    """
    根据边来源质量计算权重。

    ICEWS + GDELT 共同支持：1.20
    ICEWS 单源：1.00
    GDELT 单源高证据：0.85
    GDELT 单源低证据：0.60
    无法判断：0.75
    """
    sources_text = str(row.get("sources", row.get("source_datasets", ""))).upper()

    icews_count = row.get("icews_event_count", row.get("icews_count", 0))
    gdelt_count = row.get("gdelt_event_count", row.get("gdelt_count", 0))

    try:
        icews_count = float(icews_count)
    except Exception:
        icews_count = 1.0 if "ICEWS" in sources_text else 0.0

    try:
        gdelt_count = float(gdelt_count)
    except Exception:
        gdelt_count = 1.0 if "GDELT" in sources_text else 0.0

    confidence = safe_float(row.get("confidence", 0.0), 0.0)
    num_mentions = safe_float(row.get("num_mentions", row.get("NumMentions", 0)), 0.0)
    num_articles = safe_float(row.get("num_articles", row.get("NumArticles", 0)), 0.0)

    has_icews = icews_count > 0 or "ICEWS" in sources_text
    has_gdelt = gdelt_count > 0 or "GDELT" in sources_text

    if has_icews and has_gdelt:
        return 1.20

    if has_icews and not has_gdelt:
        return 1.00

    if has_gdelt and not has_icews:
        if confidence < 0.5 or (num_mentions <= 1 and num_articles <= 1):
            return 0.60
        return 0.85

    return 0.75


def build_interval_weight_map(edges: pd.DataFrame) -> Dict[Tuple[str, str, str], float]:
    """
    根据 relation_edges_scored.csv 为每个 subject-relation-object 生成平均来源权重。

    key:
        (subject_org_id, relation_type, object_org_id)
    """
    if edges.empty:
        return {}

    subj_col = get_col(edges, ["subject_org_id", "subject_id", "source_org_id"])
    obj_col = get_col(edges, ["object_org_id", "object_id", "target_org_id"])
    rel_col = get_col(edges, ["relation_type", "relation_code", "event_root_code", "cameo_root", "cameo_code"])

    if not subj_col or not obj_col or not rel_col:
        log("[WARN] relation_edges_scored.csv 缺少 subject/object/relation 字段，无法计算来源权重，使用默认 1.0。")
        return {}

    tmp = edges.copy()
    tmp[subj_col] = tmp[subj_col].astype(str)
    tmp[obj_col] = tmp[obj_col].astype(str)
    tmp["_relation_type_norm"] = tmp[rel_col].apply(normalize_relation_type)
    tmp["_source_weight"] = tmp.apply(edge_source_weight, axis=1)

    weight_map = (
        tmp.groupby([subj_col, "_relation_type_norm", obj_col])["_source_weight"]
        .mean()
        .to_dict()
    )

    log(f"[OK] 已构建来源权重映射 keys={len(weight_map)}")

    return weight_map


def interval_pair_weight(
    row_a: pd.Series,
    row_b: pd.Series,
    weight_map: Dict[Tuple[str, str, str], float],
) -> float:
    key_a = (
        str(row_a["subject_org_id"]),
        str(row_a["relation_type"]),
        str(row_a["object_org_id"]),
    )
    key_b = (
        str(row_b["subject_org_id"]),
        str(row_b["relation_type"]),
        str(row_b["object_org_id"]),
    )

    wa = float(weight_map.get(key_a, 1.0))
    wb = float(weight_map.get(key_b, 1.0))

    return (wa + wb) / 2.0


# ============================================================
# PaTeCon-style 时间区间关系判断
# ============================================================

def relation_between_intervals(
    a_start: pd.Period,
    a_end: pd.Period,
    b_start: pd.Period,
    b_end: pd.Period,
) -> str:
    """
    判断两个时间区间之间的关系。

    返回：
        before
        after
        meet
        equal
        contains
        during
        overlap
    """
    if a_end < b_start:
        if a_end + 1 == b_start:
            return "meet"
        return "before"

    if b_end < a_start:
        if b_end + 1 == a_start:
            return "meet"
        return "after"

    if a_start == b_start and a_end == b_end:
        return "equal"

    if a_start <= b_start and a_end >= b_end:
        return "contains"

    if a_start >= b_start and a_end <= b_end:
        return "during"

    return "overlap"


def patecon_truth_for_mutex(
    interval_relation: str,
    forbidden_temporal_relation: str,
) -> int:
    """
    PaTeCon-style 三值判断。

    返回：
        1  = positive / support，表示该实例支持候选互斥约束
        -1 = negative / violation，表示该实例违反候选互斥约束
        0  = unknown，表示无法判断，不进入 confidence 分母

    对本文互斥约束：
        forbidden_temporal_relation = overlap 或 same_month 时：
            before / after / meet 视为支持互斥；
            overlap / during / contains / equal 视为违反互斥。
    """
    forbidden = str(forbidden_temporal_relation or "overlap").strip()

    non_overlap_relations = {"before", "after", "meet", "disjoint"}
    overlap_relations = {"overlap", "during", "contains", "equal"}

    if forbidden in {"overlap", "same_month"}:
        if interval_relation in non_overlap_relations:
            return 1
        if interval_relation in overlap_relations:
            return -1
        return 0

    if interval_relation == forbidden:
        return -1

    if interval_relation in non_overlap_relations:
        return 1

    return 0


# ============================================================
# 约束状态判断
# ============================================================

def decide_auto_filter_status(
    candidate_level: str,
    comparable_count: int,
    patecon_confidence: float,
    source_weighted_confidence: float,
    violation_rate: float,
    llm_expected_action: str,
) -> str:
    """
    将 PaTeCon-style 评分结果转成自动筛选状态。

    pass:
        严格强约束，通常很少出现。
    strong_soft:
        实验用强软约束，可用于 mark_mixed，但不建议 hide。
    mixed_candidate:
        适合进行 mixed 标记的候选。
    review:
        只进入人工复核或提示。
    weak_review:
        弱复核候选。
    weak:
        数据支持不足。
    """
    candidate_level = str(candidate_level or "")
    llm_expected_action = str(llm_expected_action or "")

    if (
        comparable_count >= 30
        and patecon_confidence >= 0.85
        and source_weighted_confidence >= 0.85
        and violation_rate <= 0.15
    ):
        return "pass"

    if (
        candidate_level == "strong_candidate"
        and comparable_count >= 500
        and patecon_confidence >= 0.74
        and source_weighted_confidence >= 0.735
        and violation_rate <= 0.26
    ):
        return "strong_soft"

    if (
        llm_expected_action == "mark_mixed"
        and comparable_count >= 500
        and patecon_confidence >= 0.72
        and violation_rate <= 0.29
    ):
        return "mixed_candidate"

    if comparable_count >= 30 and patecon_confidence >= 0.70:
        return "review"

    if comparable_count >= 10 and patecon_confidence >= 0.60:
        return "weak_review"

    return "weak"


# ============================================================
# 单条候选互斥约束评分
# ============================================================

def score_one_constraint(
    constraint: pd.Series,
    pair_groups: Dict[Tuple[str, str], pd.DataFrame],
    weight_map: Dict[Tuple[str, str, str], float],
    max_instances: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:

    cid = str(constraint.get("constraint_id", "")).strip()
    cname = str(constraint.get("constraint_name", cid)).strip()
    ckey = str(constraint.get("constraint_key", "")).strip()

    relation_a = normalize_relation_type(constraint.get("relation_a", ""))
    relation_b = normalize_relation_type(constraint.get("relation_b", ""))
    forbidden = str(constraint.get("forbidden_temporal_relation", "overlap")).strip()

    if not ckey:
        ckey = f"{min(relation_a, relation_b)}_{max(relation_a, relation_b)}_{forbidden}"

    candidate_level = str(constraint.get("candidate_level", "")).strip()
    program_candidate_reason = str(constraint.get("program_candidate_reason", "")).strip()
    program_recommended_action = str(constraint.get("program_recommended_action", "")).strip()

    llm_hard_or_soft = str(constraint.get("hard_or_soft", "soft")).strip()
    llm_expected_action = str(constraint.get("expected_action", "review")).strip()
    llm_reason = str(constraint.get("reason", "")).strip()
    llm_exception = str(constraint.get("exception", "")).strip()

    comparable_count = 0

    # PaTeCon-style fact-level statistics
    patecon_support_count = 0
    patecon_negative_count = 0
    patecon_unknown_count = 0
    patecon_instantiation_count = 0

    # 兼容旧字段
    non_overlap_count = 0
    violation_count = 0

    weighted_instantiation = 0.0
    weighted_support = 0.0

    relation_counter = {
        "before": 0,
        "after": 0,
        "meet": 0,
        "overlap": 0,
        "during": 0,
        "contains": 0,
        "equal": 0,
        "unknown": 0,
    }

    # PaTeCon-style entity-level statistics
    # 一个 same_subject_object 组视为一个 entity-level instantiation
    entity_instantiation_count = 0
    entity_support_count = 0
    entity_negative_count = 0
    entity_unknown_count = 0

    trigger_rows: List[Dict[str, Any]] = []
    violation_rows: List[Dict[str, Any]] = []

    for (s, o), g in pair_groups.items():
        ga = g[g["relation_type"] == relation_a]
        gb = g[g["relation_type"] == relation_b]

        if ga.empty or gb.empty:
            continue

        group_has_comparable = False
        group_has_negative = False
        group_has_positive = False
        group_all_unknown = True

        for _, ia in ga.iterrows():
            for _, ib in gb.iterrows():
                interval_relation = relation_between_intervals(
                    ia["start_period"],
                    ia["end_period"],
                    ib["start_period"],
                    ib["end_period"],
                )

                comparable_count += 1
                group_has_comparable = True

                relation_counter[interval_relation] = relation_counter.get(interval_relation, 0) + 1

                truth = patecon_truth_for_mutex(interval_relation, forbidden)
                w = interval_pair_weight(ia, ib, weight_map)

                if truth == 1:
                    patecon_support_count += 1
                    patecon_instantiation_count += 1
                    non_overlap_count += 1

                    weighted_support += w
                    weighted_instantiation += w

                    group_has_positive = True
                    group_all_unknown = False

                elif truth == -1:
                    patecon_negative_count += 1
                    patecon_instantiation_count += 1
                    violation_count += 1

                    weighted_instantiation += w

                    group_has_negative = True
                    group_all_unknown = False

                else:
                    patecon_unknown_count += 1
                    relation_counter["unknown"] = relation_counter.get("unknown", 0) + 1

                is_violation = truth == -1

                if len(trigger_rows) < max_instances:
                    trigger_rows.append({
                        "constraint_id": cid,
                        "constraint_key": ckey,
                        "constraint_name": cname,
                        "subject_org_id": s,
                        "object_org_id": o,
                        "relation_a": relation_a,
                        "relation_b": relation_b,
                        "interval_a_id": ia["interval_id"],
                        "interval_b_id": ib["interval_id"],
                        "a_start_time": ia["start_time"],
                        "a_end_time": ia["end_time"],
                        "b_start_time": ib["start_time"],
                        "b_end_time": ib["end_time"],
                        "interval_relation": interval_relation,
                        "patecon_truth": truth,
                        "is_violation": is_violation,
                        "pair_weight": round(w, 6),
                    })

                if is_violation and len(violation_rows) < max_instances:
                    violation_rows.append({
                        "constraint_id": cid,
                        "constraint_key": ckey,
                        "constraint_name": cname,
                        "subject_org_id": s,
                        "object_org_id": o,
                        "relation_a": relation_a,
                        "relation_b": relation_b,
                        "interval_a_id": ia["interval_id"],
                        "interval_b_id": ib["interval_id"],
                        "a_start_time": ia["start_time"],
                        "a_end_time": ia["end_time"],
                        "b_start_time": ib["start_time"],
                        "b_end_time": ib["end_time"],
                        "interval_relation": interval_relation,
                        "patecon_truth": truth,
                        "pair_weight": round(w, 6),
                    })

        # entity-level confidence:
        # 若某个 same_subject_object 组至少有一个可判断实例，则计入 entity_instantiation。
        # 如果组内没有 violation，则认为该组支持候选互斥关系。
        if group_has_comparable:
            if group_all_unknown:
                entity_unknown_count += 1
            else:
                entity_instantiation_count += 1

                if group_has_negative:
                    entity_negative_count += 1
                elif group_has_positive:
                    entity_support_count += 1

    if patecon_instantiation_count > 0:
        patecon_confidence = patecon_support_count / patecon_instantiation_count
        violation_rate = patecon_negative_count / patecon_instantiation_count
    else:
        patecon_confidence = 0.0
        violation_rate = 0.0

    if weighted_instantiation > 0:
        source_weighted_confidence = weighted_support / weighted_instantiation
    else:
        source_weighted_confidence = 0.0

    if entity_instantiation_count > 0:
        entity_level_confidence = entity_support_count / entity_instantiation_count
        entity_violation_rate = entity_negative_count / entity_instantiation_count
    else:
        entity_level_confidence = 0.0
        entity_violation_rate = 0.0

    auto_filter_status = decide_auto_filter_status(
        candidate_level=candidate_level,
        comparable_count=patecon_instantiation_count,
        patecon_confidence=patecon_confidence,
        source_weighted_confidence=source_weighted_confidence,
        violation_rate=violation_rate,
        llm_expected_action=llm_expected_action,
    )

    score_row = {
        "constraint_id": cid,
        "constraint_key": ckey,
        "constraint_name": cname,
        "relation_a": relation_a,
        "relation_b": relation_b,
        "scope": str(constraint.get("scope", "same_subject_object")).strip(),
        "forbidden_temporal_relation": forbidden,

        # 原有统计
        "comparable_count": comparable_count,
        "non_overlap_count": non_overlap_count,
        "violation_count": violation_count,

        # PaTeCon-style fact-level statistics
        "patecon_support_count": patecon_support_count,
        "patecon_negative_count": patecon_negative_count,
        "patecon_unknown_count": patecon_unknown_count,
        "patecon_instantiation_count": patecon_instantiation_count,
        "patecon_confidence": round(patecon_confidence, 6),
        "source_weighted_confidence": round(source_weighted_confidence, 6),
        "violation_rate": round(violation_rate, 6),

        # PaTeCon-style entity-level statistics
        "entity_support_count": entity_support_count,
        "entity_negative_count": entity_negative_count,
        "entity_unknown_count": entity_unknown_count,
        "entity_instantiation_count": entity_instantiation_count,
        "entity_level_confidence": round(entity_level_confidence, 6),
        "entity_violation_rate": round(entity_violation_rate, 6),

        # temporal relation distribution
        "before_count": relation_counter.get("before", 0),
        "after_count": relation_counter.get("after", 0),
        "meet_count": relation_counter.get("meet", 0),
        "overlap_count": relation_counter.get("overlap", 0),
        "during_count": relation_counter.get("during", 0),
        "contains_count": relation_counter.get("contains", 0),
        "equal_count": relation_counter.get("equal", 0),
        "unknown_count": relation_counter.get("unknown", 0),

        # LLM and program metadata
        "candidate_level": candidate_level,
        "program_candidate_reason": program_candidate_reason,
        "program_recommended_action": program_recommended_action,
        "llm_hard_or_soft": llm_hard_or_soft,
        "llm_expected_action": llm_expected_action,
        "llm_reason": llm_reason,
        "llm_exception": llm_exception,

        "auto_filter_status": auto_filter_status,
    }

    return score_row, trigger_rows, violation_rows


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_instances",
        type=int,
        default=5000,
        help="每条约束最多保存多少条触发/违反样例",
    )
    parser.add_argument(
        "--input_intervals",
        type=str,
        default=INTERVALS_IN,
        help="org_relation_intervals.tsv 路径",
    )
    parser.add_argument(
        "--input_candidates",
        type=str,
        default=CANDIDATES_IN,
        help="mutual_exclusion_llm_candidates.csv 路径",
    )
    parser.add_argument(
        "--input_edges",
        type=str,
        default=EDGES_IN,
        help="relation_edges_scored.csv 路径",
    )
    args = parser.parse_args()

    ensure_dirs()

    log("========== 第 10 步：为 LLM 候选互斥约束计算 PaTeCon-style 置信度 ==========")
    log(f"[INFO] Project root: {PROJECT_ROOT}")
    log(f"[INFO] Intervals: {args.input_intervals}")
    log(f"[INFO] Candidates: {args.input_candidates}")
    log(f"[INFO] Edges: {args.input_edges}")

    intervals = read_intervals(args.input_intervals)
    log(f"[OK] 读取区间: {args.input_intervals}, rows={len(intervals)}")

    if not os.path.exists(args.input_candidates):
        raise FileNotFoundError(f"未找到候选互斥约束文件: {args.input_candidates}")

    candidates = pd.read_csv(args.input_candidates, low_memory=False)
    log(f"[OK] 读取 LLM 候选约束: {args.input_candidates}, rows={len(candidates)}")

    if candidates.empty:
        log("[WARN] 候选约束为空，生成空评分文件。")
        pd.DataFrame().to_csv(SCORED_OUT, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(TRIGGER_OUT, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(VIOLATION_OUT, index=False, encoding="utf-8-sig")
        return

    # 规范候选关系字段
    if "relation_a" not in candidates.columns or "relation_b" not in candidates.columns:
        raise ValueError("mutual_exclusion_llm_candidates.csv 必须包含 relation_a 和 relation_b 字段。")

    candidates["relation_a"] = candidates["relation_a"].apply(normalize_relation_type)
    candidates["relation_b"] = candidates["relation_b"].apply(normalize_relation_type)

    # 读取边文件，用于来源加权
    if os.path.exists(args.input_edges):
        edges = pd.read_csv(args.input_edges, low_memory=False)
        log(f"[OK] 读取关系边用于来源加权: {args.input_edges}, rows={len(edges)}")
    else:
        edges = pd.DataFrame()
        log("[WARN] 未找到 relation_edges_scored.csv，来源加权全部按 1.0 处理。")

    weight_map = build_interval_weight_map(edges)

    # same_subject_object 分组
    pair_groups = {
        key: g.copy()
        for key, g in intervals.groupby(["subject_org_id", "object_org_id"], sort=False)
    }
    log(f"[STAT] same_subject_object pair groups: {len(pair_groups)}")

    score_rows: List[Dict[str, Any]] = []
    trigger_all: List[Dict[str, Any]] = []
    violation_all: List[Dict[str, Any]] = []

    for _, c in candidates.iterrows():
        score_row, trigger_rows, violation_rows = score_one_constraint(
            constraint=c,
            pair_groups=pair_groups,
            weight_map=weight_map,
            max_instances=args.max_instances,
        )

        score_rows.append(score_row)
        trigger_all.extend(trigger_rows)
        violation_all.extend(violation_rows)

        log(
            f"[SCORE] {score_row['constraint_id']} "
            f"{score_row['relation_a']} vs {score_row['relation_b']} "
            f"instantiation={score_row['patecon_instantiation_count']} "
            f"support={score_row['patecon_support_count']} "
            f"negative={score_row['patecon_negative_count']} "
            f"confidence={score_row['patecon_confidence']} "
            f"weighted_conf={score_row['source_weighted_confidence']} "
            f"violation_rate={score_row['violation_rate']} "
            f"status={score_row['auto_filter_status']}"
        )

    scored_df = pd.DataFrame(score_rows)
    trigger_df = pd.DataFrame(trigger_all)
    violation_df = pd.DataFrame(violation_all)

    scored_df.to_csv(SCORED_OUT, index=False, encoding="utf-8-sig")
    trigger_df.to_csv(TRIGGER_OUT, index=False, encoding="utf-8-sig")
    violation_df.to_csv(VIOLATION_OUT, index=False, encoding="utf-8-sig")

    log(f"[OK] 已生成: {SCORED_OUT}, rows={len(scored_df)}")
    log(f"[OK] 已生成: {TRIGGER_OUT}, rows={len(trigger_df)}")
    log(f"[OK] 已生成: {VIOLATION_OUT}, rows={len(violation_df)}")

    if not scored_df.empty and "auto_filter_status" in scored_df.columns:
        status_counts = scored_df["auto_filter_status"].value_counts(dropna=False).to_dict()
        log(f"[STAT] auto_filter_status 分布: {status_counts}")

    log("========== 第 10 步完成 ==========")


if __name__ == "__main__":
    main()
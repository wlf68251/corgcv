# scripts/09_generate_llm_constraints.py
# -*- coding: utf-8 -*-

"""
第 09 步：基于 LLM 生成候选互斥约束。

核心方法：
1. 读取当前图谱真实关系类型。
2. 读取月度关系区间，统计 same_subject_object 范围内的关系对时间共现。
3. 程序先生成“候选互斥关系对池”，避免 LLM 自由生成过宽或过窄的关系对。
4. Prompt 中明确说明可用时序联系，但要求 LLM 最终只围绕“互斥关系”输出候选。
5. LLM 只生成候选互斥约束，不直接作为最终约束。
6. 后续第 10 步再用 PaTeCon-style 逻辑计算数据置信度。

输入：
    data/processed/relation_edges_scored.csv
    data/processed/org_relation_intervals.tsv

输出：
    data/processed/relation_inventory_for_llm.csv
    data/processed/relation_examples_for_llm.csv
    data/processed/relation_pair_temporal_summary_for_llm.csv
    data/processed/candidate_mutex_pair_pool_for_llm.csv
    data/processed/mutual_exclusion_prompt.txt
    data/processed/mutual_exclusion_raw_response.txt
    data/processed/mutual_exclusion_llm_candidates.json
    data/processed/mutual_exclusion_llm_candidates.csv
    logs/09_generate_llm_constraints.log

运行：
    python scripts/09_generate_llm_constraints.py

只生成 prompt，不调用 LLM：
    python scripts/09_generate_llm_constraints.py --no_llm

不重新调用 LLM，直接解析已有 raw_response：
    python scripts/09_generate_llm_constraints.py --parse_existing

环境变量：
    DASHSCOPE_API_KEY 或 OPENAI_API_KEY
    BASE_URL，默认 https://dashscope.aliyuncs.com/compatible-mode/v1
    MODEL_NAME，默认 qwen-turbo
"""

import os
import re
import ast
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

EDGES_IN = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")
INTERVALS_IN = os.path.join(PROCESSED_DIR, "org_relation_intervals.tsv")

INVENTORY_OUT = os.path.join(PROCESSED_DIR, "relation_inventory_for_llm.csv")
EXAMPLES_OUT = os.path.join(PROCESSED_DIR, "relation_examples_for_llm.csv")
TEMPORAL_SUMMARY_OUT = os.path.join(PROCESSED_DIR, "relation_pair_temporal_summary_for_llm.csv")
CANDIDATE_POOL_OUT = os.path.join(PROCESSED_DIR, "candidate_mutex_pair_pool_for_llm.csv")

PROMPT_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_prompt.txt")
RAW_RESPONSE_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_raw_response.txt")
RAW_RESPONSE_DEBUG_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_raw_response_head_tail.txt")

CANDIDATES_JSON_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_llm_candidates.json")
CANDIDATES_CSV_OUT = os.path.join(PROCESSED_DIR, "mutual_exclusion_llm_candidates.csv")

LOG_OUT = os.path.join(LOG_DIR, "09_generate_llm_constraints.log")


RELATION_DESCRIPTIONS = {
    "verbal_cooperation": "言语合作，包括表达支持、协商、会谈、声明合作意愿等。",
    "material_cooperation": "实质合作，包括援助、签署协议、实际经济或军事合作等。",
    "verbal_conflict": "言语冲突，包括批评、谴责、威胁、外交抗议等。",
    "material_conflict": "实质冲突，包括制裁、攻击、拘捕、军事行动、暴力冲突等。",
    "mixed_relation": "混合或难以单独归类的复杂关系，通常表示合作与冲突并存或关系语义不稳定。",

    "01": "公开声明或一般性表态。",
    "02": "呼吁、请求或要求。",
    "03": "表达合作意愿。",
    "04": "协商、磋商或外交接触。",
    "05": "开展外交合作。",
    "06": "开展实质合作。",
    "07": "提供援助或支持。",
    "08": "让步、妥协或同意。",
    "09": "调查或寻求信息。",
    "10": "要求、命令或施压。",
    "11": "反对、批评或拒绝。",
    "12": "拒绝合作或拒绝请求。",
    "13": "威胁、警告或强硬表态。",
    "14": "抗议、示威或政治冲突。",
    "15": "使用非常规暴力或恐怖行为。",
    "16": "减少关系、制裁或断绝合作。",
    "17": "胁迫、拘捕或强制行动。",
    "18": "攻击、军事打击或武装行动。",
    "19": "战斗或军事冲突。",
    "20": "大规模暴力、战争或严重冲突。",
}


RELATION_ZH = {
    "verbal_cooperation": "言语合作",
    "material_cooperation": "实质合作",
    "verbal_conflict": "言语冲突",
    "material_conflict": "实质冲突",
    "mixed_relation": "混合关系",

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
}


# 关系分组：用于候选关系对池生成，不是最终规则
BROAD_RELATIONS = {"01"}

PROCESS_CHAIN_PAIRS = {
    ("01", "02"),
    ("01", "03"),
    ("01", "04"),
    ("01", "05"),
    ("02", "03"),
    ("02", "04"),
    ("02", "05"),
    ("03", "04"),
    ("03", "05"),
    ("04", "05"),
    ("06", "07"),
    ("06", "08"),
    ("07", "08"),
}

VERBAL_DIPLOMACY = {"03", "04", "05"}
MATERIAL_COOPERATION = {"06", "07", "08"}
COOPERATION_ALL = {"03", "04", "05", "06", "07", "08"}

VERBAL_CONFLICT = {"10", "11", "12", "13", "14"}
SEVERE_CONFLICT = {"16", "17", "18", "19", "20"}
MILITARY_OR_COERCIVE = {"17", "18", "19", "20"}
SANCTION_OR_BREAK = {"16"}
THREAT_REJECT_CRITICIZE = {"11", "12", "13"}

INTERNAL_CONFLICT_PAIRS = {
    ("10", "11"),
    ("10", "12"),
    ("11", "12"),
    ("11", "13"),
    ("12", "13"),
    ("13", "14"),
    ("16", "17"),
    ("17", "18"),
    ("18", "19"),
    ("19", "20"),
}


def log(msg: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(text)
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到输入文件: {path}")
    return pd.read_csv(path, low_memory=False)


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def normalize_relation_type(x: Any) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    if x.isdigit() and len(x) == 1:
        x = "0" + x
    return x


def ordered_pair(a: str, b: str) -> Tuple[str, str]:
    a = normalize_relation_type(a)
    b = normalize_relation_type(b)
    return tuple(sorted([a, b]))


def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_relation_col(df: pd.DataFrame) -> str:
    col = get_col(df, ["relation_type", "relation_code", "event_root_code", "cameo_root", "cameo_code"])
    if col is None:
        raise ValueError("找不到关系字段：relation_type / relation_code / event_root_code / cameo_root / cameo_code")
    return col


def detect_month_col(df: pd.DataFrame) -> Optional[str]:
    return get_col(df, ["event_month", "month", "event_date", "date"])


def parse_month(x: Any) -> pd.Period:
    """
    将各种时间格式统一解析为 pandas Period(月粒度)。

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

    m = re.match(r"^(\d{4})-(\d{1,2})(?:-\d{1,2})?$", s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
        return pd.NaT

    digits = re.sub(r"\D", "", s)

    if len(digits) >= 8:
        year = int(digits[:4])
        month = int(digits[4:6])
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
        return pd.NaT

    if len(digits) == 6:
        year = int(digits[:4])
        month = int(digits[4:6])
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
        return pd.NaT

    if len(digits) == 7:
        year = int(digits[:4])
        month = int(digits[4:6])
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
        return pd.NaT

    return pd.NaT


def relation_between_intervals(
    a_start: pd.Period,
    a_end: pd.Period,
    b_start: pd.Period,
    b_end: pd.Period,
) -> str:
    """
    判断两个区间之间的时间关系。
    月粒度下，meet 表示 A 结束月份的下一个月刚好是 B 开始月份，或反过来。
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


def classify_mutex_pair(relation_a: str, relation_b: str) -> Tuple[str, str, str]:
    """
    对关系对进行分层。

    返回：
        candidate_level:
            strong_candidate
            weak_candidate
            blocked
        candidate_reason:
            中文说明
        recommended_action:
            review / mark_mixed / mark_mixed_or_downgrade
    """
    a = normalize_relation_type(relation_a)
    b = normalize_relation_type(relation_b)

    if not a or not b:
        return "blocked", "关系类型为空。", "review"

    if a == b:
        return "blocked", "同一关系类型不生成互斥约束。", "review"

    pair = ordered_pair(a, b)

    if a in BROAD_RELATIONS or b in BROAD_RELATIONS:
        return "blocked", "01 公开声明过于宽泛，不作为互斥约束的一端。", "review"

    if pair in PROCESS_CHAIN_PAIRS:
        return "blocked", "该关系对通常是外交、合作或互动过程链条，不应作为互斥约束。", "review"

    if pair in INTERNAL_CONFLICT_PAIRS:
        return "blocked", "该关系对通常是同一冲突过程中的升级或并发，不直接作为互斥约束。", "review"

    if a in VERBAL_CONFLICT and b in VERBAL_CONFLICT:
        return "blocked", "言语冲突内部关系通常可共存，不直接互斥。", "review"

    if a in SEVERE_CONFLICT and b in SEVERE_CONFLICT:
        return "blocked", "实质冲突内部关系通常表示升级或并发，不直接互斥。", "review"

    if (a in MATERIAL_COOPERATION and b in SEVERE_CONFLICT) or (
        b in MATERIAL_COOPERATION and a in SEVERE_CONFLICT
    ):
        return (
            "strong_candidate",
            "实质合作、援助或让步与制裁、强制、攻击、战斗等强冲突关系同月重叠时，可能表示复杂或矛盾关系。",
            "mark_mixed_or_downgrade",
        )

    if (a in VERBAL_DIPLOMACY and b in MILITARY_OR_COERCIVE) or (
        b in VERBAL_DIPLOMACY and a in MILITARY_OR_COERCIVE
    ):
        return (
            "weak_candidate",
            "外交表态、协商或外交合作与强制行动、攻击、战斗同月重叠时，通常更适合标记 mixed 或 review，而不是直接隐藏。",
            "mark_mixed",
        )

    if (a in {"04", "05"} and b in SANCTION_OR_BREAK) or (
        b in {"04", "05"} and a in SANCTION_OR_BREAK
    ):
        return (
            "weak_candidate",
            "协商或外交合作与制裁断交同月重叠时，可能表示关系复杂化，适合复核或标记 mixed。",
            "mark_mixed",
        )

    if (a in COOPERATION_ALL and b in THREAT_REJECT_CRITICIZE) or (
        b in COOPERATION_ALL and a in THREAT_REJECT_CRITICIZE
    ):
        return (
            "weak_candidate",
            "合作类关系与反对、拒绝、威胁类关系同月重叠时，可能表示复杂关系，但不宜直接判为硬冲突。",
            "review",
        )

    if (a in {"07", "08"} and b in {"14", "15"}) or (
        b in {"07", "08"} and a in {"14", "15"}
    ):
        return (
            "weak_candidate",
            "援助、让步与抗议或非常规暴力同月重叠时可能表示关系复杂，需要复核。",
            "review",
        )

    return "blocked", "该关系对不属于当前方法优先考虑的互斥候选范围。", "review"


def is_semantically_plausible_mutex_pair(relation_a: str, relation_b: str) -> bool:
    level, _, _ = classify_mutex_pair(relation_a, relation_b)
    return level in {"strong_candidate", "weak_candidate"}


def normalize_expected_action_by_pair(relation_a: str, relation_b: str, expected_action: str) -> str:
    level, _, recommended_action = classify_mutex_pair(relation_a, relation_b)

    if level == "strong_candidate":
        return recommended_action

    if level == "weak_candidate":
        return recommended_action

    allowed_actions = {"review", "downgrade", "hide", "mark_mixed", "mark_mixed_or_downgrade"}
    return expected_action if expected_action in allowed_actions else "review"


def build_relation_inventory(edges: pd.DataFrame) -> pd.DataFrame:
    relation_col = detect_relation_col(edges)
    month_col = detect_month_col(edges)

    edges = edges.copy()
    edges["_relation_type_norm"] = edges[relation_col].apply(normalize_relation_type)

    count_df = (
        edges.groupby("_relation_type_norm")
        .size()
        .reset_index(name="event_count")
        .rename(columns={"_relation_type_norm": "relation_type"})
    )

    interval_count_map = {}

    if os.path.exists(INTERVALS_IN):
        try:
            intervals = pd.read_csv(INTERVALS_IN, sep="\t", header=None, dtype=str)
            if intervals.shape[1] >= 5:
                intervals = intervals.iloc[:, :5].copy()
                intervals.columns = ["subject_org_id", "relation_type", "object_org_id", "start_time", "end_time"]
                intervals["relation_type"] = intervals["relation_type"].apply(normalize_relation_type)
                interval_count_map = intervals.groupby("relation_type").size().to_dict()
        except Exception as e:
            log(f"[WARN] 读取区间文件失败，仅使用边文件统计 interval_count: {e}")

    subj_col = get_col(edges, ["subject_name", "source_name", "Actor1Name", "subject_org_id"])
    obj_col = get_col(edges, ["object_name", "target_name", "Actor2Name", "object_org_id"])
    conf_col = get_col(edges, ["confidence", "final_confidence", "score"])
    event_count_col = get_col(edges, ["event_count", "num_mentions", "NumMentions"])

    examples_by_rel = {}
    high_conf_by_rel = {}

    if subj_col and obj_col:
        for r, g in edges.groupby("_relation_type_norm"):
            gg = g.copy()

            sort_cols = []
            ascending = []

            if event_count_col:
                sort_cols.append(event_count_col)
                ascending.append(False)

            if conf_col:
                sort_cols.append(conf_col)
                ascending.append(False)

            if sort_cols:
                gg = gg.sort_values(sort_cols, ascending=ascending)

            samples = []
            for _, e in gg.head(5).iterrows():
                m = str(e.get(month_col, ""))[:7] if month_col else ""
                samples.append(f"{e.get(subj_col, '')} --{r}--> {e.get(obj_col, '')} {m}")
            examples_by_rel[r] = " | ".join(samples)

            if conf_col:
                hg = g.sort_values(conf_col, ascending=False).head(5)
            else:
                hg = g.head(5)

            high_samples = []
            for _, e in hg.iterrows():
                m = str(e.get(month_col, ""))[:7] if month_col else ""
                c = e.get(conf_col, "")
                high_samples.append(f"{e.get(subj_col, '')} --{r}--> {e.get(obj_col, '')} {m} confidence={c}")
            high_conf_by_rel[r] = " | ".join(high_samples)

    records = []

    for _, row in count_df.iterrows():
        r = row["relation_type"]
        records.append({
            "relation_type": r,
            "relation_code": r,
            "relation_name_zh": RELATION_ZH.get(r, r),
            "relation_description": RELATION_DESCRIPTIONS.get(r, f"当前数据中出现的关系类型 {r}。"),
            "event_count": int(row["event_count"]),
            "interval_count": int(interval_count_map.get(r, 0)),
            "top_subject_object_examples": examples_by_rel.get(r, ""),
            "high_confidence_examples": high_conf_by_rel.get(r, ""),
        })

    inventory = pd.DataFrame(records)
    inventory = inventory.sort_values(["event_count", "relation_type"], ascending=[False, True])
    return inventory


def build_relation_examples(
    edges: pd.DataFrame,
    inventory: pd.DataFrame,
    max_examples_per_relation: int = 5,
) -> pd.DataFrame:
    relation_col = detect_relation_col(edges)
    month_col = detect_month_col(edges)

    subj_id_col = get_col(edges, ["subject_org_id", "subject_id", "source_org_id"])
    obj_id_col = get_col(edges, ["object_org_id", "object_id", "target_org_id"])
    subj_name_col = get_col(edges, ["subject_name", "source_name", "Actor1Name"])
    obj_name_col = get_col(edges, ["object_name", "target_name", "Actor2Name"])
    conf_col = get_col(edges, ["confidence", "final_confidence", "score"])
    event_count_col = get_col(edges, ["event_count", "num_mentions", "NumMentions"])
    source_col = get_col(edges, ["sources", "source_datasets", "dataset"])

    edges = edges.copy()
    edges["_relation_type_norm"] = edges[relation_col].apply(normalize_relation_type)

    records = []

    for rel in inventory["relation_type"].astype(str).tolist():
        g = edges[edges["_relation_type_norm"] == rel].copy()
        if g.empty:
            continue

        sort_cols = []
        ascending = []

        if conf_col:
            sort_cols.append(conf_col)
            ascending.append(False)

        if event_count_col:
            sort_cols.append(event_count_col)
            ascending.append(False)

        if sort_cols:
            g = g.sort_values(sort_cols, ascending=ascending)

        for _, row in g.head(max_examples_per_relation).iterrows():
            subject_label = row.get(subj_name_col, row.get(subj_id_col, "")) if subj_name_col else row.get(subj_id_col, "")
            object_label = row.get(obj_name_col, row.get(obj_id_col, "")) if obj_name_col else row.get(obj_id_col, "")
            month = str(row.get(month_col, ""))[:7] if month_col else ""

            records.append({
                "relation_type": rel,
                "subject_org_id": row.get(subj_id_col, "") if subj_id_col else "",
                "object_org_id": row.get(obj_id_col, "") if obj_id_col else "",
                "subject_name": subject_label,
                "object_name": object_label,
                "month": month,
                "confidence": row.get(conf_col, "") if conf_col else "",
                "event_count": row.get(event_count_col, "") if event_count_col else "",
                "sources": row.get(source_col, "") if source_col else "",
                "example_text": f"{subject_label} --{rel}--> {object_label} {month}",
            })

    return pd.DataFrame(records)


def read_intervals_for_temporal_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        log(f"[WARN] 未找到区间文件，无法生成关系对时间摘要: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t", header=None, dtype=str)

    if df.shape[1] < 5:
        log("[WARN] org_relation_intervals.tsv 少于 5 列，无法生成关系对时间摘要。")
        return pd.DataFrame()

    df = df.iloc[:, :5].copy()
    df.columns = ["subject_org_id", "relation_type", "object_org_id", "start_time", "end_time"]

    df["relation_type"] = df["relation_type"].apply(normalize_relation_type)
    df["start_period"] = df["start_time"].apply(parse_month)
    df["end_period"] = df["end_time"].apply(parse_month)

    bad_time = df["start_period"].isna() | df["end_period"].isna()
    if bad_time.any():
        bad_path = os.path.join(PROCESSED_DIR, "bad_interval_time_rows_for_09.csv")
        df.loc[bad_time].to_csv(bad_path, index=False, encoding="utf-8-sig")
        log(f"[WARN] 09 中发现无法解析的区间时间 {bad_time.sum()} 条，已保存到: {bad_path}")
        df = df.loc[~bad_time].copy()

    bad_order = df["end_period"] < df["start_period"]
    if bad_order.any():
        log(f"[WARN] 09 中发现 end_time < start_time 的区间 {bad_order.sum()} 条，已自动交换。")

        old_start_period = df.loc[bad_order, "start_period"].copy()
        old_start_time = df.loc[bad_order, "start_time"].copy()

        df.loc[bad_order, "start_period"] = df.loc[bad_order, "end_period"]
        df.loc[bad_order, "start_time"] = df.loc[bad_order, "end_time"]

        df.loc[bad_order, "end_period"] = old_start_period
        df.loc[bad_order, "end_time"] = old_start_time

    df["start_time"] = df["start_period"].astype(str)
    df["end_time"] = df["end_period"].astype(str)

    return df


def build_relation_pair_temporal_summary(
    intervals: pd.DataFrame,
    max_pairs_for_summary: int = 500,
    max_examples_per_pair: int = 3,
) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame(columns=[
            "relation_a",
            "relation_b",
            "comparable_count",
            "before_count",
            "after_count",
            "meet_count",
            "overlap_count",
            "during_count",
            "contains_count",
            "equal_count",
            "non_overlap_count",
            "violation_like_overlap_count",
            "overlap_rate",
            "non_overlap_rate",
            "top_overlap_examples",
            "top_non_overlap_examples",
        ])

    pair_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    relation_counts = intervals["relation_type"].value_counts().to_dict()

    for (s, o), g in intervals.groupby(["subject_org_id", "object_org_id"]):
        if len(g) < 2:
            continue

        rows = g.to_dict("records")

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                ri = rows[i]
                rj = rows[j]

                rel_i = normalize_relation_type(ri["relation_type"])
                rel_j = normalize_relation_type(rj["relation_type"])

                if not rel_i or not rel_j or rel_i == rel_j:
                    continue

                rel_a, rel_b = ordered_pair(rel_i, rel_j)

                temporal_rel = relation_between_intervals(
                    ri["start_period"],
                    ri["end_period"],
                    rj["start_period"],
                    rj["end_period"],
                )

                key = (rel_a, rel_b)

                if key not in pair_stats:
                    pair_stats[key] = {
                        "relation_a": rel_a,
                        "relation_b": rel_b,
                        "comparable_count": 0,
                        "before_count": 0,
                        "after_count": 0,
                        "meet_count": 0,
                        "overlap_count": 0,
                        "during_count": 0,
                        "contains_count": 0,
                        "equal_count": 0,
                        "top_overlap_examples": [],
                        "top_non_overlap_examples": [],
                    }

                stat = pair_stats[key]
                stat["comparable_count"] += 1

                count_key = f"{temporal_rel}_count"
                if count_key in stat:
                    stat[count_key] += 1

                example = (
                    f"{s} --{rel_i}--> {o} [{ri['start_time']},{ri['end_time']}] "
                    f"{temporal_rel} "
                    f"{s} --{rel_j}--> {o} [{rj['start_time']},{rj['end_time']}]"
                )

                if temporal_rel in {"overlap", "during", "contains", "equal"}:
                    if len(stat["top_overlap_examples"]) < max_examples_per_pair:
                        stat["top_overlap_examples"].append(example)
                else:
                    if len(stat["top_non_overlap_examples"]) < max_examples_per_pair:
                        stat["top_non_overlap_examples"].append(example)

    records = []

    for _, stat in pair_stats.items():
        comparable = stat["comparable_count"]

        non_overlap_count = (
            stat["before_count"]
            + stat["after_count"]
            + stat["meet_count"]
        )

        violation_like_overlap_count = (
            stat["overlap_count"]
            + stat["during_count"]
            + stat["contains_count"]
            + stat["equal_count"]
        )

        overlap_rate = violation_like_overlap_count / comparable if comparable > 0 else 0.0
        non_overlap_rate = non_overlap_count / comparable if comparable > 0 else 0.0

        relation_a = stat["relation_a"]
        relation_b = stat["relation_b"]

        records.append({
            "relation_a": relation_a,
            "relation_b": relation_b,
            "relation_a_total_interval_count": int(relation_counts.get(relation_a, 0)),
            "relation_b_total_interval_count": int(relation_counts.get(relation_b, 0)),
            "comparable_count": comparable,
            "before_count": stat["before_count"],
            "after_count": stat["after_count"],
            "meet_count": stat["meet_count"],
            "overlap_count": stat["overlap_count"],
            "during_count": stat["during_count"],
            "contains_count": stat["contains_count"],
            "equal_count": stat["equal_count"],
            "non_overlap_count": non_overlap_count,
            "violation_like_overlap_count": violation_like_overlap_count,
            "overlap_rate": round(overlap_rate, 6),
            "non_overlap_rate": round(non_overlap_rate, 6),
            "top_overlap_examples": " | ".join(stat["top_overlap_examples"]),
            "top_non_overlap_examples": " | ".join(stat["top_non_overlap_examples"]),
        })

    summary = pd.DataFrame(records)

    if summary.empty:
        return summary

    summary = summary.sort_values(
        ["comparable_count", "non_overlap_rate", "overlap_rate"],
        ascending=[False, False, False],
    )

    if max_pairs_for_summary and max_pairs_for_summary > 0:
        summary = summary.head(max_pairs_for_summary).copy()

    return summary


def build_candidate_mutex_pair_pool(
    inventory: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    min_pair_comparable: int = 5,
    min_relation_interval_count: int = 10,
    max_candidate_pairs: int = 80,
) -> pd.DataFrame:
    """
    生成候选互斥关系对池。

    关键点：
    1. LLM 不再自由创造 relation_a/relation_b。
    2. 程序先根据语义分组和时间统计构造候选池。
    3. LLM 只能从候选池中挑选并解释。
    """
    relations = inventory["relation_type"].astype(str).tolist()

    interval_count_map = {
        str(row["relation_type"]): int(row.get("interval_count", 0))
        for _, row in inventory.iterrows()
    }

    temporal_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if temporal_summary is not None and not temporal_summary.empty:
        for _, row in temporal_summary.iterrows():
            a, b = ordered_pair(row["relation_a"], row["relation_b"])
            temporal_map[(a, b)] = row.to_dict()

    records = []

    for i in range(len(relations)):
        for j in range(i + 1, len(relations)):
            a, b = ordered_pair(relations[i], relations[j])
            level, reason, action = classify_mutex_pair(a, b)

            if level == "blocked":
                continue

            ts = temporal_map.get((a, b), {})

            comparable_count = int(ts.get("comparable_count", 0) or 0)
            overlap_rate = float(ts.get("overlap_rate", 0.0) or 0.0)
            non_overlap_rate = float(ts.get("non_overlap_rate", 0.0) or 0.0)

            a_interval_count = int(interval_count_map.get(a, 0))
            b_interval_count = int(interval_count_map.get(b, 0))

            # 强候选即使 comparable 较少也可进入候选池，但至少两个关系本身要有一定规模。
            keep = False

            if comparable_count >= min_pair_comparable:
                keep = True
            elif level == "strong_candidate" and min(a_interval_count, b_interval_count) >= min_relation_interval_count:
                keep = True
            elif level == "weak_candidate" and comparable_count > 0:
                keep = True

            if not keep:
                continue

            pair_id = f"{a}_{b}_overlap"

            records.append({
                "pair_id": pair_id,
                "relation_a": a,
                "relation_b": b,
                "relation_a_name_zh": RELATION_ZH.get(a, a),
                "relation_b_name_zh": RELATION_ZH.get(b, b),
                "candidate_level": level,
                "candidate_reason": reason,
                "recommended_action": action,
                "forbidden_temporal_relation": "overlap",
                "scope": "same_subject_object",
                "relation_a_total_interval_count": a_interval_count,
                "relation_b_total_interval_count": b_interval_count,
                "comparable_count": comparable_count,
                "before_count": int(ts.get("before_count", 0) or 0),
                "after_count": int(ts.get("after_count", 0) or 0),
                "meet_count": int(ts.get("meet_count", 0) or 0),
                "overlap_count": int(ts.get("overlap_count", 0) or 0),
                "during_count": int(ts.get("during_count", 0) or 0),
                "contains_count": int(ts.get("contains_count", 0) or 0),
                "equal_count": int(ts.get("equal_count", 0) or 0),
                "non_overlap_count": int(ts.get("non_overlap_count", 0) or 0),
                "violation_like_overlap_count": int(ts.get("violation_like_overlap_count", 0) or 0),
                "overlap_rate": round(overlap_rate, 6),
                "non_overlap_rate": round(non_overlap_rate, 6),
                "top_overlap_examples": safe_str(ts.get("top_overlap_examples", "")),
                "top_non_overlap_examples": safe_str(ts.get("top_non_overlap_examples", "")),
            })

    pool = pd.DataFrame(records)

    if pool.empty:
        return pool

    level_rank = {
        "strong_candidate": 2,
        "weak_candidate": 1,
    }

    pool["_level_rank"] = pool["candidate_level"].map(level_rank).fillna(0)

    pool = pool.sort_values(
        ["_level_rank", "comparable_count", "overlap_rate", "non_overlap_rate"],
        ascending=[False, False, False, False],
    ).drop(columns=["_level_rank"])

    if max_candidate_pairs and max_candidate_pairs > 0:
        pool = pool.head(max_candidate_pairs).copy()

    return pool


def build_prompt(
    inventory: pd.DataFrame,
    examples: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    max_relations_in_prompt: int = 80,
    max_temporal_pairs_in_prompt: int = 80,
    max_candidate_pairs_in_prompt: int = 50,
    max_output_constraints: int = 10,
) -> str:
    inventory_for_prompt = inventory.head(max_relations_in_prompt).copy()

    relation_items = []

    for _, row in inventory_for_prompt.iterrows():
        rel = row["relation_type"]
        exs = examples[examples["relation_type"] == rel]["example_text"].dropna().head(5).tolist()

        relation_items.append({
            "relation_type": rel,
            "relation_code": row.get("relation_code", rel),
            "relation_name_zh": row.get("relation_name_zh", rel),
            "description": row.get("relation_description", ""),
            "event_count": int(row.get("event_count", 0)),
            "interval_count": int(row.get("interval_count", 0)),
            "examples": exs,
        })

    relation_json = json.dumps(relation_items, ensure_ascii=False, indent=2)

    temporal_items = []

    if temporal_summary is not None and not temporal_summary.empty:
        for _, row in temporal_summary.head(max_temporal_pairs_in_prompt).iterrows():
            temporal_items.append({
                "relation_a": row.get("relation_a", ""),
                "relation_b": row.get("relation_b", ""),
                "comparable_count": int(row.get("comparable_count", 0)),
                "non_overlap_count": int(row.get("non_overlap_count", 0)),
                "violation_like_overlap_count": int(row.get("violation_like_overlap_count", 0)),
                "non_overlap_rate": float(row.get("non_overlap_rate", 0.0)),
                "overlap_rate": float(row.get("overlap_rate", 0.0)),
                "before_count": int(row.get("before_count", 0)),
                "after_count": int(row.get("after_count", 0)),
                "meet_count": int(row.get("meet_count", 0)),
                "overlap_count": int(row.get("overlap_count", 0)),
                "during_count": int(row.get("during_count", 0)),
                "contains_count": int(row.get("contains_count", 0)),
                "equal_count": int(row.get("equal_count", 0)),
            })

    temporal_json = json.dumps(temporal_items, ensure_ascii=False, indent=2)

    candidate_items = []

    if candidate_pool is not None and not candidate_pool.empty:
        for _, row in candidate_pool.head(max_candidate_pairs_in_prompt).iterrows():
            candidate_items.append({
                "pair_id": row.get("pair_id", ""),
                "relation_a": row.get("relation_a", ""),
                "relation_b": row.get("relation_b", ""),
                "relation_a_name_zh": row.get("relation_a_name_zh", ""),
                "relation_b_name_zh": row.get("relation_b_name_zh", ""),
                "candidate_level": row.get("candidate_level", ""),
                "candidate_reason": row.get("candidate_reason", ""),
                "recommended_action": row.get("recommended_action", ""),
                "scope": row.get("scope", "same_subject_object"),
                "forbidden_temporal_relation": row.get("forbidden_temporal_relation", "overlap"),
                "comparable_count": int(row.get("comparable_count", 0)),
                "non_overlap_count": int(row.get("non_overlap_count", 0)),
                "violation_like_overlap_count": int(row.get("violation_like_overlap_count", 0)),
                "overlap_rate": float(row.get("overlap_rate", 0.0)),
                "non_overlap_rate": float(row.get("non_overlap_rate", 0.0)),
                "top_overlap_examples": row.get("top_overlap_examples", ""),
                "top_non_overlap_examples": row.get("top_non_overlap_examples", ""),
            })

    candidate_json = json.dumps(candidate_items, ensure_ascii=False, indent=2)

    prompt = f"""
你是一个多源组织机构关系图谱质量校验助手。

任务：
为多源组织机构关系图谱生成“候选互斥约束”。

核心要求：
1. 你最终只能输出“互斥约束候选”。
2. 不要输出因果约束、演化约束、趋势约束、证据质量规则。
3. 约束只用于校验派生关系边，不删除原始 ICEWS/GDELT 事件证据。
4. 主实验 scope 只允许 same_subject_object。
5. forbidden_temporal_relation 优先使用 overlap 或 same_month。
6. 你只能从“候选互斥关系对池”中选择 relation_a 和 relation_b。
7. 不要自行创造候选池之外的 relation_a/relation_b。
8. 输出必须是一个 JSON 数组。
9. 最多输出 {max_output_constraints} 条候选互斥约束。
10. 不要输出 Markdown，不要输出代码块，不要输出解释，不要输出表格。

互斥约束含义：
在同一主体 A、同一客体 B、同一月份或重叠时间窗口内，某两类关系通常不应同时作为两个独立稳定关系并存。
如果二者同时出现，应执行 review、downgrade、hide、mark_mixed 或 mark_mixed_or_downgrade。

注意：
同月共现本身不一定是错误。
很多情况下，同月共现只表示关系复杂，应标记为 mark_mixed 或 review，而不是直接 hide。

可用时序联系定义：
before: A 的结束时间早于 B 的开始时间
after: A 的开始时间晚于 B 的结束时间
meet: A 的结束时间刚好连接 B 的开始时间
overlap: A 与 B 时间区间有交集
during: A 完全处于 B 时间区间内部
contains: A 完全包含 B 时间区间
equal: A 与 B 时间区间完全相同
disjoint: A 与 B 时间区间不重叠，可包含 before 或 after
same_month: A 与 B 在同一月份窗口内出现

重要要求：
虽然上面给出了可用时序联系定义，但你最终只围绕“互斥关系”输出候选。
不要把 before、after、meet、contains、during 等作为独立时序演化规则输出。
这些时序联系只用于帮助你理解“overlap 或 same_month 是否可能构成互斥候选”。

明确反例：
1. 不要生成“同一主体不能同时与多个客体发生某关系”的规则。
2. 不要生成“多个主体不能同时对同一客体发生某关系”的规则。
3. 不要把 verbal_conflict 和 material_conflict 简单互斥。
4. 不要把 verbal_cooperation 和 material_cooperation 简单互斥。
5. 01 公开声明过于宽泛，通常不能作为互斥约束的一端。
6. 以下关系通常是同一外交或合作过程中的不同阶段，不应生成互斥约束：
   01-02、01-03、01-04、01-05、02-03、02-04、02-05、03-04、03-05、04-05。
7. 协商、合作意愿、外交合作同月共现通常表示过程链条，不表示冲突。
8. 言语冲突内部关系、实质冲突内部关系通常表示升级或并发，不要简单互斥。

真实关系类型列表：
{relation_json}

关系对时间共现摘要：
这些统计来自 org_relation_intervals.tsv，只在 same_subject_object 范围内比较：
(A, relation_a, B, t1)
(A, relation_b, B, t2)

字段说明：
- comparable_count: 这两个关系在同一主体-客体范围内可比较的区间对数量。
- non_overlap_count: before + after + meet 的数量。
- violation_like_overlap_count: overlap + during + contains + equal 的数量。
- non_overlap_rate: 不重叠比例。
- overlap_rate: 重叠或包含比例。

注意：
这些时间统计只能辅助你生成候选约束，不能直接作为最终判断。
最终约束会在下一步由程序重新计算 PaTeCon-style confidence，并经过人工筛选。

关系对时间共现摘要：
{temporal_json}

候选互斥关系对池：
你只能从下面的候选关系对池中选择 relation_a 和 relation_b。
candidate_level 的含义：
- strong_candidate: 语义上较强的互斥候选，可进入 PaTeCon-style 评分。
- weak_candidate: 语义上较弱，更适合 review 或 mark_mixed，不适合直接 hide。
recommended_action 是程序建议动作，但你可以在允许范围内微调。

候选互斥关系对池：
{candidate_json}

输出格式：
请只输出如下 JSON 数组，不要输出任何其他文字：
[
  {{
    "constraint_name": "",
    "relation_a": "",
    "relation_b": "",
    "scope": "same_subject_object",
    "forbidden_temporal_relation": "overlap",
    "allowed_temporal_relations": ["before", "after", "disjoint"],
    "hard_or_soft": "soft",
    "expected_action": "mark_mixed",
    "typical_positive_example": "",
    "typical_negative_example": "",
    "exception": "",
    "reason": ""
  }}
]

字段要求：
- relation_a 必须来自候选互斥关系对池。
- relation_b 必须来自候选互斥关系对池。
- relation_a 不能等于 relation_b。
- relation_a/relation_b 的组合必须存在于候选互斥关系对池中。
- scope 必须是 same_subject_object。
- forbidden_temporal_relation 只能选择 overlap 或 same_month。
- allowed_temporal_relations 通常为 ["before", "after", "disjoint"]。
- hard_or_soft 默认 soft，除非极其确定，否则不要使用 hard。
- expected_action 只能从 review、downgrade、hide、mark_mixed、mark_mixed_or_downgrade 中选择。
- weak_candidate 优先使用 review 或 mark_mixed。
- strong_candidate 可以使用 mark_mixed_or_downgrade，但不要直接 hide。
- exception 必须写出可能误判的例外。
- reason 用中文解释互斥理由。
""".strip()

    return prompt


def call_llm(prompt: str) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.getenv("MODEL_NAME", "qwen-turbo")

    if not api_key:
        log("[WARN] 未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，本次只生成 prompt，不调用 LLM。")
        return ""

    try:
        from openai import OpenAI
    except Exception as e:
        log(f"[WARN] 未安装 openai 包，无法调用 LLM: {e}")
        return ""

    log("[INFO] 正在调用 LLM 生成候选互斥约束...")
    log(f"[INFO] BASE_URL: {base_url}")
    log(f"[INFO] MODEL_NAME: {model_name}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你只输出一个 JSON 数组，不输出任何额外解释。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content or ""


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    从 LLM 原始输出中提取 JSON 数组。

    支持：
    1. 纯 JSON 数组
    2. {"constraints": [...]}
    3. ```json ... ```
    4. 前后带解释文字
    5. 单引号 Python literal 风格
    6. 尾逗号
    7. 多个 {...} 对象但没有数组
    """
    if not text or not str(text).strip():
        return []

    raw = str(text).strip()

    raw = re.sub(r"```json", "```", raw, flags=re.IGNORECASE)
    raw = raw.replace("```JSON", "```")

    fence_match = re.search(r"```([\s\S]*?)```", raw)
    if fence_match:
        raw_fenced = fence_match.group(1).strip()
    else:
        raw_fenced = raw

    candidates_text = [raw_fenced]

    if raw_fenced != raw:
        candidates_text.append(raw)

    left = raw.find("[")
    right = raw.rfind("]")
    if left != -1 and right != -1 and right > left:
        candidates_text.append(raw[left:right + 1])

    left_obj = raw.find("{")
    right_obj = raw.rfind("}")
    if left_obj != -1 and right_obj != -1 and right_obj > left_obj:
        candidates_text.append(raw[left_obj:right_obj + 1])

    def clean_json_like(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"//.*", "", s)
        s = re.sub(r",\s*([\]}])", r"\1", s)
        s = s.replace("“", "\"").replace("”", "\"")
        s = s.replace("‘", "'").replace("’", "'")
        return s

    def normalize_loaded_data(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if isinstance(data, dict):
            for key in ["constraints", "rules", "candidates", "data", "result"]:
                if key in data and isinstance(data[key], list):
                    return [x for x in data[key] if isinstance(x, dict)]

        return []

    for cand in candidates_text:
        cand = clean_json_like(cand)

        if not cand:
            continue

        try:
            data = json.loads(cand)
            arr = normalize_loaded_data(data)
            if arr:
                return arr
        except Exception:
            pass

        try:
            data = ast.literal_eval(cand)
            arr = normalize_loaded_data(data)
            if arr:
                return arr
        except Exception:
            pass

    decoder = json.JSONDecoder()

    for i, ch in enumerate(raw):
        if ch not in "[{":
            continue

        sub = clean_json_like(raw[i:])

        try:
            data, _ = decoder.raw_decode(sub)
            arr = normalize_loaded_data(data)
            if arr:
                return arr
        except Exception:
            continue

    object_texts = re.findall(r"\{[\s\S]*?\}", raw)
    objs = []

    for obj_text in object_texts:
        obj_text = clean_json_like(obj_text)

        try:
            obj = json.loads(obj_text)
            if isinstance(obj, dict) and ("relation_a" in obj or "relation_b" in obj):
                objs.append(obj)
        except Exception:
            try:
                obj = ast.literal_eval(obj_text)
                if isinstance(obj, dict) and ("relation_a" in obj or "relation_b" in obj):
                    objs.append(obj)
            except Exception:
                pass

    return objs


def is_meaningless_reason(reason: str) -> bool:
    reason = str(reason or "")

    bad_patterns = [
        "同一主体不能同时与多个客体",
        "同一主体不能同时和多个客体",
        "不能同时与多个客体",
        "多个主体不能同时对同一客体",
        "多个主体不能同时批评",
        "同一组织不能同时与多个组织",
        "同一个主体不能同时对多个对象",
        "一个主体只能有一个客体",
    ]

    return any(p in reason for p in bad_patterns)


def candidate_pool_key_set(candidate_pool: pd.DataFrame) -> set:
    keys = set()

    if candidate_pool is None or candidate_pool.empty:
        return keys

    for _, row in candidate_pool.iterrows():
        a, b = ordered_pair(row["relation_a"], row["relation_b"])
        keys.add((a, b))

    return keys


def filter_candidates(
    raw_candidates: List[Dict[str, Any]],
    valid_relations: set,
    candidate_pool: pd.DataFrame,
) -> List[Dict[str, Any]]:
    allowed_actions = {
        "review",
        "downgrade",
        "hide",
        "mark_mixed",
        "mark_mixed_or_downgrade",
    }

    allowed_temporal = {
        "overlap",
        "same_month",
    }

    pool_keys = candidate_pool_key_set(candidate_pool)

    kept = []
    cid = 1
    seen = set()

    for c in raw_candidates:
        if not isinstance(c, dict):
            continue

        relation_a = normalize_relation_type(c.get("relation_a", ""))
        relation_b = normalize_relation_type(c.get("relation_b", ""))
        scope = str(c.get("scope", "")).strip()
        forbidden = str(c.get("forbidden_temporal_relation", "overlap")).strip()
        hard_or_soft = str(c.get("hard_or_soft", "soft")).strip()
        expected_action = str(c.get("expected_action", "review")).strip()
        reason = str(c.get("reason", "")).strip()
        exception = str(c.get("exception", "")).strip()

        if not relation_a or not relation_b:
            continue

        if relation_a == relation_b:
            continue

        if relation_a.lower() == "any" or relation_b.lower() == "any":
            continue

        if relation_a not in valid_relations or relation_b not in valid_relations:
            continue

        pair = ordered_pair(relation_a, relation_b)

        # 必须来自候选池
        if pool_keys and pair not in pool_keys:
            continue

        if not is_semantically_plausible_mutex_pair(relation_a, relation_b):
            continue

        if scope != "same_subject_object":
            continue

        if forbidden not in allowed_temporal:
            continue

        if expected_action not in allowed_actions:
            expected_action = "review"

        expected_action = normalize_expected_action_by_pair(
            relation_a,
            relation_b,
            expected_action,
        )

        if is_meaningless_reason(reason):
            continue

        if hard_or_soft == "hard" and not exception:
            continue

        if hard_or_soft == "hard" and expected_action == "hide":
            if "证据" not in exception and "低证据" not in reason and "低置信" not in reason:
                continue

        dedup_key = (pair[0], pair[1], forbidden)

        if dedup_key in seen:
            continue

        seen.add(dedup_key)

        level, pair_reason, recommended_action = classify_mutex_pair(relation_a, relation_b)

        c2 = dict(c)
        c2["constraint_id"] = f"LLM_MUTEX_{cid:04d}"
        c2["constraint_key"] = f"{pair[0]}_{pair[1]}_{forbidden}"
        c2["constraint_name"] = c2.get("constraint_name") or f"{RELATION_ZH.get(pair[0], pair[0])}与{RELATION_ZH.get(pair[1], pair[1])}互斥候选"
        c2["relation_a"] = pair[0]
        c2["relation_b"] = pair[1]
        c2["scope"] = scope
        c2["forbidden_temporal_relation"] = forbidden
        c2["hard_or_soft"] = hard_or_soft if hard_or_soft in {"soft", "hard"} else "soft"
        c2["expected_action"] = expected_action
        c2["candidate_level"] = level
        c2["program_candidate_reason"] = pair_reason
        c2["program_recommended_action"] = recommended_action
        c2["reason"] = reason
        c2["exception"] = exception
        c2["auto_filter_status"] = "kept"

        kept.append(c2)
        cid += 1

    return kept


def write_candidate_outputs(candidates: List[Dict[str, Any]]) -> None:
    with open(CANDIDATES_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    if candidates:
        df = pd.DataFrame(candidates)
    else:
        df = pd.DataFrame(columns=[
            "constraint_id",
            "constraint_key",
            "constraint_name",
            "relation_a",
            "relation_b",
            "scope",
            "forbidden_temporal_relation",
            "allowed_temporal_relations",
            "hard_or_soft",
            "expected_action",
            "candidate_level",
            "program_candidate_reason",
            "program_recommended_action",
            "typical_positive_example",
            "typical_negative_example",
            "exception",
            "reason",
            "auto_filter_status",
        ])

    df.to_csv(CANDIDATES_CSV_OUT, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_llm", action="store_true", help="只生成 prompt，不调用 LLM")
    parser.add_argument("--parse_existing", action="store_true", help="不调用 LLM，直接解析已有 raw_response 文件")
    parser.add_argument("--max_relations_in_prompt", type=int, default=80, help="prompt 中最多放多少个关系类型")
    parser.add_argument("--max_temporal_pairs_in_prompt", type=int, default=80, help="prompt 中最多放多少个关系对时间摘要")
    parser.add_argument("--max_pairs_for_summary", type=int, default=500, help="关系对时间摘要文件中最多保存多少个关系对")
    parser.add_argument("--max_candidate_pairs", type=int, default=80, help="候选互斥关系对池最多保存多少个关系对")
    parser.add_argument("--max_candidate_pairs_in_prompt", type=int, default=50, help="prompt 中最多放多少个候选关系对")
    parser.add_argument("--max_output_constraints", type=int, default=10, help="要求 LLM 最多输出多少条候选约束")
    parser.add_argument("--min_pair_comparable", type=int, default=5, help="候选池中关系对的最小可比较样本数")
    parser.add_argument("--min_relation_interval_count", type=int, default=10, help="强候选在无足够共现时要求单关系最小区间数")
    args = parser.parse_args()

    ensure_dirs()

    log("========== 第 09 步：生成 LLM 候选互斥约束 ==========")

    edges = read_csv_safely(EDGES_IN)
    log(f"[OK] 读取关系边: {EDGES_IN}, rows={len(edges)}")

    inventory = build_relation_inventory(edges)
    inventory.to_csv(INVENTORY_OUT, index=False, encoding="utf-8-sig")
    log(f"[OK] 已生成关系类型清单: {INVENTORY_OUT}, rows={len(inventory)}")

    examples = build_relation_examples(edges, inventory)
    examples.to_csv(EXAMPLES_OUT, index=False, encoding="utf-8-sig")
    log(f"[OK] 已生成关系样例: {EXAMPLES_OUT}, rows={len(examples)}")

    intervals = read_intervals_for_temporal_summary(INTERVALS_IN)

    temporal_summary = build_relation_pair_temporal_summary(
        intervals=intervals,
        max_pairs_for_summary=args.max_pairs_for_summary,
        max_examples_per_pair=3,
    )

    temporal_summary.to_csv(TEMPORAL_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    log(f"[OK] 已生成关系对时间共现摘要: {TEMPORAL_SUMMARY_OUT}, rows={len(temporal_summary)}")

    candidate_pool = build_candidate_mutex_pair_pool(
        inventory=inventory,
        temporal_summary=temporal_summary,
        min_pair_comparable=args.min_pair_comparable,
        min_relation_interval_count=args.min_relation_interval_count,
        max_candidate_pairs=args.max_candidate_pairs,
    )

    candidate_pool.to_csv(CANDIDATE_POOL_OUT, index=False, encoding="utf-8-sig")
    log(f"[OK] 已生成候选互斥关系对池: {CANDIDATE_POOL_OUT}, rows={len(candidate_pool)}")

    if candidate_pool.empty:
        log("[WARN] 候选互斥关系对池为空。请降低 --min_pair_comparable 或检查关系分组规则。")

    prompt = build_prompt(
        inventory=inventory,
        examples=examples,
        temporal_summary=temporal_summary,
        candidate_pool=candidate_pool,
        max_relations_in_prompt=args.max_relations_in_prompt,
        max_temporal_pairs_in_prompt=args.max_temporal_pairs_in_prompt,
        max_candidate_pairs_in_prompt=args.max_candidate_pairs_in_prompt,
        max_output_constraints=args.max_output_constraints,
    )

    with open(PROMPT_OUT, "w", encoding="utf-8") as f:
        f.write(prompt)

    log(f"[OK] 已生成互斥约束 prompt: {PROMPT_OUT}")

    if args.parse_existing:
        if os.path.exists(RAW_RESPONSE_OUT):
            with open(RAW_RESPONSE_OUT, "r", encoding="utf-8") as f:
                raw_response = f.read()
            log(f"[INFO] --parse_existing 模式，读取已有 LLM 输出: {RAW_RESPONSE_OUT}")
        else:
            raw_response = ""
            log(f"[WARN] --parse_existing 模式但未找到: {RAW_RESPONSE_OUT}")
    elif args.no_llm:
        raw_response = ""
        log("[INFO] --no_llm 模式，跳过 LLM 调用。")
    else:
        raw_response = call_llm(prompt)

    if not args.parse_existing:
        with open(RAW_RESPONSE_OUT, "w", encoding="utf-8") as f:
            f.write(raw_response or "")

    log(f"[OK] 已保存/读取 LLM 原始输出: {RAW_RESPONSE_OUT}, length={len(raw_response or '')}")

    raw_candidates = extract_json_array(raw_response)
    log(f"[STAT] LLM raw candidates: {len(raw_candidates)}")

    if len(raw_candidates) == 0 and raw_response:
        with open(RAW_RESPONSE_DEBUG_OUT, "w", encoding="utf-8") as f:
            f.write("===== HEAD 3000 CHARS =====\n")
            f.write(raw_response[:3000])
            f.write("\n\n===== TAIL 3000 CHARS =====\n")
            f.write(raw_response[-3000:])
        log(f"[WARN] LLM 有输出但未解析出 JSON，已保存首尾片段: {RAW_RESPONSE_DEBUG_OUT}")

    valid_relations = set(inventory["relation_type"].astype(str).tolist())

    kept = filter_candidates(
        raw_candidates=raw_candidates,
        valid_relations=valid_relations,
        candidate_pool=candidate_pool,
    )

    log(f"[STAT] kept candidates: {len(kept)}")
    log(f"[STAT] filtered candidates: {len(raw_candidates) - len(kept)}")

    write_candidate_outputs(kept)

    log(f"[OK] 已生成: {CANDIDATES_JSON_OUT}")
    log(f"[OK] 已生成: {CANDIDATES_CSV_OUT}")

    if len(kept) == 0:
        log("[WARN] 没有保留下来的候选互斥约束。")
        log("[WARN] 如果使用 --no_llm，这是正常的。")
        log("[WARN] 如果已调用 LLM，请检查 raw_response 是否为合法 JSON 数组，或检查候选池是否过窄。")
        log("[WARN] 可尝试降低 --min_pair_comparable，或增大 --max_candidate_pairs_in_prompt。")

    log("========== 第 09 步完成 ==========")


if __name__ == "__main__":
    main()
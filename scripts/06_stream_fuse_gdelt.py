#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_stream_fuse_gdelt.py

功能：
    按步骤.md中的“流式融合 GDELT 数据”要求，
    将 GDELT 事件按日期模拟流式接入，融合到 ICEWS 种子图谱中。

输入：
    ../data/processed/organizations.csv
    ../data/processed/relation_edges_seed.csv
    ../data/raw/gdelt/gdelt_raw_events.csv

输出：
    ../data/processed/relation_edges_before_check.csv
    ../data/processed/stream_update_log.csv

运行：
    python ./06_stream_fuse_gdelt.py
"""

import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# 0. 固定路径配置
# ============================================================

ORGANIZATIONS_PATH = Path("../data/processed/organizations.csv")
SEED_GRAPH_PATH = Path("../data/processed/relation_edges_seed.csv")
GDELT_PATH = Path("../data/raw/gdelt/gdelt_raw_events.csv")

OUT_DIR = Path("../data/processed")
RELATION_EDGES_BEFORE_CHECK_PATH = OUT_DIR / "relation_edges_before_check.csv"
STREAM_UPDATE_LOG_PATH = OUT_DIR / "stream_update_log.csv"

CHUNKSIZE = 200000


# ============================================================
# 1. 基础工具函数
# ============================================================

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_name(name):
    """
    与 03 保持一致的轻量名称规范化。
    不做复杂实体对齐，只用于查 organizations.csv。
    """
    name = safe_str(name)
    name = name.upper()
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        text = str(x).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        text = str(x).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def sql_date_to_month(sql_date):
    """
    GDELT SQLDATE:
        20230101 -> 2023-01
    """
    text = safe_str(sql_date)

    if len(text) < 6:
        return ""

    year = text[:4]
    month = text[4:6]

    return f"{year}-{month}"


def sql_date_to_day(sql_date):
    """
    GDELT SQLDATE:
        20230101 -> 2023-01-01
    """
    text = safe_str(sql_date)

    if len(text) < 8:
        return ""

    year = text[:4]
    month = text[4:6]
    day = text[6:8]

    return f"{year}-{month}-{day}"


def month_start(month):
    if not month:
        return ""
    return f"{month}-01"


def month_end(month):
    """
    简单生成月末日期。
    当前数据是 2023-01 到 2023-04，仍写成通用版。
    """
    if not month:
        return ""

    year, mon = month.split("-")
    year = int(year)
    mon = int(mon)

    if mon in {1, 3, 5, 7, 8, 10, 12}:
        day = 31
    elif mon in {4, 6, 9, 11}:
        day = 30
    else:
        if (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
            day = 29
        else:
            day = 28

    return f"{year:04d}-{mon:02d}-{day:02d}"


# ============================================================
# 2. GDELT 事件类型映射
# ============================================================

def map_gdelt_relation_type(row):
    """
    将 GDELT 事件映射到粗粒度组织关系类型。

    优先使用 QuadClass：
        1 = verbal_cooperation
        2 = material_cooperation
        3 = verbal_conflict
        4 = material_conflict

    若 QuadClass 缺失，则使用 EventRootCode 兜底。
    """
    quad = safe_int(row.get("QuadClass", ""), default=0)

    if quad == 1:
        return "verbal_cooperation"
    if quad == 2:
        return "material_cooperation"
    if quad == 3:
        return "verbal_conflict"
    if quad == 4:
        return "material_conflict"

    root = safe_str(row.get("EventRootCode", ""))

    if root in {"01", "02", "03", "04", "05"}:
        return "verbal_cooperation"
    if root in {"06", "07", "08", "09"}:
        return "material_cooperation"
    if root in {"10", "11", "12", "13"}:
        return "verbal_conflict"
    if root in {"14", "15", "16", "17", "18", "19", "20"}:
        return "material_conflict"

    return "mixed_relation"


# ============================================================
# 3. 读取组织表并构建名称映射
# ============================================================

def load_organization_map():
    """
    从 organizations.csv 构建名称到组织节点的映射。

    兼容 03 输出格式：
        org_id
        canonical_name
        raw_name
        normalized_name
        clean_name
    """
    if not ORGANIZATIONS_PATH.exists():
        raise FileNotFoundError(f"organizations.csv 不存在: {ORGANIZATIONS_PATH}")

    org_df = pd.read_csv(ORGANIZATIONS_PATH, dtype=str).fillna("")

    required_cols = {"org_id", "canonical_name"}
    missing = required_cols - set(org_df.columns)

    if missing:
        raise ValueError(
            f"organizations.csv 缺少必要字段: {missing}\n"
            f"当前字段为: {list(org_df.columns)}"
        )

    name_to_org = {}

    for _, row in org_df.iterrows():
        org_id = safe_str(row["org_id"])
        canonical_name = safe_str(row["canonical_name"])

        possible_names = set()

        for col in ["canonical_name", "raw_name", "normalized_name", "clean_name"]:
            if col in org_df.columns:
                value = safe_str(row[col])
                if value:
                    possible_names.add(value)

        for name in possible_names:
            norm = normalize_name(name)
            if norm:
                name_to_org[norm] = {
                    "org_id": org_id,
                    "canonical_name": canonical_name,
                }

    print(f"[OK] loaded organizations: {len(org_df)}")
    print(f"[OK] name_to_org size: {len(name_to_org)}")

    return name_to_org


def map_actor_to_org(actor_name, name_to_org):
    norm = normalize_name(actor_name)
    if not norm:
        return None
    return name_to_org.get(norm)


# ============================================================
# 4. 关系边结构
# ============================================================

def new_edge(
    subject_org_id,
    subject_name,
    object_org_id,
    object_name,
    relation_type,
    month,
):
    return {
        "subject_org_id": subject_org_id,
        "subject_name": subject_name,
        "object_org_id": object_org_id,
        "object_name": object_name,
        "relation_type": relation_type,
        "month": month,
        "start_date": month_start(month),
        "end_date": month_end(month),

        "event_count": 0,
        "icews_event_count": 0,
        "gdelt_event_count": 0,

        "sources": set(),

        "goldstein_sum": 0.0,
        "goldstein_count": 0,

        "tone_sum": 0.0,
        "tone_count": 0,

        "num_mentions": 0,
        "num_sources": 0,
        "num_articles": 0,

        "evidence_urls": set(),

        "update_status": "new",
    }


def edge_key(subject_org_id, object_org_id, relation_type, month):
    return (
        subject_org_id,
        object_org_id,
        relation_type,
        month,
    )


def add_gdelt_evidence(edge, row):
    edge["event_count"] += 1
    edge["gdelt_event_count"] += 1
    edge["sources"].add("GDELT")

    goldstein = safe_float(row.get("GoldsteinScale", ""), default=0.0)
    edge["goldstein_sum"] += goldstein
    edge["goldstein_count"] += 1

    tone = safe_float(row.get("AvgTone", ""), default=0.0)
    edge["tone_sum"] += tone
    edge["tone_count"] += 1

    edge["num_mentions"] += safe_int(row.get("NumMentions", ""), default=0)
    edge["num_sources"] += safe_int(row.get("NumSources", ""), default=0)
    edge["num_articles"] += safe_int(row.get("NumArticles", ""), default=0)

    url = safe_str(row.get("SOURCEURL", ""))
    if url:
        if len(edge["evidence_urls"]) < 5:
            edge["evidence_urls"].add(url)


# ============================================================
# 5. 加载 ICEWS 种子图谱
# ============================================================

def find_existing_col(df, candidates):
    """
    在不同版本的 relation_edges_seed.csv 中兼容字段名。
    """
    normalized_map = {
        re.sub(r"[^a-z0-9]", "", col.lower()): col
        for col in df.columns
    }

    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in normalized_map:
            return normalized_map[key]

    return None


def load_seed_edges():
    """
    加载 05 输出的 relation_edges_seed.csv。

    期望字段可以是以下几类之一：
        subject_org_id / object_org_id
        source_org_id / target_org_id
        subject_name / object_name
        source_name / target_name
        relation_type
        month

    如果文件不存在，则从空图开始融合 GDELT。
    """
    edges = {}

    if not SEED_GRAPH_PATH.exists():
        print(f"[WARN] seed graph 不存在，将从空图开始: {SEED_GRAPH_PATH}")
        return edges

    seed_df = pd.read_csv(SEED_GRAPH_PATH, dtype=str).fillna("")

    print(f"[OK] loaded seed graph: {len(seed_df)} rows")

    subject_id_col = find_existing_col(seed_df, [
        "subject_org_id", "source_org_id", "head_org_id", "src_org_id"
    ])
    object_id_col = find_existing_col(seed_df, [
        "object_org_id", "target_org_id", "tail_org_id", "dst_org_id"
    ])

    subject_name_col = find_existing_col(seed_df, [
        "subject_name", "source_name", "head_name", "src_name"
    ])
    object_name_col = find_existing_col(seed_df, [
        "object_name", "target_name", "tail_name", "dst_name"
    ])

    relation_col = find_existing_col(seed_df, [
        "relation_type", "relation", "edge_type"
    ])

    month_col = find_existing_col(seed_df, [
        "month", "event_month"
    ])

    event_count_col = find_existing_col(seed_df, [
        "event_count", "icews_event_count", "count"
    ])

    if relation_col is None:
        raise ValueError(
            f"relation_edges_seed.csv 缺少 relation_type 字段。\n"
            f"当前字段为: {list(seed_df.columns)}"
        )

    if month_col is None:
        raise ValueError(
            f"relation_edges_seed.csv 缺少 month 字段。\n"
            f"当前字段为: {list(seed_df.columns)}"
        )

    for _, row in seed_df.iterrows():
        subject_org_id = safe_str(row[subject_id_col]) if subject_id_col else ""
        object_org_id = safe_str(row[object_id_col]) if object_id_col else ""

        subject_name = safe_str(row[subject_name_col]) if subject_name_col else subject_org_id
        object_name = safe_str(row[object_name_col]) if object_name_col else object_org_id

        relation_type = safe_str(row[relation_col])
        month = safe_str(row[month_col])

        if not subject_org_id:
            subject_org_id = normalize_name(subject_name)

        if not object_org_id:
            object_org_id = normalize_name(object_name)

        if not subject_org_id or not object_org_id or not relation_type or not month:
            continue

        key = edge_key(
            subject_org_id,
            object_org_id,
            relation_type,
            month,
        )

        edge = new_edge(
            subject_org_id=subject_org_id,
            subject_name=subject_name,
            object_org_id=object_org_id,
            object_name=object_name,
            relation_type=relation_type,
            month=month,
        )

        count = safe_int(row[event_count_col], default=1) if event_count_col else 1

        edge["event_count"] = count
        edge["icews_event_count"] = count
        edge["gdelt_event_count"] = 0
        edge["sources"].add("ICEWS")
        edge["update_status"] = "seed"

        edges[key] = edge

    print(f"[OK] initialized seed edges: {len(edges)}")

    return edges


# ============================================================
# 6. 流式融合 GDELT
# ============================================================

def process_gdelt_stream(edges, name_to_org):
    """
    按 chunk 读取 GDELT，并按日期模拟流式接入。

    对每条 GDELT 事件：
        1. 对齐 Actor1Name / Actor2Name 到组织节点
        2. 映射事件关系类型
        3. 按 subject, object, month, relation_type 合并到已有边
        4. 没有则新建低置信度边
    """
    if not GDELT_PATH.exists():
        raise FileNotFoundError(f"GDELT 文件不存在: {GDELT_PATH}")

    usecols = [
        "SQLDATE",
        "Actor1Name",
        "Actor2Name",
        "EventCode",
        "EventRootCode",
        "QuadClass",
        "GoldsteinScale",
        "NumMentions",
        "NumSources",
        "NumArticles",
        "AvgTone",
        "ActionGeo_FullName",
        "ActionGeo_CountryCode",
        "SOURCEURL",
    ]

    logs_by_day = defaultdict(lambda: {
        "processed_events": 0,
        "mapped_events": 0,
        "skipped_unmapped": 0,
        "new_edges": 0,
        "updated_edges": 0,
    })

    chunk_id = 0

    for chunk in pd.read_csv(
        GDELT_PATH,
        usecols=usecols,
        dtype=str,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        chunk_id += 1
        chunk = chunk.fillna("")

        print(f"[RUN] GDELT chunk {chunk_id}, rows={len(chunk)}")

        # 按 SQLDATE 排序，模拟按日期流式进入
        chunk = chunk.sort_values(by="SQLDATE")

        for _, row in chunk.iterrows():
            day = sql_date_to_day(row["SQLDATE"])
            month = sql_date_to_month(row["SQLDATE"])

            if not day or not month:
                continue

            log = logs_by_day[day]
            log["processed_events"] += 1

            actor1 = safe_str(row["Actor1Name"])
            actor2 = safe_str(row["Actor2Name"])

            if not actor1 or not actor2:
                log["skipped_unmapped"] += 1
                continue

            subject_org = map_actor_to_org(actor1, name_to_org)
            object_org = map_actor_to_org(actor2, name_to_org)

            if subject_org is None or object_org is None:
                log["skipped_unmapped"] += 1
                continue

            subject_org_id = subject_org["org_id"]
            object_org_id = object_org["org_id"]

            if subject_org_id == object_org_id:
                log["skipped_unmapped"] += 1
                continue

            relation_type = map_gdelt_relation_type(row)

            key = edge_key(
                subject_org_id,
                object_org_id,
                relation_type,
                month,
            )

            if key not in edges:
                edges[key] = new_edge(
                    subject_org_id=subject_org_id,
                    subject_name=subject_org["canonical_name"],
                    object_org_id=object_org_id,
                    object_name=object_org["canonical_name"],
                    relation_type=relation_type,
                    month=month,
                )
                edges[key]["update_status"] = "new_from_gdelt"
                log["new_edges"] += 1
            else:
                if "GDELT" not in edges[key]["sources"]:
                    edges[key]["update_status"] = "merged_cross_source"
                else:
                    if edges[key]["update_status"] == "seed":
                        edges[key]["update_status"] = "merged_cross_source"

                log["updated_edges"] += 1

            add_gdelt_evidence(edges[key], row)
            log["mapped_events"] += 1

    return logs_by_day


# ============================================================
# 7. 输出融合后的关系边
# ============================================================

def edges_to_dataframe(edges):
    rows = []

    for i, (_, edge) in enumerate(edges.items(), start=1):
        if edge["goldstein_count"] > 0:
            avg_goldstein = edge["goldstein_sum"] / edge["goldstein_count"]
        else:
            avg_goldstein = 0.0

        if edge["tone_count"] > 0:
            avg_tone = edge["tone_sum"] / edge["tone_count"]
        else:
            avg_tone = 0.0

        sources = sorted(edge["sources"])

        rows.append({
            "edge_id": f"EDGE_{i:08d}",

            "subject_org_id": edge["subject_org_id"],
            "subject_name": edge["subject_name"],
            "object_org_id": edge["object_org_id"],
            "object_name": edge["object_name"],

            "relation_type": edge["relation_type"],
            "month": edge["month"],
            "start_date": edge["start_date"],
            "end_date": edge["end_date"],

            "event_count": edge["event_count"],
            "icews_event_count": edge["icews_event_count"],
            "gdelt_event_count": edge["gdelt_event_count"],

            "source_count": len(sources),
            "sources": ";".join(sources),

            "avg_goldstein": round(avg_goldstein, 4),
            "avg_tone": round(avg_tone, 4),

            "num_mentions": edge["num_mentions"],
            "num_sources": edge["num_sources"],
            "num_articles": edge["num_articles"],

            "evidence_urls": " | ".join(sorted(edge["evidence_urls"])),

            "update_status": edge["update_status"],
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["month", "subject_name", "object_name", "relation_type"],
            ascending=[True, True, True, True],
        )

    return df


def save_relation_edges(edges):
    df = edges_to_dataframe(edges)

    df.to_csv(
        RELATION_EDGES_BEFORE_CHECK_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] relation_edges_before_check.csv: {RELATION_EDGES_BEFORE_CHECK_PATH}")
    print(f"[STAT] relation edges: {len(df)}")

    return df


def save_stream_update_log(logs_by_day):
    rows = []

    for day, log in sorted(logs_by_day.items()):
        month = day[:7]

        rows.append({
            "date": day,
            "month": month,
            "processed_events": log["processed_events"],
            "mapped_events": log["mapped_events"],
            "skipped_unmapped": log["skipped_unmapped"],
            "new_edges": log["new_edges"],
            "updated_edges": log["updated_edges"],
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        STREAM_UPDATE_LOG_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] stream_update_log.csv: {STREAM_UPDATE_LOG_PATH}")
    print(f"[STAT] stream log days: {len(df)}")

    return df


# ============================================================
# 8. 主流程
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[CONFIG] ORGANIZATIONS_PATH:", ORGANIZATIONS_PATH)
    print("[CONFIG] SEED_GRAPH_PATH:", SEED_GRAPH_PATH)
    print("[CONFIG] GDELT_PATH:", GDELT_PATH)
    print()

    name_to_org = load_organization_map()

    edges = load_seed_edges()

    logs_by_day = process_gdelt_stream(
        edges=edges,
        name_to_org=name_to_org,
    )

    save_relation_edges(edges)
    save_stream_update_log(logs_by_day)

    print("[DONE] 06_stream_fuse_gdelt.py finished.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_stream_fuse_gdelt.py

功能：
    将 GDELT 数据融合到 ICEWS 种子图谱中。

输入：
    ../data/processed/organizations.csv
    ../data/processed/relation_edges_seed.csv
    ../data/raw/gdelt/gdelt_raw_events.csv

输出：
    ../data/processed/relation_edges_before_check.csv
    ../data/processed/stream_update_log.csv

输出 relation_edges_before_check.csv 字段与 05 保持一致：

    edge_id
    subject_org_id
    object_org_id
    event_month
    relation_type
    subject_name
    object_name
    event_count
    source_datasets
    icews_event_count
    gdelt_event_count
    confidence
    status

说明：
    本脚本不输出 event_code / event_codes。
    EventCode / EventRootCode 只用于在 06 中正确映射 relation_type。
    如果后续需要为 PaTeCon 使用 CAMEO 顶层码，应在 08_build_intervals.py 中处理。

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

# GDELT 单源新增边过滤阈值
# 避免把只出现 1 次的低证据 GDELT 边全部写入后续流程
MIN_GDELT_ONLY_EVENT_COUNT = 3


# ============================================================
# 1. 基础工具函数
# ============================================================

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_int(x, default=0):
    try:
        text = safe_str(x)
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        text = safe_str(x)
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def normalize_name(name):
    name = safe_str(name)
    name = name.upper()
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def normalize_event_code(code):
    """
    标准化 GDELT / ICEWS 事件码为纯数字字符串。

    这里只用于 relation_type 映射，不输出到 CSV。

    例：
        "4"    -> "4"
        "40"   -> "40"
        "040"  -> "040"
        "73"   -> "73"
        "073"  -> "073"
        "10"   -> "10"
        "100"  -> "100"
        "120"  -> "120"
        "51.0" -> "51"
    """
    if code is None or pd.isna(code):
        return ""

    text = str(code).strip()

    if not text:
        return ""

    if "." in text:
        text = text.split(".")[0]

    text = re.sub(r"\D", "", text)

    return text


def get_cameo_top2(code):
    """
    从 EventCode / EventRootCode 中提取两位 CAMEO 顶层码。

    适配 GDELT 常见情况：
        EventRootCode:
            1  -> 01
            4  -> 04
            7  -> 07
            10 -> 10
            19 -> 19

        EventCode:
            40  -> 04
            73  -> 07
            100 -> 10
            120 -> 12
            190 -> 19
            020 -> 02
            051 -> 05
    """
    text = normalize_event_code(code)

    if not text:
        return ""

    # EventRootCode 通常是 1~20
    if len(text) <= 2:
        n = safe_int(text, default=-1)

        if 1 <= n <= 20:
            return f"{n:02d}"

        # 例如 40、73 这类虽然是两位，但其实是 EventCode
        # 40 -> 04, 73 -> 07
        first = safe_int(text[0], default=-1)
        if 1 <= first <= 9:
            return f"{first:02d}"

        return ""

    # 三位及以上 EventCode：
    # 020 -> 02
    # 051 -> 05
    # 100 -> 10
    # 120 -> 12
    # 190 -> 19
    top = text[:2]
    n = safe_int(top, default=-1)

    if 1 <= n <= 20:
        return f"{n:02d}"

    return ""


def sql_date_to_month(sql_date):
    """
    GDELT SQLDATE:
        20230101 -> 2023-01
    """
    text = safe_str(sql_date)
    if len(text) < 6:
        return ""
    return f"{text[:4]}-{text[4:6]}"


def sql_date_to_day(sql_date):
    """
    GDELT SQLDATE:
        20230101 -> 2023-01-01
    """
    text = safe_str(sql_date)
    if len(text) < 8:
        return ""
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


# ============================================================
# 2. GDELT 事件类型映射
# ============================================================

def map_gdelt_relation_type(row):
    """
    将 GDELT 事件映射为原来的粗粒度 relation_type。

    优先级：
        1. EventRootCode
        2. EventCode
        3. QuadClass 兜底

    输出仍然保持原有类型：
        verbal_cooperation
        material_cooperation
        verbal_conflict
        material_conflict
        mixed_relation

    注意：
        本函数不输出 CAMEO_01 ~ CAMEO_20。
        PaTeCon 所需 CAMEO 顶层码应在 08 中转换。
    """

    # 1. 优先使用 EventRootCode
    root = get_cameo_top2(row.get("EventRootCode", ""))

    # 2. 如果 EventRootCode 无法识别，再从 EventCode 推导
    if not root:
        root = get_cameo_top2(row.get("EventCode", ""))

    # 3. 根据 CAMEO 顶层码映射为粗粒度关系
    if root in {"01", "02", "03", "04", "05"}:
        return "verbal_cooperation"

    if root in {"06", "07", "08"}:
        return "material_cooperation"

    if root in {"09", "10", "11", "12", "13"}:
        return "verbal_conflict"

    if root in {"14", "15", "16", "17", "18", "19", "20"}:
        return "material_conflict"

    # 4. CAMEO 无法识别时，用 QuadClass 兜底
    quad = safe_int(row.get("QuadClass", ""), default=0)

    if quad == 1:
        return "verbal_cooperation"
    if quad == 2:
        return "material_cooperation"
    if quad == 3:
        return "verbal_conflict"
    if quad == 4:
        return "material_conflict"

    return "mixed_relation"


# ============================================================
# 3. 读取 organizations.csv，构建名称到 org_id 的映射
# ============================================================

def load_organization_map():
    if not ORGANIZATIONS_PATH.exists():
        raise FileNotFoundError(f"organizations.csv 不存在: {ORGANIZATIONS_PATH}")

    org_df = pd.read_csv(ORGANIZATIONS_PATH, dtype=str).fillna("")

    required = {"org_id", "canonical_name"}
    missing = required - set(org_df.columns)

    if missing:
        raise ValueError(
            f"organizations.csv 缺少字段: {missing}\n"
            f"当前字段: {list(org_df.columns)}"
        )

    name_to_org = {}

    for _, row in org_df.iterrows():
        org_id = safe_str(row["org_id"])
        canonical_name = safe_str(row["canonical_name"])

        names = set()

        for col in ["canonical_name", "raw_name", "normalized_name", "clean_name"]:
            if col in org_df.columns:
                value = safe_str(row[col])
                if value:
                    names.add(value)

        for name in names:
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
# 4. 读取 05 生成的 ICEWS 种子边
# ============================================================

def make_edge_key(subject_org_id, object_org_id, event_month, relation_type):
    return (
        safe_str(subject_org_id),
        safe_str(object_org_id),
        safe_str(event_month),
        safe_str(relation_type),
    )


def normalize_source_datasets(value):
    """
    将 source_datasets 统一成集合。
    """
    text = safe_str(value)

    if not text:
        return set()

    parts = re.split(r"[;,|]+", text)
    return {p.strip() for p in parts if p.strip()}


def load_seed_edges():
    """
    加载 relation_edges_seed.csv。

    该函数严格按照 05 当前输出字段读取：

        edge_id
        subject_org_id
        object_org_id
        event_month
        relation_type
        subject_name
        object_name
        event_count
        source_datasets
        icews_event_count
        gdelt_event_count
        confidence
        status

    注意：
        这里不读取 event_code / event_codes。
        06 不改变输出列。
    """
    if not SEED_GRAPH_PATH.exists():
        raise FileNotFoundError(f"relation_edges_seed.csv 不存在: {SEED_GRAPH_PATH}")

    seed_df = pd.read_csv(SEED_GRAPH_PATH, dtype=str).fillna("")

    required_cols = [
        "edge_id",
        "subject_org_id",
        "object_org_id",
        "event_month",
        "relation_type",
        "subject_name",
        "object_name",
        "event_count",
        "source_datasets",
        "icews_event_count",
        "gdelt_event_count",
        "confidence",
        "status",
    ]

    missing = [col for col in required_cols if col not in seed_df.columns]

    if missing:
        raise ValueError(
            f"relation_edges_seed.csv 缺少字段: {missing}\n"
            f"当前字段: {list(seed_df.columns)}"
        )

    edges = {}

    for _, row in seed_df.iterrows():
        subject_org_id = safe_str(row["subject_org_id"])
        object_org_id = safe_str(row["object_org_id"])
        event_month = safe_str(row["event_month"])
        relation_type = safe_str(row["relation_type"])

        if not subject_org_id or not object_org_id or not event_month or not relation_type:
            continue

        key = make_edge_key(
            subject_org_id,
            object_org_id,
            event_month,
            relation_type,
        )

        source_set = normalize_source_datasets(row["source_datasets"])
        if not source_set:
            source_set = {"ICEWS"}

        edges[key] = {
            "subject_org_id": subject_org_id,
            "object_org_id": object_org_id,
            "event_month": event_month,
            "relation_type": relation_type,
            "subject_name": safe_str(row["subject_name"]),
            "object_name": safe_str(row["object_name"]),

            "event_count": safe_int(row["event_count"], default=0),
            "source_datasets": source_set,
            "icews_event_count": safe_int(row["icews_event_count"], default=0),
            "gdelt_event_count": safe_int(row["gdelt_event_count"], default=0),

            "confidence": safe_float(row["confidence"], default=0.6),
            "status": safe_str(row["status"]) or "active",
        }

    print(f"[OK] loaded seed graph: {len(seed_df)} rows")
    print(f"[OK] initialized seed edges: {len(edges)}")

    return edges


# ============================================================
# 5. 融合 GDELT
# ============================================================

def create_gdelt_edge(subject_org, object_org, event_month, relation_type):
    """
    新建 GDELT 单源边。

    confidence 只是初始值，后续 07 会重新计算。
    """
    return {
        "subject_org_id": subject_org["org_id"],
        "object_org_id": object_org["org_id"],
        "event_month": event_month,
        "relation_type": relation_type,
        "subject_name": subject_org["canonical_name"],
        "object_name": object_org["canonical_name"],

        "event_count": 0,
        "source_datasets": {"GDELT"},
        "icews_event_count": 0,
        "gdelt_event_count": 0,

        "confidence": 0.3,
        "status": "active",
    }


def add_gdelt_to_edge(edge):
    """
    将一条 GDELT 事件证据加入关系边。

    注意：
        这里不保存 EventCode。
        EventCode 只在 map_gdelt_relation_type() 中用于生成 relation_type。
    """
    edge["event_count"] += 1
    edge["gdelt_event_count"] += 1
    edge["source_datasets"].add("GDELT")

    # 如果是 ICEWS + GDELT 跨源支持，给一个临时较高初始值
    # 后续 07_score_relations.py 会重新计算最终 confidence
    if "ICEWS" in edge["source_datasets"] and "GDELT" in edge["source_datasets"]:
        edge["confidence"] = max(edge["confidence"], 0.7)


def process_gdelt_stream(edges, name_to_org):
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

        chunk = chunk.sort_values(by="SQLDATE")

        for _, row in chunk.iterrows():
            day = sql_date_to_day(row["SQLDATE"])
            event_month = sql_date_to_month(row["SQLDATE"])

            if not day or not event_month:
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

            key = make_edge_key(
                subject_org_id,
                object_org_id,
                event_month,
                relation_type,
            )

            if key not in edges:
                edges[key] = create_gdelt_edge(
                    subject_org=subject_org,
                    object_org=object_org,
                    event_month=event_month,
                    relation_type=relation_type,
                )
                log["new_edges"] += 1
            else:
                log["updated_edges"] += 1

            add_gdelt_to_edge(edges[key])
            log["mapped_events"] += 1

    return logs_by_day


# ============================================================
# 6. 输出 relation_edges_before_check.csv
# ============================================================

def should_keep_edge(edge):
    """
    输出过滤规则：

    1. ICEWS 种子边一定保留；
    2. ICEWS + GDELT 跨源融合边一定保留；
    3. GDELT 单源新增边，至少需要出现 MIN_GDELT_ONLY_EVENT_COUNT 次。
    """
    sources = edge["source_datasets"]

    if "ICEWS" in sources:
        return True

    if "GDELT" in sources:
        return edge["gdelt_event_count"] >= MIN_GDELT_ONLY_EVENT_COUNT

    return False


def save_relation_edges(edges):
    rows = []
    dropped = 0

    for edge in edges.values():
        if not should_keep_edge(edge):
            dropped += 1
            continue

        rows.append({
            "subject_org_id": edge["subject_org_id"],
            "object_org_id": edge["object_org_id"],
            "event_month": edge["event_month"],
            "relation_type": edge["relation_type"],
            "subject_name": edge["subject_name"],
            "object_name": edge["object_name"],
            "event_count": edge["event_count"],
            "source_datasets": ";".join(sorted(edge["source_datasets"])),
            "icews_event_count": edge["icews_event_count"],
            "gdelt_event_count": edge["gdelt_event_count"],
            "confidence": round(edge["confidence"], 4),
            "status": edge["status"],
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["event_month", "subject_org_id", "object_org_id", "relation_type"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

        df.insert(
            0,
            "edge_id",
            [f"E_{i:09d}" for i in range(1, len(df) + 1)],
        )
    else:
        df = pd.DataFrame(columns=[
            "edge_id",
            "subject_org_id",
            "object_org_id",
            "event_month",
            "relation_type",
            "subject_name",
            "object_name",
            "event_count",
            "source_datasets",
            "icews_event_count",
            "gdelt_event_count",
            "confidence",
            "status",
        ])

    df.to_csv(
        RELATION_EDGES_BEFORE_CHECK_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] relation_edges_before_check.csv: {RELATION_EDGES_BEFORE_CHECK_PATH}")
    print(f"[STAT] relation edges: {len(df)}")
    print(f"[STAT] dropped low-evidence GDELT-only edges: {dropped}")

    return df


def save_stream_update_log(logs_by_day):
    rows = []

    for day, log in sorted(logs_by_day.items()):
        rows.append({
            "date": day,
            "event_month": day[:7],
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
# 7. 主流程
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
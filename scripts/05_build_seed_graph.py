# scripts/05_build_seed_graph.py
import os
import re
import pandas as pd

RAW_ICEWS = "../data/raw/icews/icews_raw_events.csv"

PROCESSED_DIR = "../data/processed"
ORGANIZATIONS_FILE = os.path.join(PROCESSED_DIR, "organizations.csv")
ALIASES_FILE = os.path.join(PROCESSED_DIR, "organization_aliases.csv")

EVENT_FACTS_OUT = os.path.join(PROCESSED_DIR, "event_facts.csv")
SOURCE_EVIDENCE_OUT = os.path.join(PROCESSED_DIR, "source_evidence.csv")
EVENT_FACTS_REL_OUT = os.path.join(PROCESSED_DIR, "event_facts_with_relation_type.csv")
ICEWS_SEED_GRAPH_OUT = os.path.join(PROCESSED_DIR, "icews_seed_graph.csv")
RELATION_EDGES_SEED_OUT = os.path.join(PROCESSED_DIR, "relation_edges_seed.csv")

CHUNKSIZE = 200000


def clean_name(name):
    if pd.isna(name):
        return ""

    name = str(name).upper().strip()
    name = re.sub(r"\(.*?\)", " ", name)
    name = re.sub(r"\[.*?\]", " ", name)
    name = re.sub(r"[^A-Z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"^THE\s+", "", name)
    return name

def join_unique_codes(series):
    values = []
    for x in series:
        x = str(x).strip()
        if x and x.lower() != "nan":
            values.append(x)
    return ";".join(sorted(set(values)))

def pick_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def normalize_cameo_code(code):
    """
    保留 CAMEO 原始语义，修复 pandas 可能造成的前导零丢失。

    例：
        10   -> 010
        20   -> 020
        40   -> 040
        51   -> 051
        100  -> 100
        120  -> 120
        010  -> 010
    """
    if code is None or pd.isna(code):
        return ""

    text = str(code).strip()

    if not text:
        return ""

    if "." in text:
        text = text.split(".")[0]

    text = re.sub(r"\D", "", text)

    if not text:
        return ""

    if len(text) == 1:
        return text.zfill(3)

    if len(text) == 2:
        return text.zfill(3)

    return text


def get_cameo_root_code(code):
    """
    从标准化后的 CAMEO 码中取顶层两位码。

    例：
        010 -> 01
        020 -> 02
        051 -> 05
        100 -> 10
        120 -> 12
        190 -> 19
    """
    code = normalize_cameo_code(code)

    if len(code) < 2:
        return ""

    return code[:2]

def map_relation_type(cameo_code=None, quad_class=None):
    """
    将 CAMEO / GDELT QuadClass 映射到粗粒度关系类型。

    QuadClass:
    1 = verbal cooperation
    2 = material cooperation
    3 = verbal conflict
    4 = material conflict
    """

    if quad_class is not None and not pd.isna(quad_class):
        try:
            q = int(float(quad_class))
            if q == 1:
                return "verbal_cooperation"
            if q == 2:
                return "material_cooperation"
            if q == 3:
                return "verbal_conflict"
            if q == 4:
                return "material_conflict"
        except Exception:
            pass

    if cameo_code is None or pd.isna(cameo_code):
        return "mixed_relation"

    code = str(cameo_code).strip()

    if not code:
        return "mixed_relation"

    # 取 CAMEO 顶层两位码
    try:
        top = int(code[:2])
    except Exception:
        return "mixed_relation"

    # CAMEO 大致映射
    if 1 <= top <= 5:
        return "verbal_cooperation"
    elif 6 <= top <= 8:
        return "material_cooperation"
    elif 9 <= top <= 13:
        return "verbal_conflict"
    elif 14 <= top <= 20:
        return "material_conflict"
    else:
        return "mixed_relation"


def load_org_mapping():
    orgs = pd.read_csv(ORGANIZATIONS_FILE, low_memory=False)

    mapping = {}

    for _, row in orgs.iterrows():
        org_id = row["org_id"]

        for col in ["canonical_name", "raw_name", "clean_name", "normalized_name"]:
            if col in orgs.columns and not pd.isna(row.get(col, "")):
                mapping[clean_name(row[col])] = org_id

    if os.path.exists(ALIASES_FILE):
        aliases = pd.read_csv(ALIASES_FILE, low_memory=False)
        for _, row in aliases.iterrows():
            mapping[clean_name(row["alias"])] = row["org_id"]
            mapping[clean_name(row["alias_clean"])] = row["org_id"]

    return mapping


def main():
    if not os.path.exists(RAW_ICEWS):
        print(f"未找到 ICEWS 文件: {RAW_ICEWS}")
        return

    if not os.path.exists(ORGANIZATIONS_FILE):
        print(f"未找到组织文件: {ORGANIZATIONS_FILE}")
        return

    org_map = load_org_mapping()

    first_event = True
    first_evidence = True
    fact_id_counter = 1

    print("开始构建 ICEWS 统一事件表...")

    for chunk in pd.read_csv(
        RAW_ICEWS,
        chunksize=CHUNKSIZE,
        low_memory=False,
        dtype=str
    ):
        source_col = pick_col(chunk, ["source_name", "Source Name", "SourceName", "Source Actor", "Source"])
        target_col = pick_col(chunk, ["target_name", "Target Name", "TargetName", "Target Actor", "Target"])
        date_col = pick_col(chunk, ["event_date", "Event Date", "EventDate", "Date"])
        cameo_col = pick_col(chunk, ["cameo_code", "CAMEO Code", "CAMEOCode", "Event Code", "EventCode"])
        city_col = pick_col(chunk, ["city", "City"])
        province_col = pick_col(chunk, ["province", "Province", "State"])
        country_col = pick_col(chunk, ["country", "Country"])
        publisher_col = pick_col(chunk, ["publisher", "Publisher"])
        story_col = pick_col(chunk, ["story_id", "Story ID", "StoryID"])
        sentence_col = pick_col(chunk, ["sentence_number", "Sentence Number", "SentenceNumber"])
        intensity_col = pick_col(chunk, ["intensity", "Intensity"])

        if not source_col or not target_col or not date_col:
            print("ICEWS 字段名不匹配，请检查 source/target/date 字段。")
            print("当前字段:", list(chunk.columns))
            return

        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        chunk = chunk.dropna(subset=[date_col])

        events = []
        evidences = []

        for _, row in chunk.iterrows():
            source_name = row.get(source_col, "")
            target_name = row.get(target_col, "")

            subject_org_id = org_map.get(clean_name(source_name))
            object_org_id = org_map.get(clean_name(target_name))

            # 只保留主体和客体都能对齐到组织表的事件
            if not subject_org_id or not object_org_id:
                continue

            event_date = row[date_col]
            raw_cameo_code = row.get(cameo_col, "") if cameo_col else ""
            cameo_code = normalize_cameo_code(raw_cameo_code)
            event_root_code = get_cameo_root_code(cameo_code)

            relation_type = map_relation_type(cameo_code=cameo_code)

            location_parts = []
            for c in [city_col, province_col, country_col]:
                if c and not pd.isna(row.get(c, "")):
                    location_parts.append(str(row.get(c, "")))
            location = ", ".join(location_parts)

            fact_id = "F_%09d" % fact_id_counter
            fact_id_counter += 1

            events.append({
                "fact_id": fact_id,
                "source_dataset": "ICEWS",
                "subject_org_id": subject_org_id,
                "subject_name": source_name,
                "object_org_id": object_org_id,
                "object_name": target_name,
                "event_code": cameo_code,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "event_month": event_date.strftime("%Y-%m"),
                "location": location,
                "relation_type": relation_type,
                "intensity": row.get(intensity_col, "") if intensity_col else "",
                "num_sources": "",
                "num_articles": "",
                "num_mentions": ""
            })

            evidences.append({
                "fact_id": fact_id,
                "source_dataset": "ICEWS",
                "publisher": row.get(publisher_col, "") if publisher_col else "",
                "story_id": row.get(story_col, "") if story_col else "",
                "sentence_number": row.get(sentence_col, "") if sentence_col else "",
                "source_url": ""
            })

        if events:
            pd.DataFrame(events).to_csv(
                EVENT_FACTS_OUT,
                mode="w" if first_event else "a",
                index=False,
                header=first_event,
                encoding="utf-8-sig"
            )

            pd.DataFrame(events).to_csv(
                EVENT_FACTS_REL_OUT,
                mode="w" if first_event else "a",
                index=False,
                header=first_event,
                encoding="utf-8-sig"
            )

            first_event = False

        if evidences:
            pd.DataFrame(evidences).to_csv(
                SOURCE_EVIDENCE_OUT,
                mode="w" if first_evidence else "a",
                index=False,
                header=first_evidence,
                encoding="utf-8-sig"
            )

            first_evidence = False

    if not os.path.exists(EVENT_FACTS_REL_OUT):
        print("没有生成可用事件，请检查组织对齐结果。")
        return

    print("开始构建 ICEWS 种子图谱...")

    events = pd.read_csv(EVENT_FACTS_REL_OUT, low_memory=False, dtype=str).fillna("")

    # 每条事件转为一条图谱边
    seed_graph = events[
        [
            "fact_id",
            "subject_org_id",
            "subject_name",
            "object_org_id",
            "object_name",
            "relation_type",
            "event_date",
            "event_month",
            "event_code",
            "location",
            "intensity"
        ]
    ].copy()

    seed_graph.to_csv(ICEWS_SEED_GRAPH_OUT, index=False, encoding="utf-8-sig")

    # 按组织对、月份、关系类型聚合
    relation_edges = events.groupby(
        [
            "subject_org_id",
            "object_org_id",
            "event_month",
            "relation_type"
        ],
        as_index=False
    ).agg(
        subject_name=("subject_name", "first"),
        object_name=("object_name", "first"),
        event_count=("fact_id", "count"),
        event_codes=("event_code", join_unique_codes)
    )

    relation_edges.insert(
        0,
        "edge_id",
        ["E_%09d" % (i + 1) for i in range(len(relation_edges))]
    )

    relation_edges["source_datasets"] = "ICEWS"
    relation_edges["icews_event_count"] = relation_edges["event_count"]
    relation_edges["gdelt_event_count"] = 0
    relation_edges["confidence"] = 0.6
    relation_edges["status"] = "active"

    relation_edges.to_csv(RELATION_EDGES_SEED_OUT, index=False, encoding="utf-8-sig")

    print("ICEWS 种子图谱构建完成")
    print("输出:", EVENT_FACTS_OUT)
    print("输出:", SOURCE_EVIDENCE_OUT)
    print("输出:", EVENT_FACTS_REL_OUT)
    print("输出:", ICEWS_SEED_GRAPH_OUT)
    print("输出:", RELATION_EDGES_SEED_OUT)
    print("种子边数量:", len(relation_edges))


if __name__ == "__main__":
    main()
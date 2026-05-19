#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_extract_organizations.py

功能：
    从 ICEWS 和 GDELT 行动者字段中抽取组织机构候选。

输入：
    ../data/raw/icews/icews_raw_events.csv
    ../data/raw/gdelt/gdelt_raw_events.csv

输出：
    ../data/processed/actor_candidates.csv
    ../data/processed/organizations.csv

运行：
    python ./03_extract_organizations.py
"""

import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# 0. 固定路径配置
# ============================================================

ICEWS_PATH = Path("../data/raw/icews/icews_raw_events.csv")
GDELT_PATH = Path("../data/raw/gdelt/gdelt_raw_events.csv")

OUT_DIR = Path("../data/processed")
ACTOR_CANDIDATES_PATH = OUT_DIR / "actor_candidates.csv"
ORGANIZATIONS_PATH = OUT_DIR / "organizations.csv"

CHUNKSIZE = 100000

# 最终至少希望保留 1000 个组织节点
TARGET_ORG_NUM = 1000

# 候选最小出现频次
MIN_FREQUENCY = 2


# ============================================================
# 1. 简单规则
# ============================================================

ORG_TYPE_CODES = {
    "GOV", "MIL", "COP", "SPY", "JUD", "LEG",
    "PTY", "NGO", "IGO", "BUS", "MNC",
    "MED", "EDU", "REL", "LAB", "CRM",
}

ORG_SECTOR_KEYWORDS = {
    "GOVERNMENT",
    "MINISTRY",
    "CABINET",
    "EXECUTIVE",
    "LEGISLATIVE",
    "PARLIAMENTARY",
    "PARTIES",
    "PARTY",
    "INTERNATIONAL GOVERNMENT ORGANIZATION",
    "GLOBAL DIPLOMATIC IGOS",
    "JUDICIAL",
    "JUSTICE",
    "BUSINESS",
    "MEDIA",
    "EDUCATION",
    "UNIVERSITY",
    "MILITARY",
    "DEFENSE",
    "SECURITY",
    "POLICE",
}

ORG_NAME_KEYWORDS = {
    "UNITED NATIONS",
    "NATO",
    "EU",
    "WORLD BANK",
    "IMF",
    "WHO",
    "MINISTRY",
    "DEPARTMENT",
    "GOVERNMENT",
    "PARLIAMENT",
    "COURT",
    "COMMISSION",
    "COMMITTEE",
    "COUNCIL",
    "PARTY",
    "ARMY",
    "POLICE",
    "FORCE",
    "FORCES",
    "BANK",
    "COMPANY",
    "CORPORATION",
    "GROUP",
    "UNIVERSITY",
    "AGENCY",
    "ORGANIZATION",
    "ORGANISATION",
    "ASSOCIATION",
    "INSTITUTE",
}

BAD_NAMES = {
    "",
    "NAN",
    "NONE",
    "NULL",
    "UNKNOWN",
    "PEOPLE",
    "CITIZENS",
    "CIVILIANS",
    "RESIDENTS",
    "MEN",
    "WOMEN",
    "CHILDREN",
    "STUDENTS",
    "WORKERS",
    "FARMERS",
    "PROTESTERS",
    "DEMONSTRATORS",
    "REFUGEES",
    "MIGRANTS",
}


# ============================================================
# 2. 基础清洗函数
# ============================================================

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_name(name):
    """
    生成 normalized_name。
    只做轻量规范化，不做实体对齐。
    """
    name = safe_str(name)
    name = name.upper()
    name = re.sub(r"\s+", " ", name)
    name = name.strip()
    return name


def clean_name(name):
    """
    生成 clean_name。
    这里仅做少量常见名称简化。
    后续更复杂的别名合并应放到 04。
    """
    alias = {
        "UNITED STATES": "US",
        "UNITED STATES OF AMERICA": "US",
        "UNITED KINGDOM": "UNITED KINGDOM",
        "UNITED NATIONS": "UNITED NATIONS",
    }
    return alias.get(name, name)


def is_bad_name(name):
    if not name:
        return True

    if name in BAD_NAMES:
        return True

    if len(name) <= 1:
        return True

    if re.fullmatch(r"\d+", name):
        return True

    if name.startswith("CITIZEN ("):
        return True

    if len(name.split()) > 12:
        return True

    return False


def split_sector(sector_text):
    """
    ICEWS 的 Source Sectors / Target Sectors 通常是逗号分隔。
    """
    sector_text = safe_str(sector_text)
    if not sector_text:
        return []

    parts = [p.strip().upper() for p in sector_text.split(",")]
    return [p for p in parts if p]


def has_org_type(type_info_set):
    """
    判断 GDELT type_info 中是否包含组织类代码。
    """
    for item in type_info_set:
        for code in ORG_TYPE_CODES:
            if re.search(rf"\b{code}\b", item):
                return True
    return False


def has_org_sector(sector_info_set):
    """
    判断 ICEWS sector_info 中是否包含组织类关键词。
    """
    for sector in sector_info_set:
        sector_upper = sector.upper()
        for kw in ORG_SECTOR_KEYWORDS:
            if kw in sector_upper:
                return True
    return False


def has_org_name(name):
    """
    判断名称本身是否像组织。
    """
    for kw in ORG_NAME_KEYWORDS:
        if kw in name:
            return True
    return False


def should_keep_as_candidate(record):
    """
    判断是否进入 actor_candidates.csv。
    """
    name = record["normalized_name"]

    if is_bad_name(name):
        return False

    if record["frequency"] < MIN_FREQUENCY:
        return False

    return True


def should_keep_as_organization(record):
    """
    判断是否优先进入 organizations.csv。

    规则来自 md：
        1. 高频
        2. sector/type 判断
        3. 去掉普通人群、泛称群体
        4. 保留跨源出现
    """
    name = record["normalized_name"]

    if is_bad_name(name):
        return False

    if record["frequency"] < MIN_FREQUENCY:
        return False

    if record["source_count"] >= 2:
        return True

    if has_org_type(record["type_info_set"]):
        return True

    if has_org_sector(record["sector_info_set"]):
        return True

    if has_org_name(name):
        return True

    return False


# ============================================================
# 3. 统计结构
# ============================================================

def new_record(raw_name, normalized_name):
    return {
        "raw_name": raw_name,
        "normalized_name": normalized_name,
        "frequency": 0,
        "sources": set(),
        "type_info_set": set(),
        "sector_info_set": set(),
    }


def add_actor(records, name, source, type_info="", sector_info=""):
    raw_name = safe_str(name)
    normalized = normalize_name(raw_name)

    if not normalized:
        return

    if normalized not in records:
        records[normalized] = new_record(raw_name, normalized)

    rec = records[normalized]
    rec["frequency"] += 1
    rec["sources"].add(source)

    if type_info:
        rec["type_info_set"].add(type_info)

    if sector_info:
        rec["sector_info_set"].add(sector_info)


# ============================================================
# 4. 处理 ICEWS
# ============================================================

def process_icews(records):
    print(f"[RUN] read ICEWS: {ICEWS_PATH}")

    usecols = [
        "Source Name",
        "Source Sectors",
        "Target Name",
        "Target Sectors",
    ]

    for chunk in pd.read_csv(
        ICEWS_PATH,
        usecols=usecols,
        dtype=str,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        for _, row in chunk.iterrows():
            source_name = row["Source Name"]
            source_sectors = row["Source Sectors"]

            target_name = row["Target Name"]
            target_sectors = row["Target Sectors"]

            for sector in split_sector(source_sectors):
                add_actor(
                    records,
                    name=source_name,
                    source="ICEWS",
                    sector_info=sector,
                )

            if not split_sector(source_sectors):
                add_actor(
                    records,
                    name=source_name,
                    source="ICEWS",
                    sector_info="",
                )

            for sector in split_sector(target_sectors):
                add_actor(
                    records,
                    name=target_name,
                    source="ICEWS",
                    sector_info=sector,
                )

            if not split_sector(target_sectors):
                add_actor(
                    records,
                    name=target_name,
                    source="ICEWS",
                    sector_info="",
                )


# ============================================================
# 5. 处理 GDELT
# ============================================================

def build_gdelt_type_info(known_group, type1, type2, type3):
    """
    生成类似旧结果中的 type_info：

        GOV MIL nan nan
        BUS nan nan nan
        nan nan nan nan

    这里统一四列：
        KnownGroupCode Type1Code Type2Code Type3Code

    如果 KnownGroupCode 为空，则用 Type1Code 开头，尽量贴近旧格式。
    """
    kg = safe_str(known_group)
    t1 = safe_str(type1)
    t2 = safe_str(type2)
    t3 = safe_str(type3)

    values = []

    if kg:
        values.append(kg)
    elif t1:
        values.append(t1)
    else:
        values.append("nan")

    if kg:
        values.append(t1 if t1 else "nan")
    else:
        values.append(t2 if t2 else "nan")

    if kg:
        values.append(t2 if t2 else "nan")
    else:
        values.append(t3 if t3 else "nan")

    values.append("nan")

    return " ".join(v.upper() for v in values)


def process_gdelt(records):
    print(f"[RUN] read GDELT: {GDELT_PATH}")

    usecols = [
        "Actor1Name",
        "Actor1KnownGroupCode",
        "Actor1Type1Code",
        "Actor1Type2Code",
        "Actor1Type3Code",
        "Actor2Name",
        "Actor2KnownGroupCode",
        "Actor2Type1Code",
        "Actor2Type2Code",
        "Actor2Type3Code",
    ]

    for chunk in pd.read_csv(
        GDELT_PATH,
        usecols=usecols,
        dtype=str,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        for _, row in chunk.iterrows():
            actor1_type_info = build_gdelt_type_info(
                row["Actor1KnownGroupCode"],
                row["Actor1Type1Code"],
                row["Actor1Type2Code"],
                row["Actor1Type3Code"],
            )

            actor2_type_info = build_gdelt_type_info(
                row["Actor2KnownGroupCode"],
                row["Actor2Type1Code"],
                row["Actor2Type2Code"],
                row["Actor2Type3Code"],
            )

            add_actor(
                records,
                name=row["Actor1Name"],
                source="GDELT",
                type_info=actor1_type_info,
            )

            add_actor(
                records,
                name=row["Actor2Name"],
                source="GDELT",
                type_info=actor2_type_info,
            )


# ============================================================
# 6. 输出 actor_candidates.csv
# ============================================================

def records_to_dataframe(records):
    rows = []

    for normalized_name, rec in records.items():
        sources = sorted(rec["sources"])
        type_info = sorted(rec["type_info_set"])
        sector_info = sorted(rec["sector_info_set"])

        row = {
            "raw_name": rec["raw_name"],
            "normalized_name": normalized_name,
            "frequency": rec["frequency"],
            "source_count": len(sources),
            "sources": ";".join(sources),
            "type_info": ";".join(type_info),
            "sector_info": ";".join(sector_info),
            "org_signal": rec["frequency"],
            "_type_info_set": rec["type_info_set"],
            "_sector_info_set": rec["sector_info_set"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(
        by=["frequency", "source_count"],
        ascending=[False, False],
    )

    return df


def save_actor_candidates(df):
    candidate_rows = []

    for _, row in df.iterrows():
        record = {
            "normalized_name": row["normalized_name"],
            "frequency": row["frequency"],
            "source_count": row["source_count"],
            "type_info_set": row["_type_info_set"],
            "sector_info_set": row["_sector_info_set"],
        }

        if should_keep_as_candidate(record):
            candidate_rows.append(row)

    candidate_df = pd.DataFrame(candidate_rows)

    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=[
            "raw_name",
            "normalized_name",
            "frequency",
            "source_count",
            "sources",
            "type_info",
            "sector_info",
            "org_signal",
        ])
    else:
        candidate_df = candidate_df[
            [
                "raw_name",
                "normalized_name",
                "frequency",
                "source_count",
                "sources",
                "type_info",
                "sector_info",
                "org_signal",
            ]
        ]

    candidate_df.to_csv(
        ACTOR_CANDIDATES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] actor_candidates.csv: {ACTOR_CANDIDATES_PATH}")
    print(f"[STAT] actor_candidates rows: {len(candidate_df)}")

    return candidate_df


# ============================================================
# 7. 输出 organizations.csv
# ============================================================

def save_organizations(df):
    org_rows = []

    for _, row in df.iterrows():
        record = {
            "normalized_name": row["normalized_name"],
            "frequency": row["frequency"],
            "source_count": row["source_count"],
            "type_info_set": row["_type_info_set"],
            "sector_info_set": row["_sector_info_set"],
        }

        if should_keep_as_organization(record):
            org_rows.append(row)

    org_df = pd.DataFrame(org_rows)

    if org_df.empty:
        org_df = pd.DataFrame(columns=[
            "org_id",
            "canonical_name",
            "raw_name",
            "normalized_name",
            "clean_name",
            "frequency",
            "source_count",
            "sources",
            "type_info",
            "sector_info",
            "org_signal",
            "old_org_id",
        ])
    else:
        # 如果不足 1000，则按频次从候选中补足
        if len(org_df) < TARGET_ORG_NUM:
            existing_names = set(org_df["normalized_name"])

            supplement = df[
                ~df["normalized_name"].isin(existing_names)
            ].copy()

            supplement = supplement[
                supplement["frequency"] >= MIN_FREQUENCY
            ]

            supplement = supplement.sort_values(
                by=["frequency", "source_count"],
                ascending=[False, False],
            )

            need_num = TARGET_ORG_NUM - len(org_df)
            org_df = pd.concat(
                [org_df, supplement.head(need_num)],
                ignore_index=True,
            )

        org_df = org_df.sort_values(
            by=["frequency", "source_count"],
            ascending=[False, False],
        ).reset_index(drop=True)

        org_ids = [
            f"ORG_{i:06d}"
            for i in range(1, len(org_df) + 1)
        ]

        org_df.insert(0, "org_id", org_ids)
        org_df.insert(1, "canonical_name", org_df["normalized_name"])
        org_df.insert(
            4,
            "clean_name",
            org_df["normalized_name"].apply(clean_name),
        )
        org_df["old_org_id"] = org_df["org_id"]

        org_df = org_df[
            [
                "org_id",
                "canonical_name",
                "raw_name",
                "normalized_name",
                "clean_name",
                "frequency",
                "source_count",
                "sources",
                "type_info",
                "sector_info",
                "org_signal",
                "old_org_id",
            ]
        ]

    org_df.to_csv(
        ORGANIZATIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] organizations.csv: {ORGANIZATIONS_PATH}")
    print(f"[STAT] organizations rows: {len(org_df)}")

    return org_df


# ============================================================
# 8. 主流程
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ICEWS_PATH.exists():
        raise FileNotFoundError(f"ICEWS 文件不存在: {ICEWS_PATH}")

    if not GDELT_PATH.exists():
        raise FileNotFoundError(f"GDELT 文件不存在: {GDELT_PATH}")

    records = {}

    process_icews(records)
    process_gdelt(records)

    all_df = records_to_dataframe(records)

    print(f"[STAT] total unique actors: {len(all_df)}")

    save_actor_candidates(all_df)
    save_organizations(all_df)

    print("[DONE] 03_extract_organizations.py finished.")


if __name__ == "__main__":
    main()
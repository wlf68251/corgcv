# scripts/04_align_organizations.py
# -*- coding: utf-8 -*-

"""
04_align_organizations.py

功能：
    对 03_extract_organizations.py 生成的 organizations_raw.csv 进行组织名称规范化与实体对齐。

输入：
    ../data/processed/organizations_raw.csv

输出：
    ../data/processed/organizations.csv
    ../data/processed/organization_aliases.csv

本版修改重点：
    1. 保留括号及括号内内容。
       例如：
           Head of Government (Ukraine)
           -> HEAD OF GOVERNMENT (UKRAINE)

       不再清洗成：
           HEAD OF GOVERNMENT

       原因：
           括号内的国家/地区信息是区分不同国家同名组织、部门、职位的重要上下文。

    2. 只过滤“泛化名称”，不按包含关键词过滤。
       会过滤：
           COLLEGE
           UNIVERSITY
           SCHOOL
           CORPORATION
           INDUSTRY

       不会过滤：
           HARVARD COLLEGE
           UNIVERSITY OF OXFORD
           CORPORATION FOR PUBLIC BROADCASTING
           POLICE (ISRAEL)
           GOVERNMENT (UKRAINE)

    3. 模糊合并时，如果名称主体相同但括号限定词不同，不合并。
       例如：
           HEAD OF GOVERNMENT (UKRAINE)
           HEAD OF GOVERNMENT (RUSSIA)
       不会被合并。
"""

import os
import re
import pandas as pd


PROCESSED_DIR = "../data/processed"

ORGANIZATIONS_IN = os.path.join(PROCESSED_DIR, "organizations_raw.csv")
ORGANIZATIONS_OUT = os.path.join(PROCESSED_DIR, "organizations.csv")
ALIASES_OUT = os.path.join(PROCESSED_DIR, "organization_aliases.csv")
ENABLE_FUZZY_MERGE = True

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


ABBREVIATION_MAP = {
    "UNITED NATIONS": "UN",
    "U N": "UN",
    "U NATIONS": "UN",
    "EUROPEAN UNION": "EU",
    "NORTH ATLANTIC TREATY ORGANIZATION": "NATO",
    "WORLD HEALTH ORGANIZATION": "WHO",
    "WORLD TRADE ORGANIZATION": "WTO",
    "INTERNATIONAL MONETARY FUND": "IMF",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
    "U S": "US",
    "USA": "US",
    "PEOPLES REPUBLIC OF CHINA": "CHINA",
    "PEOPLE S REPUBLIC OF CHINA": "CHINA",
    "PRC": "CHINA",
    "RUSSIAN FEDERATION": "RUSSIA",
    "UKRAINE GOVERNMENT": "UKRAINE",
}


STOP_PREFIXES = [
    "THE "
]


GENERIC_ORG_NAMES = {
    "COLLEGE",
    "UNIVERSITY",
    "SCHOOL",
    "ACADEMY",
    "FACULTY",
    "CAMPUS",
    "CORPORATION",
    "COMPANY",
    "INDUSTRY",
    "BUSINESS",
    "BANK",
    "FIRM",
    "ENTERPRISE",
    "MEDIA",
    "JOURNALISTS",
    "JOURNALIST",
    "CITIZENS",
    "CITIZEN",
    "CIVILIANS",
    "CIVILIAN",
    "PEOPLE",
    "PUBLIC",
    "STUDENTS",
    "STUDENT",
    "GOVERNMENT",
    "MILITARY",
    "POLICE",
    "ARMY",
    "NAVY",
    "AIR FORCE",
    "PARLIAMENT",
    "CONGRESS",
    "SENATE",
    "COURT",
    "JUDICIARY",
    "MINISTRY",
    "CABINET",
    "OPPOSITION",
    "REBELS",
    "REBEL",
    "PROTESTERS",
    "PROTESTER",
    "PARTY",
    "NGO",
}


def clean_name(name):
    """
    清洗组织名称，但保留括号及括号内内容。

    示例：
        Head of Government (Ukraine)
        -> HEAD OF GOVERNMENT (UKRAINE)

        Police (Israel)
        -> POLICE (ISRAEL)
    """
    if pd.isna(name):
        return ""

    name = str(name).upper().strip()

    name = name.replace("[", "(").replace("]", ")")
    name = name.replace("{", "(").replace("}", ")")

    name = re.sub(r"[^A-Z0-9\s\(\)]", " ", name)

    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"\s+\)", ")", name)
    name = re.sub(r"\(\s+", "(", name)
    name = re.sub(r"\s+\(", " (", name)

    name = re.sub(r"\(\s*\)", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    for p in STOP_PREFIXES:
        if name.startswith(p):
            name = name[len(p):].strip()

    if name in ABBREVIATION_MAP:
        name = ABBREVIATION_MAP[name]

    return name


def normalize_generic_check_name(name):
    return clean_name(name)


def is_generic_org_name(name):
    """
    只过滤完全等于泛化名称的候选。

    会过滤：
        COLLEGE
        UNIVERSITY
        CORPORATION

    不会过滤：
        HARVARD COLLEGE
        UNIVERSITY OF OXFORD
        GOVERNMENT (UKRAINE)
        POLICE (ISRAEL)
    """
    text = normalize_generic_check_name(name)

    if not text:
        return False

    return text in GENERIC_ORG_NAMES


def should_filter_org(row):
    """
    只检查主名称字段是否为泛化名称。

    不检查 all_raw_names，避免某个别名是 COLLEGE 时误删具体组织。
    """
    for col in [
        "canonical_name",
        "raw_name",
        "normalized_name",
        "clean_name",
    ]:
        if col in row.index:
            value = row.get(col, "")
            if is_generic_org_name(value):
                return True

    return False


def split_aliases(raw):
    if pd.isna(raw):
        return []

    parts = []

    for x in str(raw).split(";"):
        x = x.strip()
        if x:
            parts.append(x)

    return parts


def extract_parentheses_content(name):
    if not name:
        return set()

    items = re.findall(r"\((.*?)\)", str(name))
    return {x.strip().upper() for x in items if x.strip()}


def remove_parentheses_content(name):
    if not name:
        return ""

    text = re.sub(r"\(.*?\)", " ", str(name))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def should_skip_fuzzy_merge(name_i, name_j):
    """
    括号限定词不同，不进行模糊合并。
    """
    name_i = str(name_i).strip()
    name_j = str(name_j).strip()

    if not name_i or not name_j:
        return True

    par_i = extract_parentheses_content(name_i)
    par_j = extract_parentheses_content(name_j)

    base_i = remove_parentheses_content(name_i)
    base_j = remove_parentheses_content(name_j)

    if base_i == base_j and par_i and par_j and par_i != par_j:
        return True

    if base_i == base_j and ((par_i and not par_j) or (par_j and not par_i)):
        return True

    return False


def main():
    if not os.path.exists(ORGANIZATIONS_IN):
        print(f"未找到文件: {ORGANIZATIONS_IN}")
        print("请先运行 scripts/03_extract_organizations.py")
        return

    orgs = pd.read_csv(ORGANIZATIONS_IN, dtype=str, low_memory=False).fillna("")

    if "canonical_name" not in orgs.columns:
        if "raw_name" in orgs.columns:
            orgs["canonical_name"] = orgs["raw_name"]
        else:
            orgs["canonical_name"] = ""

    if "raw_name" not in orgs.columns:
        orgs["raw_name"] = orgs["canonical_name"]

    if "org_id" not in orgs.columns:
        orgs.insert(0, "org_id", ["ORG_%06d" % (i + 1) for i in range(len(orgs))])

    orgs["clean_name"] = orgs["canonical_name"].apply(clean_name)

    before_filter_count = len(orgs)

    filter_mask = orgs.apply(should_filter_org, axis=1)
    filtered_orgs = orgs[filter_mask].copy()
    orgs = orgs[~filter_mask].copy()

    print("泛化组织名称过滤完成")
    print("过滤前组织数量:", before_filter_count)
    print("过滤掉数量:", len(filtered_orgs))
    print("过滤后组织数量:", len(orgs))

    exact_map = {}
    kept_rows = []
    alias_rows = []

    for _, row in orgs.iterrows():
        clean = row["clean_name"]
        old_org_id = row["org_id"]

        if not clean:
            continue

        if clean not in exact_map:
            exact_map[clean] = old_org_id
            kept_rows.append(row.to_dict())

        main_old_id = exact_map[clean]

        alias_names = set()
        alias_names.add(str(row.get("canonical_name", "")))
        alias_names.add(str(row.get("raw_name", "")))

        if "all_raw_names" in orgs.columns:
            for a in split_aliases(row.get("all_raw_names", "")):
                alias_names.add(a)

        for alias in alias_names:
            if alias and alias.lower() != "nan":
                if is_generic_org_name(alias):
                    continue

                alias_rows.append({
                    "old_org_id": main_old_id,
                    "alias": alias,
                    "alias_clean": clean_name(alias),
                    "match_type": "exact_clean"
                })

    aligned = pd.DataFrame(kept_rows)

    if aligned.empty:
        print("对齐后组织为空，请检查 organizations.csv")
        return

    if ENABLE_FUZZY_MERGE and HAS_RAPIDFUZZ:
        print("检测到 RapidFuzz，执行保守模糊合并...")

        aligned = aligned.reset_index(drop=True)
        remove_old_ids = set()
        redirect = {}

        for i in range(len(aligned)):
            id_i = aligned.loc[i, "org_id"]

            if id_i in remove_old_ids:
                continue

            name_i = aligned.loc[i, "clean_name"]

            for j in range(i + 1, len(aligned)):
                id_j = aligned.loc[j, "org_id"]

                if id_j in remove_old_ids:
                    continue

                name_j = aligned.loc[j, "clean_name"]

                if not name_i or not name_j:
                    continue

                if len(name_i) < 8 or len(name_j) < 8:
                    continue

                if should_skip_fuzzy_merge(name_i, name_j):
                    continue

                score = fuzz.token_sort_ratio(name_i, name_j)

                if score >= 97:
                    redirect[id_j] = id_i
                    remove_old_ids.add(id_j)

                    alias_rows.append({
                        "old_org_id": id_i,
                        "alias": aligned.loc[j, "canonical_name"],
                        "alias_clean": name_j,
                        "match_type": f"fuzzy_{score}"
                    })

        aligned = aligned[~aligned["org_id"].isin(remove_old_ids)].copy()

        for row in alias_rows:
            oid = row["old_org_id"]
            if oid in redirect:
                row["old_org_id"] = redirect[oid]

    else:
        print("未安装 RapidFuzz，跳过模糊合并。")

    aligned = aligned.reset_index(drop=True)

    old_to_new = {}
    for i, old_id in enumerate(aligned["org_id"].astype(str).tolist()):
        old_to_new[old_id] = "ORG_%06d" % (i + 1)

    aligned["old_org_id"] = aligned["org_id"].astype(str)
    aligned["org_id"] = aligned["old_org_id"].map(old_to_new)

    alias_df = pd.DataFrame(alias_rows)

    if alias_df.empty:
        alias_df = pd.DataFrame(columns=[
            "org_id",
            "old_org_id",
            "alias",
            "alias_clean",
            "match_type"
        ])
    else:
        alias_df["old_org_id"] = alias_df["old_org_id"].astype(str)
        alias_df = alias_df[alias_df["old_org_id"].isin(old_to_new.keys())].copy()
        alias_df["org_id"] = alias_df["old_org_id"].map(old_to_new)

        alias_df = alias_df[
            [
                "org_id",
                "old_org_id",
                "alias",
                "alias_clean",
                "match_type"
            ]
        ]

        alias_df = alias_df.drop_duplicates(
            subset=["org_id", "alias_clean", "alias"]
        )

    keep_cols = [
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
        "known_group_info",
        "org_signal",
        "all_raw_names",
        "old_org_id"
    ]

    for col in keep_cols:
        if col not in aligned.columns:
            aligned[col] = ""

    aligned = aligned[keep_cols]

    aligned.to_csv(ORGANIZATIONS_OUT, index=False, encoding="utf-8-sig")
    alias_df.to_csv(ALIASES_OUT, index=False, encoding="utf-8-sig")

    print("组织名称规范化与实体对齐完成")
    print("输出:", ORGANIZATIONS_OUT)
    print("输出:", ALIASES_OUT)
    print("对齐后组织数量:", len(aligned))
    print("别名数量:", len(alias_df))


if __name__ == "__main__":
    main()

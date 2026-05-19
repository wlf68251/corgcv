# scripts/04_align_organizations.py
import os
import re
import pandas as pd

PROCESSED_DIR = "../data/processed"

ORGANIZATIONS_IN = os.path.join(PROCESSED_DIR, "organizations.csv")
ORGANIZATIONS_OUT = os.path.join(PROCESSED_DIR, "organizations.csv")
ALIASES_OUT = os.path.join(PROCESSED_DIR, "organization_aliases.csv")

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
    "PRC": "CHINA",
    "RUSSIAN FEDERATION": "RUSSIA",
    "UKRAINE GOVERNMENT": "UKRAINE",
}


STOP_PREFIXES = [
    "THE "
]


def clean_name(name):
    if pd.isna(name):
        return ""

    name = str(name).upper().strip()

    name = re.sub(r"\(.*?\)", " ", name)
    name = re.sub(r"\[.*?\]", " ", name)
    name = re.sub(r"[^A-Z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    for p in STOP_PREFIXES:
        if name.startswith(p):
            name = name[len(p):].strip()

    if name in ABBREVIATION_MAP:
        name = ABBREVIATION_MAP[name]

    return name


def split_aliases(raw):
    if pd.isna(raw):
        return []

    parts = []

    for x in str(raw).split(";"):
        x = x.strip()
        if x:
            parts.append(x)

    return parts


def main():
    if not os.path.exists(ORGANIZATIONS_IN):
        print(f"未找到文件: {ORGANIZATIONS_IN}")
        print("请先运行 scripts/03_extract_organizations.py")
        return

    orgs = pd.read_csv(ORGANIZATIONS_IN, low_memory=False)

    if "canonical_name" not in orgs.columns:
        orgs["canonical_name"] = orgs["raw_name"]

    orgs["clean_name"] = orgs["canonical_name"].apply(clean_name)

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

    # 可选：非常保守的模糊合并
    if HAS_RAPIDFUZZ:
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

                # 名称太短不模糊合并，避免 US、UN 等误合并
                if len(name_i) < 8 or len(name_j) < 8:
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

        # 同步 alias old_org_id
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
# scripts/prepare_further.py
import os
import pandas as pd

RAW_DIR = "../data/raw/gdelt"

INPUT_CSV = os.path.join(RAW_DIR, "gdelt_raw_events_big.csv")
OUTPUT_CSV = os.path.join(RAW_DIR, "gdelt_raw_events.csv")

CHUNKSIZE = 200000

# 目标输出大小，单位 MB
TARGET_OUTPUT_MB = 500

# 是否启用大小限制
ENABLE_SIZE_LIMIT = False

# 组织过滤模式：
# both_org：主体和客体都像组织才保留，数据更少
# any_org：主体或客体任意一方像组织就保留，数据更多
ORG_FILTER_MODE = "both_org"

# 如果过滤后仍然过大，可改成 2、3、5
# 表示每 N 条组织相关事件保留 1 条
SAMPLE_EVERY_N_ROWS = 20

# 是否要求 Actor1Name 和 Actor2Name 都存在
REQUIRE_BOTH_ACTOR_NAMES = True


KEEP_COLUMNS = [
    "GLOBALEVENTID",
    "SQLDATE",

    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",

    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",

    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",

    "ActionGeo_FullName",
    "ActionGeo_CountryCode",

    "DATEADDED",
    "SOURCEURL"
]


ORG_TYPE_CODES = {
    "GOV",
    "MIL",
    "COP",
    "BUS",
    "MNC",
    "NGO",
    "IGO",
    "IMG",
    "INT",
    "MED",
    "EDU",
    "ELI",
    "PTY",
    "OPP",
    "REB",
    "SEP",
    "SPY",
    "JUD",
    "LEG",
    "CRM",
}


ORG_NAME_HINTS = [
    "GOVERNMENT",
    "MINISTRY",
    "DEPARTMENT",
    "AGENCY",
    "ADMINISTRATION",
    "COMMISSION",
    "COMMITTEE",
    "COUNCIL",
    "PARLIAMENT",
    "SENATE",
    "CONGRESS",
    "COURT",
    "POLICE",
    "MILITARY",
    "ARMY",
    "NAVY",
    "AIR FORCE",
    "EMBASSY",
    "CONSULATE",
    "UNIVERSITY",
    "COLLEGE",
    "INSTITUTE",
    "BANK",
    "COMPANY",
    "CORPORATION",
    "GROUP",
    "ASSOCIATION",
    "ORGANIZATION",
    "PARTY",
    "NATO",
    "UNITED NATIONS",
    "EUROPEAN UNION",
    "WORLD BANK",
    "IMF",
    "WHO",
    "WTO",
]


BAD_ACTOR_NAMES = {
    "",
    "NAN",
    "NONE",
    "NULL",
    "UNKNOWN",
    "PEOPLE",
    "CITIZENS",
    "RESIDENTS",
    "PROTESTERS",
    "STUDENTS",
    "WORKERS",
    "WOMEN",
    "MEN",
    "CHILDREN",
    "CIVILIANS",
    "REFUGEES",
    "MIGRANTS",
}


def get_output_size_mb():
    if not os.path.exists(OUTPUT_CSV):
        return 0.0
    return os.path.getsize(OUTPUT_CSV) / 1024 / 1024


def normalize_name_series(s):
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )


def actor_has_org_signal(df, prefix):
    """
    判断 Actor1 或 Actor2 是否具有组织机构信号。
    prefix: Actor1 或 Actor2
    """

    name_col = f"{prefix}Name"
    known_col = f"{prefix}KnownGroupCode"
    type_cols = [
        f"{prefix}Type1Code",
        f"{prefix}Type2Code",
        f"{prefix}Type3Code",
    ]

    name = normalize_name_series(df[name_col])

    known_signal = (
        df[known_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    type_signal = pd.Series(False, index=df.index)

    for col in type_cols:
        col_signal = (
            df[col]
            .fillna("")
            .astype(str)
            .str.upper()
            .isin(ORG_TYPE_CODES)
        )
        type_signal = type_signal | col_signal

    name_signal = pd.Series(False, index=df.index)

    for hint in ORG_NAME_HINTS:
        name_signal = name_signal | name.str.contains(hint, regex=False)

    bad_name_signal = name.isin(BAD_ACTOR_NAMES)

    return (known_signal | type_signal | name_signal) & (~bad_name_signal)


def filter_chunk(chunk):
    """
    从大 GDELT CSV 中进一步筛选组织相关事件。
    """

    if chunk.empty:
        return chunk

    required_cols = [
        "Actor1Name",
        "Actor2Name",
        "Actor1KnownGroupCode",
        "Actor2KnownGroupCode",
        "Actor1Type1Code",
        "Actor1Type2Code",
        "Actor1Type3Code",
        "Actor2Type1Code",
        "Actor2Type2Code",
        "Actor2Type3Code",
    ]

    for col in required_cols:
        if col not in chunk.columns:
            chunk[col] = ""

    chunk["Actor1Name"] = chunk["Actor1Name"].fillna("").astype(str).str.strip()
    chunk["Actor2Name"] = chunk["Actor2Name"].fillna("").astype(str).str.strip()

    if REQUIRE_BOTH_ACTOR_NAMES:
        chunk = chunk[
            (chunk["Actor1Name"] != "") &
            (chunk["Actor2Name"] != "")
        ].copy()
    else:
        chunk = chunk[
            (chunk["Actor1Name"] != "") |
            (chunk["Actor2Name"] != "")
        ].copy()

    if chunk.empty:
        return chunk

    actor1_org = actor_has_org_signal(chunk, "Actor1")
    actor2_org = actor_has_org_signal(chunk, "Actor2")

    if ORG_FILTER_MODE == "both_org":
        chunk = chunk[actor1_org & actor2_org].copy()
    else:
        chunk = chunk[actor1_org | actor2_org].copy()

    if chunk.empty:
        return chunk

    if SAMPLE_EVERY_N_ROWS > 1:
        chunk = chunk.iloc[::SAMPLE_EVERY_N_ROWS].copy()

    keep_cols = [c for c in KEEP_COLUMNS if c in chunk.columns]
    chunk = chunk[keep_cols].copy()

    return chunk


def write_limited_csv(df, first_write):
    """
    写入输出文件，并尽量控制在 TARGET_OUTPUT_MB 附近。
    """

    if df.empty:
        return first_write, 0, False

    if not ENABLE_SIZE_LIMIT:
        df.to_csv(
            OUTPUT_CSV,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
            encoding="utf-8-sig"
        )
        return False, len(df), True

    current_mb = get_output_size_mb()

    if current_mb >= TARGET_OUTPUT_MB:
        return first_write, 0, False

    sample_n = min(1000, len(df))
    sample_csv = df.head(sample_n).to_csv(index=False, header=first_write)
    avg_bytes_per_row = max(len(sample_csv.encode("utf-8")) / sample_n, 1)

    remaining_bytes = (TARGET_OUTPUT_MB - current_mb) * 1024 * 1024
    max_rows_can_write = int(remaining_bytes / avg_bytes_per_row)

    if max_rows_can_write <= 0:
        return first_write, 0, False

    actual_df = df.head(max_rows_can_write).copy()

    actual_df.to_csv(
        OUTPUT_CSV,
        mode="w" if first_write else "a",
        index=False,
        header=first_write,
        encoding="utf-8-sig"
    )

    return False, len(actual_df), True


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"未找到输入文件: {INPUT_CSV}")
        print("请先将原始大文件命名为 gdelt_raw_events_big.csv")
        return

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print("开始进一步处理 GDELT 大文件")
    print("输入:", INPUT_CSV)
    print("输出:", OUTPUT_CSV)
    print("组织过滤模式:", ORG_FILTER_MODE)
    print("目标大小:", TARGET_OUTPUT_MB, "MB")

    first_write = True

    total_raw_rows = 0
    total_org_rows = 0
    total_written_rows = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            INPUT_CSV,
            chunksize=CHUNKSIZE,
            low_memory=False
        ),
        start=1
    ):
        total_raw_rows += len(chunk)

        filtered = filter_chunk(chunk)

        total_org_rows += len(filtered)

        first_write, written_rows, wrote = write_limited_csv(
            filtered,
            first_write
        )

        total_written_rows += written_rows

        current_mb = get_output_size_mb()

        print(
            f"chunk {chunk_no}: "
            f"raw={len(chunk)}, "
            f"org_related={len(filtered)}, "
            f"written={written_rows}, "
            f"output={current_mb:.2f}MB"
        )

        if ENABLE_SIZE_LIMIT and current_mb >= TARGET_OUTPUT_MB:
            print(f"输出文件已达到 {TARGET_OUTPUT_MB}MB 左右，停止处理。")
            break

    final_mb = get_output_size_mb()

    print("GDELT 进一步过滤完成")
    print("输出:", OUTPUT_CSV)
    print(f"最终大小: {final_mb:.2f} MB")
    print("原始读取行数:", total_raw_rows)
    print("组织相关行数:", total_org_rows)
    print("实际写出行数:", total_written_rows)

    if final_mb == 0:
        print("警告：输出文件为空，请放宽组织过滤规则。")
        print("可以尝试将 ORG_FILTER_MODE 从 both_org 改成 any_org。")


if __name__ == "__main__":
    main()
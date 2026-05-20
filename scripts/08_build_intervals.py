# scripts/08_build_intervals.py
import os
import calendar
import pandas as pd

PROCESSED_DIR = "../data/processed"

INPUT_FILE = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "org_relation_intervals.tsv")

# 08 直接输出到 PaTeCon resource，供 10_run_patecon.py 使用
PATECON_RESOURCE_DIR = "../PaTeCon-master/resource"
PATECON_OUTPUT_FILE = os.path.join(PATECON_RESOURCE_DIR, "org_relation_intervals.tsv")

MIN_EVENT_COUNT = 1
MIN_CONFIDENCE = 0.0

VALID_RELATIONS = {f"{i:02d}" for i in range(1, 21)}


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_relation_type(value):
    """
    cameotop 分支：
    relation_type 应为 01~20。

    这里不加 CAMEO_ 前缀。
    PaTeCon property 直接使用 01、02、...、20。
    """
    text = safe_str(value)

    if not text:
        return ""

    if "." in text:
        text = text.split(".")[0]

    text = "".join(ch for ch in text if ch.isdigit())

    if not text:
        return ""

    text = text.zfill(2)

    if text in VALID_RELATIONS:
        return text

    return ""


def month_to_interval(event_month):
    """
    将 event_month 转换为 PaTeCon 需要的时间区间格式。

    输入：
        2023-01

    输出：
        20230101, 20230131

    注意：
        PaTeCon 输入时间不要使用 2023-01-01 这种带横线格式。
    """
    text = safe_str(event_month)

    if len(text) != 7 or "-" not in text:
        return "", ""

    year_str, month_str = text.split("-", 1)

    try:
        year = int(year_str)
        month = int(month_str)
    except Exception:
        return "", ""

    if month < 1 or month > 12:
        return "", ""

    last_day = calendar.monthrange(year, month)[1]

    start_time = f"{year:04d}{month:02d}01"
    end_time = f"{year:04d}{month:02d}{last_day:02d}"

    return start_time, end_time
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"未找到输入文件: {INPUT_FILE}")
        print("请先运行 07_score_relations.py")
        return

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(PATECON_RESOURCE_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, dtype=str, low_memory=False).fillna("")

    required_cols = [
        "subject_org_id",
        "object_org_id",
        "event_month",
        "relation_type",
        "event_count",
        "confidence",
        "status",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print("缺少必要字段:", missing)
        print("当前字段:", list(df.columns))
        return

    rows = []

    for _, row in df.iterrows():
        status = safe_str(row.get("status", "active"))
        if status not in {"active", "review", "downgraded"}:
            continue

        relation = normalize_relation_type(row["relation_type"])
        if not relation:
            continue

        try:
            event_count = int(float(row["event_count"]))
        except Exception:
            event_count = 0

        try:
            confidence = float(row["confidence"])
        except Exception:
            confidence = 0.0

        if event_count < MIN_EVENT_COUNT:
            continue

        if confidence < MIN_CONFIDENCE:
            continue

        start_time, end_time = month_to_interval(row["event_month"])

        if not start_time or not end_time:
            continue

        subject = safe_str(row["subject_org_id"])
        obj = safe_str(row["object_org_id"])

        if not subject or not obj:
            continue

        if subject == obj:
            continue

        rows.append({
            "subject": subject,
            "property": relation,
            "object": obj,
            "start_time": start_time,
            "end_time": end_time,
        })

    out_df = pd.DataFrame(rows)

    if not out_df.empty:
        out_df = out_df.drop_duplicates()
        out_df = out_df.sort_values(
            by=["subject", "property", "object", "start_time", "end_time"]
        )

    # PaTeCon 通常读取无表头 TSV：subject property object start end
    out_df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        header=False,
        encoding="utf-8"
    )

    out_df.to_csv(
        PATECON_OUTPUT_FILE,
        sep="\t",
        index=False,
        header=False,
        encoding="utf-8"
    )

    print("[OK] 已生成 PaTeCon 区间文件:")
    print(" -", OUTPUT_FILE)
    print(" -", PATECON_OUTPUT_FILE)
    print("[STAT] interval rows:", len(out_df))
    print("[STAT] relation types:", sorted(out_df["property"].unique()) if not out_df.empty else [])


if __name__ == "__main__":
    main()
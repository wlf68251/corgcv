# scripts/08_build_intervals.py
import os
import calendar
import shutil
import pandas as pd

PROCESSED_DIR = "../data/processed"
PROJECT_RESOURCE_DIR = "../resource"
PATECON_RESOURCE_DIR = "../PaTeCon-master/resource"

INPUT_FILE = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "org_relation_intervals.tsv")
PROJECT_RESOURCE_OUTPUT_FILE = os.path.join(PROJECT_RESOURCE_DIR, "org_relation_intervals.tsv")
PATECON_RESOURCE_OUTPUT_FILE = os.path.join(PATECON_RESOURCE_DIR, "org_relation_intervals.tsv")

MIN_EVENT_COUNT = 2
MIN_CONFIDENCE = 0.45
DOMINANT_RATIO = 0.60
MIXED_RATIO = 0.30


def month_start_end(month_str):
    """
    输入:
        2023-01
    输出:
        20230101, 20230131

    PaTeCon 当前源码要求 start_time/end_time 可以 int()。
    所以这里不能输出 2023-01-01。
    """
    year, month = map(int, str(month_str).split("-"))
    last_day = calendar.monthrange(year, month)[1]

    start_time = f"{year:04d}{month:02d}01"
    end_time = f"{year:04d}{month:02d}{last_day:02d}"

    return start_time, end_time


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(PROJECT_RESOURCE_DIR, exist_ok=True)
    os.makedirs(PATECON_RESOURCE_DIR, exist_ok=True)


def main():
    ensure_dirs()

    if not os.path.exists(INPUT_FILE):
        print(f"未找到输入文件: {INPUT_FILE}")
        print("请先运行 scripts/07_score_relations.py")
        return

    edges = pd.read_csv(INPUT_FILE, low_memory=False)

    required_cols = [
        "subject_org_id",
        "object_org_id",
        "event_month",
        "relation_type",
        "event_count",
        "confidence"
    ]

    for col in required_cols:
        if col not in edges.columns:
            print(f"缺少必要字段: {col}")
            print("当前字段:", list(edges.columns))
            return

    print("开始构建 PaTeCon 月度关系区间...")

    if "status" in edges.columns:
        edges = edges[edges["status"] == "active"].copy()

    edges["event_count"] = edges["event_count"].apply(safe_int)
    edges["confidence"] = edges["confidence"].apply(safe_float)

    candidate_edges = edges[
        (edges["event_count"] >= MIN_EVENT_COUNT) |
        (edges["confidence"] >= MIN_CONFIDENCE)
    ].copy()

    if candidate_edges.empty:
        print("没有满足条件的关系边，无法生成 PaTeCon 区间。")
        return

    interval_rows = []

    group_cols = [
        "subject_org_id",
        "object_org_id",
        "event_month"
    ]

    for (subject_org_id, object_org_id, event_month), group in candidate_edges.groupby(group_cols):
        total_count = group["event_count"].sum()

        if total_count <= 0:
            continue

        group = group.sort_values(
            by=["event_count", "confidence"],
            ascending=False
        )

        top_row = group.iloc[0]
        top_ratio = top_row["event_count"] / total_count

        if top_ratio >= DOMINANT_RATIO:
            final_relation_type = top_row["relation_type"]
        else:
            strong_types = group[
                group["event_count"] / total_count >= MIXED_RATIO
            ]

            if len(strong_types) >= 2:
                final_relation_type = "mixed_relation"
            else:
                final_relation_type = top_row["relation_type"]

        start_time, end_time = month_start_end(event_month)

        interval_rows.append({
            "subject": str(subject_org_id),
            "property": str(final_relation_type),
            "object": str(object_org_id),
            "start_time": str(start_time),
            "end_time": str(end_time)
        })

    intervals = pd.DataFrame(interval_rows)

    intervals = intervals.drop_duplicates(
        subset=["subject", "property", "object", "start_time", "end_time"]
    )

    intervals.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        header=False,
        encoding="utf-8"
    )

    shutil.copyfile(OUTPUT_FILE, PROJECT_RESOURCE_OUTPUT_FILE)
    shutil.copyfile(OUTPUT_FILE, PATECON_RESOURCE_OUTPUT_FILE)

    print("月度关系区间构建完成")
    print("输出:", OUTPUT_FILE)
    print("同步输出:", PROJECT_RESOURCE_OUTPUT_FILE)
    print("同步输出:", PATECON_RESOURCE_OUTPUT_FILE)
    print("区间数量:", len(intervals))


if __name__ == "__main__":
    main()
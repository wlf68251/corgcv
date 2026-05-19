# scripts/07_score_relations.py
import os
import math
import pandas as pd

PROCESSED_DIR = "../data/processed"

INPUT_FILE = os.path.join(PROCESSED_DIR, "relation_edges_before_check.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")


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


def compute_score(row):
    """
    关系置信度 =
    事件平均分
    + 跨源共同支持加分
    + 事件数量加分
    - 低证据惩罚

    最终分数限制在 [0, 1]。
    """

    source_datasets = str(row.get("source_datasets", ""))

    event_count = safe_int(row.get("event_count", 0))
    icews_count = safe_int(row.get("icews_event_count", 0))
    gdelt_count = safe_int(row.get("gdelt_event_count", 0))

    old_confidence = safe_float(row.get("confidence", 0.4))

    # 1. 事件平均分
    # ICEWS 种子图谱默认更可信，GDELT 单源稍低
    if "ICEWS" in source_datasets and "GDELT" in source_datasets:
        base_score = 0.70
    elif "ICEWS" in source_datasets:
        base_score = 0.55
    elif "GDELT" in source_datasets:
        base_score = 0.40
    else:
        base_score = old_confidence

    # 2. 跨源共同支持加分
    cross_source_bonus = 0.0
    if icews_count > 0 and gdelt_count > 0:
        cross_source_bonus = 0.20

    # 3. 事件数量加分
    # 使用 log，避免事件数量特别大时分数膨胀
    count_bonus = 0.0
    if event_count > 0:
        count_bonus = min(math.log1p(event_count) / 10.0, 0.20)

    # 4. 低证据惩罚
    low_evidence_penalty = 0.0

    if event_count <= 1:
        low_evidence_penalty += 0.15

    if "GDELT" in source_datasets and "ICEWS" not in source_datasets and gdelt_count <= 2:
        low_evidence_penalty += 0.15

    score = base_score + cross_source_bonus + count_bonus - low_evidence_penalty

    score = max(0.0, min(1.0, score))

    return round(score, 4)


def confidence_level(score):
    if score >= 0.75:
        return "high"
    elif score >= 0.50:
        return "medium"
    else:
        return "low"


def build_reason(row, score):
    source_datasets = str(row.get("source_datasets", ""))

    event_count = safe_int(row.get("event_count", 0))
    icews_count = safe_int(row.get("icews_event_count", 0))
    gdelt_count = safe_int(row.get("gdelt_event_count", 0))

    reasons = []

    if "ICEWS" in source_datasets:
        reasons.append("ICEWS支持")

    if "GDELT" in source_datasets:
        reasons.append("GDELT支持")

    if icews_count > 0 and gdelt_count > 0:
        reasons.append("跨源共同支持")

    if event_count <= 1:
        reasons.append("事件数量较少")

    if "GDELT" in source_datasets and "ICEWS" not in source_datasets and gdelt_count <= 2:
        reasons.append("GDELT单源低证据")

    reasons.append(f"最终置信度={score}")

    return "；".join(reasons)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"未找到输入文件: {INPUT_FILE}")
        print("请先运行 scripts/06_stream_fuse_gdelt.py")
        return

    edges = pd.read_csv(INPUT_FILE, low_memory=False)

    required_cols = [
        "edge_id",
        "subject_org_id",
        "object_org_id",
        "event_month",
        "relation_type",
        "event_count",
        "source_datasets",
        "icews_event_count",
        "gdelt_event_count"
    ]

    for col in required_cols:
        if col not in edges.columns:
            print(f"缺少必要字段: {col}")
            print("当前字段:", list(edges.columns))
            return

    print("开始计算关系置信度...")

    edges["confidence"] = edges.apply(compute_score, axis=1)
    edges["confidence_level"] = edges["confidence"].apply(confidence_level)
    edges["confidence_reason"] = edges.apply(
        lambda row: build_reason(row, row["confidence"]),
        axis=1
    )

    # 保留 active 状态
    if "status" not in edges.columns:
        edges["status"] = "active"

    # 为后续校验添加派生关系标记
    edges["is_derived_relation"] = True

    # 排序便于查看
    edges = edges.sort_values(
        by=["confidence", "event_count"],
        ascending=False
    )

    edges.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("关系置信度计算完成")
    print("输出:", OUTPUT_FILE)
    print("关系边数量:", len(edges))
    print("高置信度边数量:", len(edges[edges["confidence_level"] == "high"]))
    print("中置信度边数量:", len(edges[edges["confidence_level"] == "medium"]))
    print("低置信度边数量:", len(edges[edges["confidence_level"] == "low"]))


if __name__ == "__main__":
    main()
import pandas as pd
import re

INPUT_CSV = "icews_raw_events.csv"
OUTPUT_TSV = "icews_raw_events.tsv"


def clean_entity(value):
    if pd.isna(value):
        return "null"

    value = str(value).strip()
    value = value.replace("<", "").replace(">", "")
    value = value.replace("\t", " ")
    value = value.replace("\n", " ")

    if value == "" or value.lower() in ["nan", "none", "null"]:
        return "null"

    return value


def clean_property(value):
    if pd.isna(value):
        return "null"

    value = str(value).strip()

    value = value.replace("<", "").replace(">", "")
    value = value.replace("\t", " ")
    value = value.replace("\n", " ")

    if value == "" or value.lower() in ["nan", "none", "null"]:
        return "null"

    # 不转 int，不去前导 0
    # 例如 010 保持为 CAMEO_010
    return "CAMEO_" + value


def clean_date(value):
    if pd.isna(value):
        return "null"

    value = str(value).strip()

    if value == "" or value.lower() in ["nan", "none", "null"]:
        return "null"

    # 2023-01-01 -> 20230101
    digits = re.sub(r"\D", "", value)

    if len(digits) >= 8:
        return digits[:8]

    return "null"


def main():
    # 关键：dtype=str，防止 CAMEO Code 中的 010 被读成 10
    df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)

    out = pd.DataFrame()
    out["subject"] = df["Source Name"].apply(clean_entity)
    out["property"] = df["CAMEO Code"].apply(clean_property)
    out["object"] = df["Target Name"].apply(clean_entity)

    # 事件型数据只有单点时间，所以开始时间 = 结束时间
    out["start_time"] = df["Event Date"].apply(clean_date)
    out["end_time"] = df["Event Date"].apply(clean_date)

    # 删除核心字段缺失的数据
    out = out[
        (out["subject"] != "null") &
        (out["property"] != "null") &
        (out["object"] != "null") &
        (out["start_time"] != "null")
    ]

    # 去重
    out = out.drop_duplicates()

    # PaTeCon 需要 TSV，且不需要表头
    out.to_csv(
        OUTPUT_TSV,
        sep="\t",
        index=False,
        header=False,
        encoding="utf-8"
    )

    print("转换完成：", OUTPUT_TSV)
    print("事实数量：", len(out))
    print(out.head(10))

    print("CAMEO 示例：")
    print(out["property"].drop_duplicates().head(20).tolist())


if __name__ == "__main__":
    main()
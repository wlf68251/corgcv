# scripts/01_prepare_icews.py
import pandas as pd
import os
import glob

RAW_DIR = "../data/raw/icews"
os.makedirs(RAW_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RAW_DIR, "icews_raw_events.csv")

# 获取所有 .tab 或 .tab.zip 文件
tab_files = (
    glob.glob(os.path.join(RAW_DIR, "*.tab")) +
    glob.glob(os.path.join(RAW_DIR, "*.tab.zip"))
)

df_list = []

for f in tab_files:
    print("读取文件：", f)

    # 关键：dtype=str，防止 CAMEO Code 中的 010 被读成 10
    if f.endswith(".zip"):
        df = pd.read_csv(
            f,
            sep="\t",
            compression="zip",
            dtype=str,
            low_memory=False
        )
    else:
        df = pd.read_csv(
            f,
            sep="\t",
            dtype=str,
            low_memory=False
        )

    df_list.append(df)

if not df_list:
    raise FileNotFoundError(f"在 {RAW_DIR} 下没有找到 .tab 或 .tab.zip 文件")

# 合并所有数据
icews_df = pd.concat(df_list, ignore_index=True)

# 保留原始 Event Date 字符串，另建一个临时日期列用于筛选
icews_df["_Event Date Parsed"] = pd.to_datetime(
    icews_df["Event Date"],
    errors="coerce"
)

# 筛选出 2023-01-01 到 2023-04-30 的事件
icews_df = icews_df[
    (icews_df["_Event Date Parsed"] >= "2023-01-01") &
    (icews_df["_Event Date Parsed"] <= "2023-04-30")
].copy()

# 删除临时日期列，避免输出多余字段
icews_df.drop(columns=["_Event Date Parsed"], inplace=True)

# # 再次确保 CAMEO Code 是字符串，并补齐前导 0
# if "CAMEO Code" in icews_df.columns:
#     icews_df["CAMEO Code"] = (
#         icews_df["CAMEO Code"]
#         .astype(str)
#         .str.strip()
#         .str.replace(r"\.0$", "", regex=True)
#     )

#     # CAMEO 编码至少补齐到 3 位：
#     # 10 -> 010, 20 -> 020, 46 -> 046, 51 -> 051
#     icews_df["CAMEO Code"] = icews_df["CAMEO Code"].apply(
#         lambda x: x.zfill(3)
#         if x and x.lower() not in ["nan", "none", "null"]
#         else x
#     )
# else:
#     raise ValueError("未找到字段 CAMEO Code，请检查 ICEWS 文件表头")

# 输出标准 CSV
icews_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("ICEWS 数据准备完成，仅保留 2023-01 至 2023-04 数据")
print("输出文件：", OUTPUT_FILE)
print("数据量：", len(icews_df))
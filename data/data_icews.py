import os
import glob
import pandas as pd

ICEWS_DIR = "./raw/icews"
OUTPUT_CSV = "./raw/icews/icews_all_events.csv"

# 递归查找所有 .tab 和 .tab.zip 文件
tab_files = []
tab_files.extend(glob.glob(os.path.join(ICEWS_DIR, "**", "*.tab"), recursive=True))
tab_files.extend(glob.glob(os.path.join(ICEWS_DIR, "**", "*.tab.zip"), recursive=True))

# 排除已经合并/整理过的目录，避免重复读取
tab_files = [
    f for f in tab_files
    if "all_tabs" not in os.path.normpath(f).split(os.sep)
]

tab_files = sorted(tab_files)

print(f"找到 {len(tab_files)} 个 ICEWS 文件")

if not tab_files:
    raise FileNotFoundError("没有找到 .tab 或 .tab.zip 文件，请检查 ICEWS_DIR 路径")

df_list = []

for file_path in tab_files:
    print("读取：", file_path)

    if file_path.endswith(".zip"):
        df = pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,              # 关键：所有字段按字符串读取，保留 010
            compression="zip",
            low_memory=False
        )
    else:
        df = pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,              # 关键：所有字段按字符串读取，保留 010
            low_memory=False
        )

    # 记录来源文件，方便之后排查数据来源
    df["Source File"] = os.path.basename(file_path)

    df_list.append(df)

icews_df = pd.concat(df_list, ignore_index=True)

print("合并前总行数：", len(icews_df))

# 去掉完全重复的行
icews_df = icews_df.drop_duplicates()

print("去重后总行数：", len(icews_df))

# # 再次确保 CAMEO Code 是字符串，并保留/补齐前导 0
# if "CAMEO Code" in icews_df.columns:
#     icews_df["CAMEO Code"] = (
#         icews_df["CAMEO Code"]
#         .astype(str)
#         .str.strip()
#         .str.replace(r"\.0$", "", regex=True)
#     )

#     # 对长度小于 3 的 CAMEO Code 补齐到 3 位
#     # 例如：10 -> 010, 51 -> 051, 46 -> 046
#     icews_df["CAMEO Code"] = icews_df["CAMEO Code"].apply(
#         lambda x: x.zfill(3) if x and x.lower() not in ["nan", "none", "null"] else x
#     )
# else:
#     raise ValueError("没有找到字段 CAMEO Code，请检查原始 ICEWS 文件表头")

# 保存为 CSV
icews_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("合并完成：", OUTPUT_CSV)
print("最终行数：", len(icews_df))
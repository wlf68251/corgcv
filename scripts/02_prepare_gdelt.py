# scripts/02_prepare_gdelt.py
import os
import pandas as pd
import zipfile
import glob

RAW_DIR = "../data/raw/gdelt"
OUTPUT_CSV = os.path.join(RAW_DIR, "gdelt_raw_events.csv")

columns = [
    "GLOBALEVENTID","SQLDATE","MonthYear","Year","FractionDate",
    "Actor1Code","Actor1Name","Actor1CountryCode","Actor1KnownGroupCode","Actor1EthnicCode","Actor1Religion1Code","Actor1Religion2Code","Actor1Type1Code","Actor1Type2Code","Actor1Type3Code",
    "Actor2Code","Actor2Name","Actor2CountryCode","Actor2KnownGroupCode","Actor2EthnicCode","Actor2Religion1Code","Actor2Religion2Code","Actor2Type1Code","Actor2Type2Code","Actor2Type3Code",
    "IsRootEvent","EventCode","EventBaseCode","EventRootCode","QuadClass","GoldsteinScale","NumMentions","NumSources","NumArticles","AvgTone",
    "Actor1Geo_Type","Actor1Geo_FullName","Actor1Geo_CountryCode","Actor1Geo_ADM1Code","Actor1Geo_Lat","Actor1Geo_Long",
    "Actor2Geo_Type","Actor2Geo_FullName","Actor2Geo_CountryCode","Actor2Geo_ADM1Code","Actor2Geo_Lat","Actor2Geo_Long",
    "ActionGeo_Type","ActionGeo_FullName","ActionGeo_CountryCode","ActionGeo_ADM1Code","ActionGeo_Lat","ActionGeo_Long",
    "DATEADDED","SOURCEURL"
]

zip_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.zip")) + glob.glob(os.path.join(RAW_DIR, "*.CSV.zip")))

first = True  # 第一次写入时写列名
for zf in zip_files:
    try:
        with zipfile.ZipFile(zf) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                print(f"Processing {zf} ...")
                df = pd.read_csv(f, sep="\t", header=None, low_memory=False)
                if df.shape[1] >= len(columns):
                    df = df.iloc[:, :len(columns)]
                    df.columns = columns
                else:
                    df.columns = [f"col{i}" for i in range(df.shape[1])]

                # 日期筛选
                df["SQLDATE"] = pd.to_numeric(df["SQLDATE"], errors='coerce')
                df = df[(df["SQLDATE"] >= 20230101) & (df["SQLDATE"] <= 20230430)]

                # 追加写入 CSV
                df.to_csv(OUTPUT_CSV, mode='w' if first else 'a', index=False, header=first)
                first = False
    except Exception as e:
        print(f"Failed to process {zf}: {e}")

print("GDELT 数据整理完成，输出", OUTPUT_CSV)
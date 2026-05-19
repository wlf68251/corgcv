# make_icews_level1_simple.py

INPUT_TSV = "icews_raw_events.tsv"
OUTPUT_TSV = "icews_raw_e_events.tsv"

with open(INPUT_TSV, "r", encoding="utf-8") as fin, \
     open(OUTPUT_TSV, "w", encoding="utf-8") as fout:

    for line in fin:
        parts = line.rstrip("\n").split("\t")

        # PaTeCon 输入应为 5 列：
        # subject, property, object, start_time, end_time
        if len(parts) != 5:
            continue

        # property 字段是第 2 列，下标为 1
        # CAMEO_010 -> CAMEO_01
        # CAMEO_100 -> CAMEO_10
        parts[1] = parts[1][:8]

        fout.write("\t".join(parts) + "\n")

print("生成完成：", OUTPUT_TSV)
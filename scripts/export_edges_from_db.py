# scripts/export_edges_from_db.py
import os
import sqlite3
import pandas as pd

PROCESSED_DIR = "../data/processed"

SQLITE_DB = os.path.join(PROCESSED_DIR, "relation_edges_work.db")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "relation_edges_before_check.csv")

def main():
    if not os.path.exists(SQLITE_DB):
        print(f"未找到数据库文件: {SQLITE_DB}")
        return

    conn = sqlite3.connect(SQLITE_DB)

    # 先检查有哪些表
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn
    )
    print("当前数据库中的表:")
    print(tables)

    if "relation_edges" not in tables["name"].tolist():
        print("数据库中没有 relation_edges 表")
        conn.close()
        return

    # 检查数量
    count_df = pd.read_sql_query(
        "SELECT COUNT(*) AS cnt FROM relation_edges;",
        conn
    )
    count = int(count_df.loc[0, "cnt"])
    print("relation_edges 表中记录数:", count)

    if count == 0:
        print("relation_edges 表为空，无法导出。")
        conn.close()
        return

    query = """
    SELECT
        edge_id,
        subject_org_id,
        object_org_id,
        event_month,
        relation_type,
        subject_name,
        object_name,
        event_count,
        source_datasets,
        icews_event_count,
        gdelt_event_count,
        confidence,
        status
    FROM relation_edges
    """

    first = True

    # 防止旧空文件影响
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    for chunk in pd.read_sql_query(query, conn, chunksize=100000):
        chunk.to_csv(
            OUTPUT_CSV,
            mode="w" if first else "a",
            index=False,
            header=first,
            encoding="utf-8-sig"
        )
        first = False

    conn.close()

    print("导出完成:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
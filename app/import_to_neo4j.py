# app/import_to_neo4j.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
    将 data/processed/relation_edges_after_check.csv 导入 Neo4j。
    不创建任何 UI。
    不使用 Streamlit。
    不展示图谱。
    图谱展示直接使用 Neo4j Browser 默认 Graph 视图。

运行：
    python app/import_to_neo4j.py
"""

import os
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase


# ============================================================
# 路径配置
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

CSV_FILE = PROJECT_DIR / "data" / "processed" / "relation_edges_after_check.csv"


# ============================================================
# Neo4j 配置
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# 第一次导入建议 True。
# 如果你不想清空旧图，改成 False。
CLEAR_OLD_GRAPH = True

BATCH_SIZE = 1000


# ============================================================
# 工具函数
# ============================================================

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_int(x, default=0):
    try:
        if pd.isna(x) or str(x).strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if pd.isna(x) or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_bool_text(x):
    """
    保留原始 True/False 文本，不强制转 bool，避免 Neo4j 类型兼容问题。
    """
    return safe_str(x)


# ============================================================
# Neo4j 操作
# ============================================================

def create_indexes(driver):
    cyphers = [
        """
        CREATE CONSTRAINT organization_id_unique IF NOT EXISTS
        FOR (o:Organization)
        REQUIRE o.org_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT relation_edge_id_unique IF NOT EXISTS
        FOR ()-[r:RELATION]-()
        REQUIRE r.edge_id IS UNIQUE
        """,
        """
        CREATE INDEX organization_name_index IF NOT EXISTS
        FOR (o:Organization)
        ON (o.name)
        """,
        """
        CREATE INDEX relation_type_index IF NOT EXISTS
        FOR ()-[r:RELATION]-()
        ON (r.relation_type)
        """,
        """
        CREATE INDEX relation_status_index IF NOT EXISTS
        FOR ()-[r:RELATION]-()
        ON (r.status)
        """,
        """
        CREATE INDEX relation_event_month_index IF NOT EXISTS
        FOR ()-[r:RELATION]-()
        ON (r.event_month)
        """
    ]

    with driver.session() as session:
        for cypher in cyphers:
            session.run(cypher)


def clear_graph(driver):
    with driver.session() as session:
        print("[RUN] 清空旧 RELATION 关系...")
        session.run("MATCH ()-[r:RELATION]->() DELETE r")

        print("[RUN] 清空旧 Organization 节点...")
        session.run("MATCH (o:Organization) DELETE o")


def import_batch(tx, rows):
    cypher = """
    UNWIND $rows AS row

    MERGE (s:Organization {org_id: row.subject_org_id})
    SET
        s.name = row.subject_name

    MERGE (o:Organization {org_id: row.object_org_id})
    SET
        o.name = row.object_name

    MERGE (s)-[r:RELATION {edge_id: row.edge_id}]->(o)
    SET
        r.event_month = row.event_month,
        r.relation_type = row.relation_type,
        r.event_count = row.event_count,
        r.source_datasets = row.source_datasets,
        r.icews_event_count = row.icews_event_count,
        r.gdelt_event_count = row.gdelt_event_count,
        r.confidence = row.confidence,
        r.status = row.status,
        r.confidence_level = row.confidence_level,
        r.confidence_reason = row.confidence_reason,
        r.is_derived_relation = row.is_derived_relation,
        r.check_action = row.check_action,
        r.check_reason = row.check_reason,
        r.triggered_constraints = row.triggered_constraints
    """
    tx.run(cypher, rows=rows)


# ============================================================
# CSV 读取与转换
# ============================================================

def load_edges():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到文件: {CSV_FILE}")

    print("[RUN] 读取 CSV:", CSV_FILE)

    df = pd.read_csv(
        CSV_FILE,
        dtype=str,
        low_memory=False,
    ).fillna("")

    print("[STAT] CSV rows:", len(df))
    print("[STAT] CSV columns:", list(df.columns))

    required_cols = [
        "edge_id",
        "subject_org_id",
        "object_org_id",
        "event_month",
        "relation_type",
        "subject_name",
        "object_name",
        "event_count",
        "source_datasets",
        "icews_event_count",
        "gdelt_event_count",
        "confidence",
        "status",
        "confidence_level",
        "confidence_reason",
        "is_derived_relation",
        "check_action",
        "check_reason",
        "triggered_constraints",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            "relation_edges_after_check.csv 缺少必要字段:\n"
            f"{missing}\n"
            f"当前字段:\n{list(df.columns)}"
        )

    rows = []

    for _, row in df.iterrows():
        subject_org_id = safe_str(row["subject_org_id"])
        object_org_id = safe_str(row["object_org_id"])
        subject_name = safe_str(row["subject_name"])
        object_name = safe_str(row["object_name"])
        edge_id = safe_str(row["edge_id"])

        if not edge_id:
            continue

        if not subject_org_id or not object_org_id:
            continue

        if not subject_name or not object_name:
            continue

        rows.append({
            "edge_id": edge_id,
            "subject_org_id": subject_org_id,
            "object_org_id": object_org_id,
            "event_month": safe_str(row["event_month"]),
            "relation_type": safe_str(row["relation_type"]),
            "subject_name": subject_name,
            "object_name": object_name,
            "event_count": safe_int(row["event_count"]),
            "source_datasets": safe_str(row["source_datasets"]),
            "icews_event_count": safe_int(row["icews_event_count"]),
            "gdelt_event_count": safe_int(row["gdelt_event_count"]),
            "confidence": safe_float(row["confidence"]),
            "status": safe_str(row["status"]),
            "confidence_level": safe_str(row["confidence_level"]),
            "confidence_reason": safe_str(row["confidence_reason"]),
            "is_derived_relation": safe_bool_text(row["is_derived_relation"]),
            "check_action": safe_str(row["check_action"]),
            "check_reason": safe_str(row["check_reason"]),
            "triggered_constraints": safe_str(row["triggered_constraints"]),
        })

    print("[STAT] valid import rows:", len(rows))

    return rows


# ============================================================
# 主流程
# ============================================================

def main():
    rows = load_edges()

    if not rows:
        print("[WARN] 没有可导入的数据。")
        return

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    print("[RUN] 连接 Neo4j:", NEO4J_URI)

    create_indexes(driver)

    if CLEAR_OLD_GRAPH:
        clear_graph(driver)

    total = len(rows)

    with driver.session() as session:
        for start in range(0, total, BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            session.execute_write(import_batch, batch)

            done = min(start + BATCH_SIZE, total)
            print(f"[RUN] imported {done} / {total}")

    driver.close()

    print("[DONE] 导入 Neo4j 完成。")
    print()
    print("现在打开 Neo4j Browser:")
    print("http://localhost:7474")
    print()
    print("执行:")
    print("MATCH p=(a:Organization)-[r:RELATION]->(b:Organization) RETURN p LIMIT 100;")


if __name__ == "__main__":
    main()
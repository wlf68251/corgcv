# app/neo4j_graph_app.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
neo4j_graph_app.py

功能：
    基于 Neo4j 展示组织机构复杂关系知识图谱。
    不提供 Cypher 输入，只提供正常图谱展示与基础筛选。

运行：
    streamlit run app/neo4j_graph_app.py
"""

import os
import json
import tempfile

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="组织机构复杂关系知识图谱展示",
    layout="wide"
)


# ============================================================
# Neo4j 配置
# ============================================================

DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = ""

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RELATION_EDGES_AFTER_FILE = os.path.join(DATA_DIR, "relation_edges_after_check.csv")

# ============================================================
# 工具函数
# ============================================================

def read_relation_edges_csv():
    if not os.path.exists(RELATION_EDGES_AFTER_FILE):
        st.error(f"未找到文件：{RELATION_EDGES_AFTER_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(
        RELATION_EDGES_AFTER_FILE,
        dtype=str,
        low_memory=False,
    ).fillna("")

    return df


def clear_neo4j_graph(driver):
    with driver.session() as session:
        session.run("MATCH ()-[r]->() DELETE r")
        session.run("MATCH (n:Organization) DELETE n")


def create_neo4j_indexes(driver):
    cyphers = [
        """
        CREATE CONSTRAINT organization_id_unique IF NOT EXISTS
        FOR (o:Organization)
        REQUIRE o.org_id IS UNIQUE
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


def import_edges_to_neo4j(driver, df, clear_old=True, batch_size=1000):
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
        st.error(f"relation_edges_after_check.csv 缺少字段：{missing}")
        st.write("当前字段：", list(df.columns))
        return

    create_neo4j_indexes(driver)

    if clear_old:
        clear_neo4j_graph(driver)

    rows = df[required_cols].to_dict("records")

    cypher = """
    UNWIND $rows AS row

    MERGE (s:Organization {org_id: row.subject_org_id})
    SET
        s.name = row.subject_name,
        s.display_name = row.subject_name

    MERGE (o:Organization {org_id: row.object_org_id})
    SET
        o.name = row.object_name,
        o.display_name = row.object_name

    MERGE (s)-[r:RELATION {edge_id: row.edge_id}]->(o)
    SET
        r.event_month = row.event_month,
        r.month = row.event_month,
        r.relation_type = row.relation_type,
        r.event_count = toInteger(row.event_count),
        r.source_datasets = row.source_datasets,
        r.icews_event_count = toInteger(row.icews_event_count),
        r.gdelt_event_count = toInteger(row.gdelt_event_count),
        r.confidence = toFloat(row.confidence),
        r.status = row.status,
        r.confidence_level = row.confidence_level,
        r.confidence_reason = row.confidence_reason,
        r.is_derived_relation = row.is_derived_relation,
        r.check_action = row.check_action,
        r.check_reason = row.check_reason,
        r.triggered_constraints = row.triggered_constraints
    """

    total = len(rows)

    progress = st.progress(0)
    info = st.empty()

    with driver.session() as session:
        for start in range(0, total, batch_size):
            batch = rows[start:start + batch_size]
            session.run(cypher, rows=batch)

            done = min(start + batch_size, total)
            progress.progress(done / total)
            info.write(f"已导入 {done} / {total} 条关系")

    st.success(f"Neo4j 导入完成，共导入 {total} 条关系。")

def safe_str(x):
    if x is None:
        return ""
    return str(x).strip()


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def relation_color(relation_type):
    relation_type = str(relation_type)

    color_map = {
        # 如果你之前是五类关系
        "verbal_cooperation": "#4CAF50",
        "material_cooperation": "#2196F3",
        "verbal_conflict": "#FF9800",
        "material_conflict": "#F44336",
        "mixed_relation": "#9C27B0",

        # 如果你现在是 CAMEO 顶层码 01-20
        "01": "#8BC34A",
        "02": "#7CB342",
        "03": "#689F38",
        "04": "#558B2F",
        "05": "#33691E",

        "06": "#03A9F4",
        "07": "#0288D1",
        "08": "#01579B",

        "09": "#FFB300",
        "10": "#FB8C00",
        "11": "#F57C00",
        "12": "#EF6C00",
        "13": "#E65100",

        "14": "#E53935",
        "15": "#D32F2F",
        "16": "#C62828",
        "17": "#B71C1C",
        "18": "#880E4F",
        "19": "#6A1B9A",
        "20": "#4A148C",
    }

    return color_map.get(relation_type, "#9E9E9E")


def status_color(status):
    status = str(status)

    if status == "active":
        return "#4CAF50"
    if status == "hidden":
        return "#9E9E9E"
    if status == "removed":
        return "#F44336"

    return "#607D8B"


@st.cache_resource
def get_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))


def test_connection(driver):
    with driver.session() as session:
        result = session.run("RETURN 1 AS ok")
        row = result.single()
        return row["ok"] == 1


def get_graph_stats(driver):
    cypher = """
    MATCH (o:Organization)
    WITH count(o) AS org_count
    MATCH ()-[r]->()
    RETURN org_count, count(r) AS relation_count
    """

    try:
        with driver.session() as session:
            row = session.run(cypher).single()
            if row:
                return {
                    "org_count": row["org_count"],
                    "relation_count": row["relation_count"],
                }
    except Exception:
        pass

    return {
        "org_count": 0,
        "relation_count": 0,
    }


def get_relation_type_options(driver):
    cypher = """
    MATCH ()-[r]->()
    WHERE r.relation_type IS NOT NULL
    RETURN DISTINCT r.relation_type AS relation_type
    ORDER BY relation_type
    """

    values = []

    try:
        with driver.session() as session:
            for row in session.run(cypher):
                if row["relation_type"] is not None:
                    values.append(str(row["relation_type"]))
    except Exception:
        pass

    return values


def get_status_options(driver):
    cypher = """
    MATCH ()-[r]->()
    WHERE r.status IS NOT NULL
    RETURN DISTINCT r.status AS status
    ORDER BY status
    """

    values = []

    try:
        with driver.session() as session:
            for row in session.run(cypher):
                if row["status"] is not None:
                    values.append(str(row["status"]))
    except Exception:
        pass

    return values


def query_graph(
    driver,
    keyword="",
    relation_types=None,
    statuses=None,
    min_confidence=0.0,
    max_edges=300,
):
    relation_types = relation_types or []
    statuses = statuses or []

    where_parts = []

    params = {
        "limit": max_edges,
        "min_confidence": min_confidence,
    }

    if keyword:
        params["keyword"] = keyword.upper()
        where_parts.append(
            """
            (
                toUpper(a.name) CONTAINS $keyword
                OR toUpper(b.name) CONTAINS $keyword
                OR toUpper(a.org_id) CONTAINS $keyword
                OR toUpper(b.org_id) CONTAINS $keyword
            )
            """
        )

    if relation_types:
        params["relation_types"] = relation_types
        where_parts.append("r.relation_type IN $relation_types")

    if statuses:
        params["statuses"] = statuses
        where_parts.append("r.status IN $statuses")

    where_parts.append(
        """
        (
            r.confidence IS NULL
            OR toFloat(r.confidence) >= $min_confidence
        )
        """
    )

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + "\nAND ".join(where_parts)

    cypher = f"""
    MATCH (a:Organization)-[r]->(b:Organization)
    {where_clause}
    RETURN a, r, b
    ORDER BY
        coalesce(toInteger(r.event_count), 0) DESC,
        coalesce(toFloat(r.confidence), 0.0) DESC
    LIMIT $limit
    """

    nodes = {}
    edges = []

    with driver.session() as session:
        result = session.run(cypher, params)

        for row in result:
            a = row["a"]
            b = row["b"]
            r = row["r"]

            a_props = dict(a)
            b_props = dict(b)
            r_props = dict(r)

            a_id = safe_str(a_props.get("org_id")) or str(a.element_id)
            b_id = safe_str(b_props.get("org_id")) or str(b.element_id)

            a_name = (
                safe_str(a_props.get("name"))
                or safe_str(a_props.get("canonical_name"))
                or a_id
            )

            b_name = (
                safe_str(b_props.get("name"))
                or safe_str(b_props.get("canonical_name"))
                or b_id
            )

            nodes[a_id] = {
                "id": a_id,
                "name": a_name,
                "properties": a_props,
            }

            nodes[b_id] = {
                "id": b_id,
                "name": b_name,
                "properties": b_props,
            }

            relation_type = safe_str(r_props.get("relation_type")) or r.type
            status = safe_str(r_props.get("status"))
            confidence = safe_float(r_props.get("confidence"), 0.0)
            event_count = safe_int(r_props.get("event_count"), 0)

            edges.append({
                "id": safe_str(r_props.get("edge_id")) or str(r.element_id),
                "source": a_id,
                "target": b_id,
                "source_name": a_name,
                "target_name": b_name,
                "relation_type": relation_type,
                "status": status,
                "confidence": confidence,
                "event_count": event_count,
                "month": safe_str(
                            r_props.get("event_month")
                            or r_props.get("month")
                        ),
                "start_date": safe_str(r_props.get("start_date") or r_props.get("start_time")),
                "end_date": safe_str(r_props.get("end_date") or r_props.get("end_time")),
                "properties": r_props,
            })

    return list(nodes.values()), edges


def render_graph(nodes, edges):
    net = Network(
        height="760px",
        width="100%",
        directed=True,
        notebook=False,
    )

    net.barnes_hut(
        gravity=-30000,
        central_gravity=0.3,
        spring_length=180,
        spring_strength=0.02,
        damping=0.09,
    )

    degree = {}

    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    for node in nodes:
        node_id = node["id"]
        name = node["name"]
        d = degree.get(node_id, 1)
        size = min(12 + d * 2, 45)

        title = json.dumps(
            node.get("properties", {}),
            ensure_ascii=False,
            indent=2,
        )

        net.add_node(
            node_id,
            label=name[:30],
            title=title,
            size=size,
        )

    for edge in edges:
        relation_type = edge["relation_type"]
        event_count = edge["event_count"]

        title = (
            f"{edge['source_name']} → {edge['target_name']}<br>"
            f"关系类型: {edge['relation_type']}<br>"
            f"状态: {edge['status']}<br>"
            f"置信度: {edge['confidence']}<br>"
            f"事件数: {edge['event_count']}<br>"
            f"月份: {edge['month']}<br>"
            f"开始时间: {edge['start_date']}<br>"
            f"结束时间: {edge['end_date']}"
        )

        width = max(1, min(8, event_count ** 0.5 if event_count > 0 else 1))

        net.add_edge(
            edge["source"],
            edge["target"],
            label=relation_type,
            title=title,
            color=relation_color(relation_type),
            width=width,
            arrows="to",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name

    net.save_graph(path)

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    st.components.v1.html(html, height=800, scrolling=True)

    try:
        os.remove(path)
    except Exception:
        pass


def edges_to_dataframe(edges):
    rows = []

    for e in edges:
        rows.append({
            "edge_id": e["id"],
            "subject": e["source_name"],
            "relation_type": e["relation_type"],
            "object": e["target_name"],
            "month": e["month"],
            "event_count": e["event_count"],
            "confidence": e["confidence"],
            "status": e["status"],
            "start_date": e["start_date"],
            "end_date": e["end_date"],
        })

    return pd.DataFrame(rows)


# ============================================================
# 主页面
# ============================================================

def main():
    st.title("组织机构复杂关系知识图谱展示")
    st.caption("数据来源：Neo4j 图数据库。页面自动展示局部知识图谱，不需要手动输入 Cypher。")

    with st.sidebar:
        st.header("Neo4j 连接")

        uri = st.text_input(
            "Neo4j URI",
            value=os.getenv("NEO4J_URI", DEFAULT_NEO4J_URI),
        )

        user = st.text_input(
            "用户名",
            value=os.getenv("NEO4J_USER", DEFAULT_NEO4J_USER),
        )

        password = st.text_input(
            "密码",
            value=os.getenv("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD),
            type="password",
        )

    if not password:
        st.warning("请在侧边栏输入 Neo4j 密码，或设置环境变量 NEO4J_PASSWORD。")
        return

    try:
        driver = get_driver(uri, user, password)
        test_connection(driver)
    except Exception as e:
        st.error(f"Neo4j 连接失败：{e}")
        return

    with st.sidebar:
        st.header("数据导入")

        clear_old = st.checkbox("导入前清空旧 Neo4j 图谱", value=True)

        if st.button("从 relation_edges_after_check.csv 导入 Neo4j"):
            df_csv = read_relation_edges_csv()

            if not df_csv.empty:
                st.write("CSV 字段：", list(df_csv.columns))
                st.write("CSV 行数：", len(df_csv))

                import_edges_to_neo4j(
                    driver=driver,
                    df=df_csv,
                    clear_old=clear_old,
                    batch_size=1000,
                )

    stats = get_graph_stats(driver)

    col1, col2 = st.columns(2)
    col1.metric("Neo4j 组织节点数", stats["org_count"])
    col2.metric("Neo4j 关系边数", stats["relation_count"])

    st.divider()

    relation_type_options = get_relation_type_options(driver)
    status_options = get_status_options(driver)

    with st.sidebar:
        st.header("图谱筛选")

        keyword = st.text_input(
            "组织关键词",
            value="",
            placeholder="例如 CHINA / UNITED NATIONS / NATO",
        )

        selected_statuses = st.multiselect(
            "关系状态",
            status_options,
            default=["active"] if "active" in status_options else status_options[:1],
        )

        selected_relation_types = st.multiselect(
            "关系类型",
            relation_type_options,
            default=[],
            help="不选择则不过滤关系类型。",
        )

        min_confidence = st.slider(
            "最低置信度",
            0.0,
            1.0,
            0.0,
            0.05,
        )

        max_edges = st.slider(
            "最多展示关系边数",
            50,
            1000,
            300,
            50,
        )

        refresh = st.button("刷新图谱", type="primary")

    nodes, edges = query_graph(
        driver=driver,
        keyword=keyword,
        relation_types=selected_relation_types,
        statuses=selected_statuses,
        min_confidence=min_confidence,
        max_edges=max_edges,
    )

    st.write(f"当前展示节点数：{len(nodes)}，关系边数：{len(edges)}")

    if not edges:
        st.info("当前筛选条件下没有可展示的关系。可以减少筛选条件或增大展示边数。")
        return

    render_graph(nodes, edges)

    st.divider()

    st.subheader("当前展示关系数据")
    edge_df = edges_to_dataframe(edges)
    st.dataframe(edge_df, use_container_width=True, height=420)


if __name__ == "__main__":
    main()
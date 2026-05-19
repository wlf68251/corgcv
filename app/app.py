# app/app.py
import os
import math
import tempfile
import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network


# =========================
# 路径配置
# =========================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

ORGANIZATIONS_FILE = os.path.join(DATA_DIR, "organizations.csv")
ALIASES_FILE = os.path.join(DATA_DIR, "organization_aliases.csv")

EVENT_FACTS_FILE = os.path.join(DATA_DIR, "event_facts.csv")
SOURCE_EVIDENCE_FILE = os.path.join(DATA_DIR, "source_evidence.csv")

RELATION_EDGES_BEFORE_FILE = os.path.join(DATA_DIR, "relation_edges_before_check.csv")
RELATION_EDGES_SCORED_FILE = os.path.join(DATA_DIR, "relation_edges_scored.csv")
RELATION_EDGES_AFTER_FILE = os.path.join(DATA_DIR, "relation_edges_after_check.csv")

TEMPORAL_CONSTRAINTS_FINAL_FILE = os.path.join(DATA_DIR, "temporal_constraints_final.csv")
DETECTED_CONFLICTS_FILE = os.path.join(DATA_DIR, "detected_conflicts.csv")

EVALUATION_SUMMARY_FILE = os.path.join(
    OUTPUTS_DIR,
    "evaluation_tables",
    "evaluation_summary.csv"
)


# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="组织机构复杂关系图谱可视化",
    layout="wide"
)


# =========================
# 工具函数
# =========================

@st.cache_data
def read_csv_if_exists(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        st.warning(f"读取文件失败: {path}\n错误: {e}")
        return pd.DataFrame()


@st.cache_data
def load_all_data():
    data = {
        "organizations": read_csv_if_exists(ORGANIZATIONS_FILE),
        "aliases": read_csv_if_exists(ALIASES_FILE),
        "event_facts": read_csv_if_exists(EVENT_FACTS_FILE),
        "source_evidence": read_csv_if_exists(SOURCE_EVIDENCE_FILE),
        "edges_before": read_csv_if_exists(RELATION_EDGES_BEFORE_FILE),
        "edges_scored": read_csv_if_exists(RELATION_EDGES_SCORED_FILE),
        "edges_after": read_csv_if_exists(RELATION_EDGES_AFTER_FILE),
        "constraints": read_csv_if_exists(TEMPORAL_CONSTRAINTS_FINAL_FILE),
        "conflicts": read_csv_if_exists(DETECTED_CONFLICTS_FILE),
        "evaluation_summary": read_csv_if_exists(EVALUATION_SUMMARY_FILE),
    }
    return data


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


def get_org_name_map(orgs):
    if orgs.empty or "org_id" not in orgs.columns:
        return {}

    name_col = "canonical_name" if "canonical_name" in orgs.columns else "raw_name"

    if name_col not in orgs.columns:
        return {str(r["org_id"]): str(r["org_id"]) for _, r in orgs.iterrows()}

    return {
        str(r["org_id"]): str(r[name_col])
        for _, r in orgs.iterrows()
    }


def add_org_names_to_edges(edges, org_name_map):
    if edges.empty:
        return edges

    edges = edges.copy()

    if "subject_org_id" in edges.columns:
        edges["subject_display"] = edges["subject_org_id"].astype(str).map(org_name_map)
        edges["subject_display"] = edges["subject_display"].fillna(edges["subject_org_id"].astype(str))

    if "object_org_id" in edges.columns:
        edges["object_display"] = edges["object_org_id"].astype(str).map(org_name_map)
        edges["object_display"] = edges["object_display"].fillna(edges["object_org_id"].astype(str))

    return edges


def relation_color(relation_type):
    relation_type = str(relation_type)

    color_map = {
        "verbal_cooperation": "#4CAF50",
        "material_cooperation": "#2196F3",
        "verbal_conflict": "#FF9800",
        "material_conflict": "#F44336",
        "mixed_relation": "#9C27B0"
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


def show_dataframe(df, height=400):
    if df.empty:
        st.info("暂无数据。")
    else:
        st.dataframe(df, use_container_width=True, height=height)


def metric_value(df, key, fallback=0):
    if df.empty:
        return fallback

    if "metric" not in df.columns or "value" not in df.columns:
        return fallback

    row = df[df["metric"] == key]

    if row.empty:
        return fallback

    return row.iloc[0]["value"]


# =========================
# 总览页
# =========================

def page_dashboard(data):
    st.title("组织机构复杂关系图谱总览")

    orgs = data["organizations"]
    edges_before = data["edges_before"]
    edges_after = data["edges_after"]
    conflicts = data["conflicts"]
    constraints = data["constraints"]
    summary = data["evaluation_summary"]

    col1, col2, col3, col4 = st.columns(4)

    org_count = len(orgs)
    before_count = len(edges_before)
    after_count = len(edges_after)
    conflict_count = len(conflicts)

    if not summary.empty:
        org_count = metric_value(summary, "organization_count", org_count)
        before_count = metric_value(summary, "edges_before_check", before_count)
        after_count = metric_value(summary, "edges_after_check", after_count)
        conflict_count = metric_value(summary, "detected_conflicts", conflict_count)

    col1.metric("组织节点数", org_count)
    col2.metric("校验前关系数", before_count)
    col3.metric("校验后关系数", after_count)
    col4.metric("冲突/复核记录数", conflict_count)

    st.divider()

    col5, col6, col7, col8 = st.columns(4)

    active_edges = 0
    hidden_edges = 0
    mixed_edges = 0
    cross_source_edges = 0

    if not edges_after.empty:
        if "status" in edges_after.columns:
            active_edges = len(edges_after[edges_after["status"] == "active"])
            hidden_edges = len(edges_after[edges_after["status"] == "hidden"])

        if "relation_type" in edges_after.columns:
            mixed_edges = len(edges_after[edges_after["relation_type"] == "mixed_relation"])

    if not edges_before.empty and "source_datasets" in edges_before.columns:
        s = edges_before["source_datasets"].fillna("").astype(str)
        cross_source_edges = len(edges_before[s.str.contains("ICEWS") & s.str.contains("GDELT")])

    col5.metric("Active 关系", active_edges)
    col6.metric("Hidden 关系", hidden_edges)
    col7.metric("Mixed 关系", mixed_edges)
    col8.metric("跨源共同支持关系", cross_source_edges)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("关系状态分布")
        if not edges_after.empty and "status" in edges_after.columns:
            status_df = edges_after["status"].fillna("unknown").value_counts().reset_index()
            status_df.columns = ["status", "count"]
            st.bar_chart(status_df.set_index("status"))
            show_dataframe(status_df, height=200)
        else:
            st.info("缺少 status 字段。")

    with right:
        st.subheader("关系类型分布")
        if not edges_after.empty and "relation_type" in edges_after.columns:
            relation_df = edges_after["relation_type"].fillna("unknown").value_counts().reset_index()
            relation_df.columns = ["relation_type", "count"]
            st.bar_chart(relation_df.set_index("relation_type"))
            show_dataframe(relation_df, height=200)
        else:
            st.info("缺少 relation_type 字段。")

    st.divider()

    st.subheader("最终约束概览")
    show_dataframe(constraints, height=300)


# =========================
# 时间线图谱页
# =========================

def build_pyvis_graph(edges, org_name_map, max_edges=300):
    net = Network(
        height="720px",
        width="100%",
        directed=True,
        notebook=False
    )

    net.barnes_hut(
        gravity=-30000,
        central_gravity=0.3,
        spring_length=180,
        spring_strength=0.02,
        damping=0.09
    )

    if edges.empty:
        return net

    edges = edges.head(max_edges).copy()

    node_weight = {}

    for _, row in edges.iterrows():
        s = str(row.get("subject_org_id", ""))
        o = str(row.get("object_org_id", ""))

        node_weight[s] = node_weight.get(s, 0) + 1
        node_weight[o] = node_weight.get(o, 0) + 1

    for node_id, weight in node_weight.items():
        label = org_name_map.get(node_id, node_id)
        size = min(10 + weight * 2, 45)

        net.add_node(
            node_id,
            label=label[:30],
            title=f"{label}<br>{node_id}<br>度数: {weight}",
            size=size
        )

    for _, row in edges.iterrows():
        s = str(row.get("subject_org_id", ""))
        o = str(row.get("object_org_id", ""))

        relation_type = str(row.get("relation_type", ""))
        confidence = safe_float(row.get("confidence", 0.0))
        event_count = safe_int(row.get("event_count", 0))
        status = str(row.get("status", ""))

        title = (
            f"关系类型: {relation_type}<br>"
            f"置信度: {confidence}<br>"
            f"事件数: {event_count}<br>"
            f"状态: {status}<br>"
            f"来源: {row.get('source_datasets', '')}"
        )

        width = max(1, min(8, math.log1p(event_count) + 1))

        net.add_edge(
            s,
            o,
            label=relation_type,
            title=title,
            color=relation_color(relation_type),
            width=width
        )

    return net


def render_pyvis(net):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name

    net.save_graph(path)

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    st.components.v1.html(html, height=760, scrolling=True)

    try:
        os.remove(path)
    except Exception:
        pass


def page_graph_timeline(data):
    st.title("时间线图谱")

    org_name_map = get_org_name_map(data["organizations"])
    edges = data["edges_after"]

    if edges.empty:
        st.info("未找到 relation_edges_after_check.csv，请先运行 11_update_graph.py。")
        return

    edges = add_org_names_to_edges(edges, org_name_map)

    st.sidebar.subheader("图谱筛选")

    months = sorted(edges["event_month"].dropna().astype(str).unique().tolist()) if "event_month" in edges.columns else []
    relation_types = sorted(edges["relation_type"].dropna().astype(str).unique().tolist()) if "relation_type" in edges.columns else []
    statuses = sorted(edges["status"].dropna().astype(str).unique().tolist()) if "status" in edges.columns else []

    selected_month = st.sidebar.selectbox("选择月份", months) if months else None
    selected_relations = st.sidebar.multiselect("关系类型", relation_types, default=relation_types)
    selected_statuses = st.sidebar.multiselect("状态", statuses, default=["active"] if "active" in statuses else statuses)

    min_confidence = st.sidebar.slider("最低置信度", 0.0, 1.0, 0.0, 0.05)
    max_edges = st.sidebar.slider("图中最多显示边数", 50, 1000, 300, 50)

    filtered = edges.copy()

    if selected_month:
        filtered = filtered[filtered["event_month"].astype(str) == selected_month]

    if selected_relations:
        filtered = filtered[filtered["relation_type"].astype(str).isin(selected_relations)]

    if selected_statuses and "status" in filtered.columns:
        filtered = filtered[filtered["status"].astype(str).isin(selected_statuses)]

    if "confidence" in filtered.columns:
        filtered["confidence_num"] = filtered["confidence"].apply(safe_float)
        filtered = filtered[filtered["confidence_num"] >= min_confidence]

    if "event_count" in filtered.columns:
        filtered["event_count_num"] = filtered["event_count"].apply(safe_int)
        filtered = filtered.sort_values(
            by=["event_count_num", "confidence_num" if "confidence_num" in filtered.columns else "event_count_num"],
            ascending=False
        )

    st.write(f"当前筛选后关系边数量：{len(filtered)}")
    st.caption("图谱为了保证浏览速度，只显示筛选后排名靠前的一部分边。")

    net = build_pyvis_graph(filtered, org_name_map, max_edges=max_edges)
    render_pyvis(net)

    st.subheader("当前图谱边数据")
    display_cols = [
        c for c in [
            "edge_id",
            "subject_display",
            "relation_type",
            "object_display",
            "event_month",
            "event_count",
            "confidence",
            "status",
            "source_datasets",
            "check_action",
            "check_reason"
        ]
        if c in filtered.columns
    ]

    show_dataframe(filtered[display_cols].head(1000), height=400)


# =========================
# 节点详情页
# =========================

def page_node_detail(data):
    st.title("节点详情")

    orgs = data["organizations"]
    aliases = data["aliases"]
    edges_after = data["edges_after"]

    if orgs.empty:
        st.info("未找到 organizations.csv。")
        return

    org_name_map = get_org_name_map(orgs)

    search_text = st.text_input("输入组织名称或 org_id 搜索", "")

    candidates = orgs.copy()

    if search_text:
        text = search_text.lower()

        mask = pd.Series(False, index=candidates.index)

        for col in ["org_id", "canonical_name", "raw_name", "normalized_name", "clean_name"]:
            if col in candidates.columns:
                mask = mask | candidates[col].fillna("").astype(str).str.lower().str.contains(text, regex=False)

        candidates = candidates[mask]

    st.write(f"匹配组织数量：{len(candidates)}")

    if candidates.empty:
        return

    display_col = "canonical_name" if "canonical_name" in candidates.columns else "org_id"

    selected_label = st.selectbox(
        "选择组织",
        candidates.apply(
            lambda r: f"{r.get('org_id', '')} | {r.get(display_col, '')}",
            axis=1
        ).tolist()
    )

    selected_org_id = selected_label.split("|")[0].strip()

    selected_org = orgs[orgs["org_id"].astype(str) == selected_org_id]

    st.subheader("组织基本信息")
    show_dataframe(selected_org, height=180)

    st.subheader("组织别名")
    if not aliases.empty and "org_id" in aliases.columns:
        alias_df = aliases[aliases["org_id"].astype(str) == selected_org_id]
        show_dataframe(alias_df, height=220)
    else:
        st.info("未找到别名表。")

    st.subheader("相关关系边")

    if not edges_after.empty:
        related = edges_after[
            (edges_after["subject_org_id"].astype(str) == selected_org_id) |
            (edges_after["object_org_id"].astype(str) == selected_org_id)
        ].copy()

        related = add_org_names_to_edges(related, org_name_map)

        st.write(f"相关关系数量：{len(related)}")

        col1, col2, col3 = st.columns(3)

        if not related.empty:
            col1.metric("作为主体的关系数", len(related[related["subject_org_id"].astype(str) == selected_org_id]))
            col2.metric("作为客体的关系数", len(related[related["object_org_id"].astype(str) == selected_org_id]))
            if "status" in related.columns:
                col3.metric("隐藏关系数", len(related[related["status"] == "hidden"]))

        display_cols = [
            c for c in [
                "edge_id",
                "subject_display",
                "relation_type",
                "object_display",
                "event_month",
                "event_count",
                "confidence",
                "status",
                "source_datasets",
                "check_action",
                "check_reason"
            ]
            if c in related.columns
        ]

        show_dataframe(related[display_cols].head(1000), height=450)
    else:
        st.info("未找到关系边文件。")


# =========================
# 关系详情页
# =========================

def page_edge_detail(data):
    st.title("关系详情")

    org_name_map = get_org_name_map(data["organizations"])
    edges = data["edges_after"]
    conflicts = data["conflicts"]

    if edges.empty:
        st.info("未找到 relation_edges_after_check.csv。")
        return

    edges = add_org_names_to_edges(edges, org_name_map)

    st.sidebar.subheader("关系筛选")

    relation_types = sorted(edges["relation_type"].dropna().astype(str).unique().tolist()) if "relation_type" in edges.columns else []
    statuses = sorted(edges["status"].dropna().astype(str).unique().tolist()) if "status" in edges.columns else []

    selected_relation = st.sidebar.selectbox("关系类型", ["全部"] + relation_types)
    selected_status = st.sidebar.selectbox("状态", ["全部"] + statuses)
    keyword = st.sidebar.text_input("组织名称 / edge_id 关键词", "")

    filtered = edges.copy()

    if selected_relation != "全部":
        filtered = filtered[filtered["relation_type"].astype(str) == selected_relation]

    if selected_status != "全部":
        filtered = filtered[filtered["status"].astype(str) == selected_status]

    if keyword:
        k = keyword.lower()
        mask = pd.Series(False, index=filtered.index)

        for col in ["edge_id", "subject_org_id", "object_org_id", "subject_display", "object_display"]:
            if col in filtered.columns:
                mask = mask | filtered[col].fillna("").astype(str).str.lower().str.contains(k, regex=False)

        filtered = filtered[mask]

    st.write(f"匹配关系数量：{len(filtered)}")

    if filtered.empty:
        return

    if "confidence" in filtered.columns:
        filtered["confidence_num"] = filtered["confidence"].apply(safe_float)
        filtered = filtered.sort_values(by="confidence_num", ascending=False)

    selected_edge_label = st.selectbox(
        "选择关系边",
        filtered.apply(
            lambda r: (
                f"{r.get('edge_id', '')} | "
                f"{r.get('subject_display', r.get('subject_org_id', ''))} "
                f"-[{r.get('relation_type', '')}]-> "
                f"{r.get('object_display', r.get('object_org_id', ''))} | "
                f"{r.get('event_month', '')}"
            ),
            axis=1
        ).head(5000).tolist()
    )

    selected_edge_id = selected_edge_label.split("|")[0].strip()
    selected_edge = edges[edges["edge_id"].astype(str) == selected_edge_id]

    st.subheader("关系边详情")
    show_dataframe(selected_edge, height=220)

    if not selected_edge.empty:
        row = selected_edge.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("关系类型", row.get("relation_type", ""))
        col2.metric("置信度", row.get("confidence", ""))
        col3.metric("事件数", row.get("event_count", ""))
        col4.metric("状态", row.get("status", ""))

        st.subheader("校验说明")
        st.write("触发约束：", row.get("triggered_constraints", ""))
        st.write("处理动作：", row.get("check_action", ""))
        st.write("处理原因：", row.get("check_reason", ""))

    st.subheader("相关冲突记录")

    if not conflicts.empty and "edge_id" in conflicts.columns:
        related_conflicts = conflicts[conflicts["edge_id"].astype(str) == selected_edge_id]
        show_dataframe(related_conflicts, height=320)
    else:
        st.info("未找到对应冲突记录。")


# =========================
# 冲突列表页
# =========================

def page_conflict_list(data):
    st.title("冲突列表")

    conflicts = data["conflicts"]

    if conflicts.empty:
        st.info("未找到 detected_conflicts.csv。")
        return

    st.sidebar.subheader("冲突筛选")

    actions = sorted(conflicts["action"].dropna().astype(str).unique().tolist()) if "action" in conflicts.columns else []
    conflict_types = sorted(conflicts["conflict_type"].dropna().astype(str).unique().tolist()) if "conflict_type" in conflicts.columns else []
    constraints = sorted(conflicts["triggered_constraint"].dropna().astype(str).unique().tolist()) if "triggered_constraint" in conflicts.columns else []

    selected_action = st.sidebar.selectbox("处理动作", ["全部"] + actions)
    selected_conflict_type = st.sidebar.selectbox("冲突类型", ["全部"] + conflict_types)
    selected_constraint = st.sidebar.selectbox("触发约束", ["全部"] + constraints)
    keyword = st.sidebar.text_input("关键词", "")

    filtered = conflicts.copy()

    if selected_action != "全部":
        filtered = filtered[filtered["action"].astype(str) == selected_action]

    if selected_conflict_type != "全部":
        filtered = filtered[filtered["conflict_type"].astype(str) == selected_conflict_type]

    if selected_constraint != "全部":
        filtered = filtered[filtered["triggered_constraint"].astype(str) == selected_constraint]

    if keyword:
        k = keyword.lower()
        mask = pd.Series(False, index=filtered.index)

        for col in filtered.columns:
            mask = mask | filtered[col].fillna("").astype(str).str.lower().str.contains(k, regex=False)

        filtered = filtered[mask]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("冲突记录数", len(filtered))

    if "action" in filtered.columns:
        col2.metric("Hide", len(filtered[filtered["action"] == "hide"]))
        col3.metric("Downgrade", len(filtered[filtered["action"] == "downgrade"]))
        col4.metric("Mark Mixed", len(filtered[filtered["action"].astype(str).str.contains("mixed", case=False, regex=False)]))

    st.subheader("冲突记录表")
    show_dataframe(filtered, height=620)


# =========================
# 约束列表页
# =========================

def page_constraints(data):
    st.title("最终约束规则")

    constraints = data["constraints"]

    if constraints.empty:
        st.info("未找到 temporal_constraints_final.csv。")
        return

    col1, col2, col3 = st.columns(3)

    col1.metric("最终约束数量", len(constraints))

    if "source" in constraints.columns:
        col2.metric("约束来源类型数", constraints["source"].nunique())

    if "enabled" in constraints.columns:
        enabled_count = len(
            constraints[
                constraints["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])
            ]
        )
        col3.metric("启用约束数", enabled_count)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("来源分布")
        if "source" in constraints.columns:
            source_df = constraints["source"].fillna("unknown").value_counts().reset_index()
            source_df.columns = ["source", "count"]
            st.bar_chart(source_df.set_index("source"))
            show_dataframe(source_df, height=180)

    with right:
        st.subheader("软硬约束分布")
        if "hard_or_soft" in constraints.columns:
            hs_df = constraints["hard_or_soft"].fillna("unknown").value_counts().reset_index()
            hs_df.columns = ["hard_or_soft", "count"]
            st.bar_chart(hs_df.set_index("hard_or_soft"))
            show_dataframe(hs_df, height=180)

    st.subheader("约束详情")
    show_dataframe(constraints, height=500)


# =========================
# 主程序
# =========================

def main():
    data = load_all_data()

    st.sidebar.title("组织关系图谱系统")

    page = st.sidebar.radio(
        "选择页面",
        [
            "dashboard",
            "graph_timeline",
            "node_detail",
            "edge_detail",
            "conflict_list",
            "constraints"
        ]
    )

    st.sidebar.divider()
    st.sidebar.caption("数据目录")
    st.sidebar.code(DATA_DIR)

    if page == "dashboard":
        page_dashboard(data)
    elif page == "graph_timeline":
        page_graph_timeline(data)
    elif page == "node_detail":
        page_node_detail(data)
    elif page == "edge_detail":
        page_edge_detail(data)
    elif page == "conflict_list":
        page_conflict_list(data)
    elif page == "constraints":
        page_constraints(data)


if __name__ == "__main__":
    main()
# app/app.py
# -*- coding: utf-8 -*-

"""
组织关系图谱校验更新可视化系统。

页面：
1. 总览页
2. 时间轴图谱页
3. 互斥约束页
4. 冲突列表页
5. 案例页

改动重点：
1. 不要求修改 11_update_graph.py。
2. 直接读取 relation_edges_after_check.csv 中的 subject_name / object_name。
3. 在 app 层把 ORG_000XXX 显示为真实组织名称。
4. 边关系显示中文实际含义，例如 04 -> 协商磋商，19 -> 军事冲突。
5. 保持 ECharts 连接字段 source / target 仍然使用 ORG_ID，避免图谱消失。

运行：
    streamlit run ./app.py
"""

import os
import json
import heapq
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    from streamlit_echarts import st_echarts
    HAS_ECHARTS = True
except Exception:
    HAS_ECHARTS = False


def detect_project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    parent = os.path.abspath(os.path.join(here, ".."))

    if os.path.exists(os.path.join(parent, "data", "processed")):
        return parent

    if os.path.exists(os.path.join(here, "data", "processed")):
        return here

    return parent


PROJECT_ROOT = detect_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CASE_DIR = os.path.join(OUTPUTS_DIR, "case_studies")

TIMELINE_JSON = os.path.join(PROCESSED_DIR, "timeline_graph_data.json")
EDGES_AFTER = os.path.join(PROCESSED_DIR, "relation_edges_after_check.csv")
FINAL_CONSTRAINTS = os.path.join(PROCESSED_DIR, "mutual_exclusion_constraints_final.csv")
CONFLICTS = os.path.join(PROCESSED_DIR, "mutual_exclusion_conflicts.csv")
DECISIONS = os.path.join(PROCESSED_DIR, "update_decisions.csv")

EDGE_BOOL=False


STATUS_COLORS = {
    "keep": "#4caf50",
    "downgraded": "#ff9800",
    "hidden": "#9e9e9e",
    "mark_mixed": "#7e57c2",
    "review": "#f44336",
}


RELATION_ZH = {
    "01": "公开声明",
    "02": "呼吁请求",
    "03": "表达合作意愿",
    "04": "协商磋商",
    "05": "外交合作",
    "06": "实质合作",
    "07": "提供援助",
    "08": "让步妥协",
    "09": "调查询问",
    "10": "要求施压",
    "11": "反对批评",
    "12": "拒绝合作",
    "13": "威胁警告",
    "14": "抗议冲突",
    "15": "非常规暴力",
    "16": "制裁断交",
    "17": "强制行动",
    "18": "军事攻击",
    "19": "军事冲突",
    "20": "战争暴力",

    "verbal_cooperation": "言语合作",
    "material_cooperation": "实质合作",
    "verbal_conflict": "言语冲突",
    "material_conflict": "实质冲突",
    "mixed_relation": "混合关系",
}


ACTION_ZH = {
    "keep": "保留",
    "review": "人工复核",
    "mark_mixed": "标记复杂关系",
    "downgrade": "置信度降权",
    "hide_low_evidence": "隐藏低证据边",
    "hide": "隐藏",
}


def normalize_relation_type(x: Any) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    if s.isdigit() and len(s) == 1:
        s = "0" + s

    return s


def relation_label(x: Any) -> str:
    r = normalize_relation_type(x)
    return RELATION_ZH.get(r, r)


def action_label(x: Any) -> str:
    s = str(x).strip()
    return ACTION_ZH.get(s, s)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def load_timeline_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"months": [], "graphs": {}}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        v = float(x)
        if pd.isna(v) or v == float("inf") or v == float("-inf"):
            return default
        return v
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    """把 None/NaN/inf 等异常值转为前端安全字符串。"""
    try:
        if x is None or pd.isna(x):
            return default
    except Exception:
        pass

    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return default
    return s


def normalize_node_id(x: Any) -> str:
    """统一清洗节点 ID，避免 link.source/target 与 node.id 因空格或 NaN 不匹配。"""
    return safe_str(x, "")


def get_confidence_value(link: Dict[str, Any], default: float = 0.0) -> float:
    """兼容 confidence / final_confidence 两种字段。"""
    return safe_float(link.get("confidence", link.get("final_confidence", default)), default)


def get_sources_value(link: Dict[str, Any]) -> str:
    """兼容 sources / source_datasets 两种字段。"""
    return safe_str(link.get("sources", link.get("source_datasets", "")), "")


def clean_for_echarts(obj: Any) -> Any:
    """
    递归清理 ECharts option 中不适合前端 JSON 渲染的值。
    重点处理 NaN、inf、None 等异常值，避免前端空白。
    """
    if obj is None:
        return ""

    if isinstance(obj, float):
        if pd.isna(obj) or obj == float("inf") or obj == float("-inf"):
            return 0
        return obj

    if isinstance(obj, (int, bool, str)):
        return obj

    if isinstance(obj, dict):
        return {str(k): clean_for_echarts(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_for_echarts(x) for x in obj]

    try:
        if pd.isna(obj):
            return ""
    except Exception:
        pass

    return obj

def metric_card(label: str, value: Any) -> None:
    st.metric(label, value)


def get_unique_values(df: pd.DataFrame, col: str) -> List[str]:
    if df.empty or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).unique().tolist()
    return sorted(vals)


def build_display_maps(edges: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    从 relation_edges_after_check.csv 构造展示映射。

    返回：
        org_name_map:
            ORG_000001 -> China
        relation_label_map:
            04 -> 协商磋商
    """
    org_name_map: Dict[str, str] = {}
    relation_label_map: Dict[str, str] = {}

    if edges.empty:
        return org_name_map, relation_label_map

    if {"subject_org_id", "subject_name"}.issubset(edges.columns):
        for _, r in edges[["subject_org_id", "subject_name"]].dropna().iterrows():
            org_id = str(r["subject_org_id"]).strip()
            name = str(r["subject_name"]).strip()
            if org_id and name and name.lower() != "nan":
                org_name_map[org_id] = name

    if {"object_org_id", "object_name"}.issubset(edges.columns):
        for _, r in edges[["object_org_id", "object_name"]].dropna().iterrows():
            org_id = str(r["object_org_id"]).strip()
            name = str(r["object_name"]).strip()
            if org_id and name and name.lower() != "nan":
                org_name_map[org_id] = name

    relation_col = ""
    for c in ["_relation", "relation_type", "relation_code"]:
        if c in edges.columns:
            relation_col = c
            break

    if relation_col:
        for rel in edges[relation_col].dropna().unique().tolist():
            rel_norm = normalize_relation_type(rel)
            relation_label_map[rel_norm] = relation_label(rel_norm)

    return org_name_map, relation_label_map


def enrich_timeline_data(
    timeline_data: Dict[str, Any],
    org_name_map: Dict[str, str],
    relation_label_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    不改变图谱连接结构，只增强显示字段。

    关键规则：
        node.id 不改，仍然是 ORG_ID；
        node.name 改成真实名称；
        link.source / link.target 不改，仍然是 ORG_ID；
        link.source_name / link.target_name 增加真实名称；
        link.relation_label 增加中文关系名。
    """
    data = json.loads(json.dumps(timeline_data, ensure_ascii=False))
    graphs = data.get("graphs", {})

    for month, graph in graphs.items():
        nodes = graph.get("nodes", [])
        links = graph.get("links", [])

        for n in nodes:
            node_id = str(n.get("id", "")).strip()
            display_name = org_name_map.get(node_id, n.get("name", node_id))

            if not display_name or str(display_name).lower() == "nan":
                display_name = node_id

            n["name"] = display_name
            n["org_id"] = node_id

        for link in links:
            source_id = str(link.get("source", "")).strip()
            target_id = str(link.get("target", "")).strip()

            source_name = org_name_map.get(source_id, source_id)
            target_name = org_name_map.get(target_id, target_id)

            rel = normalize_relation_type(link.get("relation", link.get("relation_code", "")))
            rel_label = relation_label_map.get(rel, relation_label(rel))

            link["source_org_id"] = source_id
            link["target_org_id"] = target_id
            link["source_name"] = source_name
            link["target_name"] = target_name
            link["relation"] = rel
            link["relation_code"] = rel
            link["relation_label"] = rel_label
            link["edge_label"] = rel_label
            link["update_action_label"] = action_label(link.get("update_action", link.get("status", "keep")))

    return data


def filter_links(
    links: List[Dict[str, Any]],
    relation_types: List[str],
    actions: List[str],
    min_confidence: float,
    only_triggered: bool,
    only_cross_source: bool,
    keyword: str,
) -> List[Dict[str, Any]]:
    result = []
    kw = safe_str(keyword, "").lower()

    for raw_link in links:
        link = dict(raw_link)

        relation = normalize_relation_type(link.get("relation", link.get("relation_code", "")))
        relation_zh = safe_str(link.get("relation_label", relation_label(relation)), relation_label(relation))

        action = safe_str(link.get("update_action", link.get("status", "keep")), "keep")
        action_zh = safe_str(link.get("update_action_label", action_label(action)), action_label(action))

        conf = get_confidence_value(link, 0.0)
        triggered = bool(link.get("is_mutex_triggered", False))
        sources = get_sources_value(link).upper()

        source_id = normalize_node_id(link.get("source", link.get("source_org_id", "")))
        target_id = normalize_node_id(link.get("target", link.get("target_org_id", "")))
        source_name = safe_str(link.get("source_name", source_id), source_id)
        target_name = safe_str(link.get("target_name", target_id), target_id)

        if not source_id or not target_id:
            continue

        link["source"] = source_id
        link["target"] = target_id
        link["source_org_id"] = safe_str(link.get("source_org_id", source_id), source_id)
        link["target_org_id"] = safe_str(link.get("target_org_id", target_id), target_id)
        link["source_name"] = source_name
        link["target_name"] = target_name
        link["relation"] = relation
        link["relation_code"] = relation
        link["relation_label"] = relation_zh
        link["confidence"] = conf
        link["sources"] = get_sources_value(link)
        link["update_action"] = action
        link["update_action_label"] = action_zh

        if relation_types and relation not in relation_types:
            continue

        if actions and action not in actions and action_zh not in actions:
            continue

        if conf < min_confidence:
            continue

        if only_triggered and not triggered:
            continue

        if only_cross_source and not ("ICEWS" in sources and "GDELT" in sources):
            continue

        if kw:
            searchable = " ".join([
                source_name,
                target_name,
                link["source_org_id"],
                link["target_org_id"],
                relation,
                relation_zh,
                action,
                action_zh,
            ]).lower()

            if kw not in searchable:
                continue

        result.append(link)

    return result

def build_echarts_option(
    timeline_data: Dict[str, Any],
    selected_months: List[str],
    relation_types: List[str],
    actions: List[str],
    min_confidence: float,
    only_triggered: bool,
    only_cross_source: bool,
    keyword: str,
    show_edge_label: bool,
    max_edges_per_month: int,
) -> Dict[str, Any]:
    """
    构建“时间轴图谱页”的 ECharts option。

    这一版专门修复全局时间轴图谱空白的问题：
    1. 每月限制边数，避免 force 布局被全量图拖死；
    2. 不再把原始节点字段 **n 全量传给 ECharts，只保留必要字段；
    3. 统一清洗 source/target/node.id；
    4. 兼容 confidence/final_confidence 与 sources/source_datasets。
    """
    all_months = timeline_data.get("months", [])
    graphs = timeline_data.get("graphs", {})

    months = selected_months if selected_months else all_months
    options = []

    for month in months:
        graph = graphs.get(month, {"nodes": [], "links": []})

        links = filter_links(
            graph.get("links", []),
            relation_types=relation_types,
            actions=actions,
            min_confidence=min_confidence,
            only_triggered=only_triggered,
            only_cross_source=only_cross_source,
            keyword=keyword,
        )

        # 只取前 N 条，不完整排序全部边，减少耗时。
        links = heapq.nlargest(
            max_edges_per_month,
            links,
            key=lambda x: safe_float(
                x.get("confidence", x.get("final_confidence", 0.0)),
                0.0
            ),
        )

        node_ids = set()
        for link in links:
            source_id = normalize_node_id(link.get("source", link.get("source_org_id", "")))
            target_id = normalize_node_id(link.get("target", link.get("target_org_id", "")))
            if not source_id or not target_id:
                continue
            link["source"] = source_id
            link["target"] = target_id
            node_ids.add(source_id)
            node_ids.add(target_id)

        nodes = []
        for raw_node in graph.get("nodes", []):
            node_id = normalize_node_id(raw_node.get("id", raw_node.get("org_id", "")))
            if not node_id or node_id not in node_ids:
                continue

            display_name = safe_str(raw_node.get("name", node_id), node_id)
            org_id = normalize_node_id(raw_node.get("org_id", node_id))
            node_value = safe_float(raw_node.get("value", 1), 1)

            nodes.append({
                "id": node_id,
                "name": display_name,
                "org_id": org_id,
                "value": node_value,
                "symbolSize": 24,
                "category": "organization",
                "label": {
                    "show": True,
                    "formatter": display_name,
                    "fontSize": 10,
                },
                "tooltip": {
                    "formatter": (
                        f"组织名称：{display_name}<br/>"
                        f"组织ID：{org_id}<br/>"
                        f"连接数：{node_value}"
                    )
                },
            })

        valid_node_ids = {n["id"] for n in nodes}
        formatted_links = []

        for link in links:
            source_id = normalize_node_id(link.get("source", link.get("source_org_id", "")))
            target_id = normalize_node_id(link.get("target", link.get("target_org_id", "")))
            if source_id not in valid_node_ids or target_id not in valid_node_ids:
                continue

            status = safe_str(link.get("status", "keep"), "keep")
            action = safe_str(link.get("update_action", status), status)
            color = STATUS_COLORS.get(status, STATUS_COLORS.get(action, "#607d8b"))

            relation_code = normalize_relation_type(link.get("relation", link.get("relation_code", "")))
            relation_zh = safe_str(link.get("relation_label", relation_label(relation_code)), relation_label(relation_code))

            source_name = safe_str(link.get("source_name", source_id), source_id)
            target_name = safe_str(link.get("target_name", target_id), target_id)
            conf_value = get_confidence_value(link, 0.0)
            sources_text = get_sources_value(link)
            is_triggered = bool(link.get("is_mutex_triggered", False))

            tooltip = (
                f"月份：{month}<br/>"
                f"主体：{source_name}<br/>"
                f"主体ID：{safe_str(link.get('source_org_id', source_id), source_id)}<br/>"
                f"客体：{target_name}<br/>"
                f"客体ID：{safe_str(link.get('target_org_id', target_id), target_id)}<br/>"
                f"关系：{relation_zh}<br/>"
                f"关系码：{relation_code}<br/>"
                f"置信度：{conf_value}<br/>"
                f"原始置信度：{safe_str(link.get('original_confidence', ''), '')}<br/>"
                f"来源：{sources_text}<br/>"
                f"状态：{status}<br/>"
                f"动作：{safe_str(link.get('update_action_label', action_label(action)), action_label(action))}<br/>"
                f"触发约束：{safe_str(link.get('triggered_constraint_ids', ''), '')}<br/>"
                f"触发互斥关系：{safe_str(link.get('triggered_mutex_relations', ''), '')}<br/>"
                f"原因：{safe_str(link.get('update_reason', ''), '')}"
            )

            formatted_links.append({
                "source": source_id,
                "target": target_id,
                "name": relation_zh,
                "value": conf_value,
                "tooltip": {"formatter": tooltip},
                "lineStyle": {
                    "color": color,
                    "width": 2.8 if is_triggered else 1.2,
                    "opacity": 0.85,
                    "curveness": 0.15,
                },
                "label": {
                    "show": show_edge_label,
                    "formatter": relation_zh,
                    "fontSize": 9,
                },
            })

        options.append({
            "title": {
                "text": f"{month} 组织关系图谱（节点：{len(nodes)}，边：{len(formatted_links)}）",
                "left": "center",
            },
            "series": [{
                "type": "graph",
                "layout": "force",
                "roam": True,
                "draggable": True,
                "data": nodes,
                "links": formatted_links,
                "categories": [{"name": "organization"}],
                "force": {
                    "repulsion": 160,
                    "edgeLength": 105,
                },
                "label": {
                    "show": True,
                    "fontSize": 10,
                },
                "edgeLabel": {
                    "show": show_edge_label,
                    "fontSize": 9,
                },
                "emphasis": {
                    "focus": "adjacency",
                    "lineStyle": {
                        "width": 4,
                    },
                },
            }]
        })

    option = {
        "baseOption": {
            "timeline": {
                "axisType": "category",
                "autoPlay": False,
                "playInterval": 1500,
                "data": months,
                "bottom": 5,
            },
            "tooltip": {},
            "legend": {
                "data": ["organization"],
                "top": 30,
            },
            "series": [{
                "type": "graph",
                "layout": "force",
            }],
        },
        "options": options,
    }

    return clean_for_echarts(option)

def filter_node_links(
    links: List[Dict[str, Any]],
    keyword: str,
    relation_types: List[str],
    actions: List[str],
    min_confidence: float,
    only_triggered: bool,
    only_cross_source: bool,
) -> List[Dict[str, Any]]:
    """
    仅筛选与某个节点相关的一阶关系边。

    与普通时间轴图谱的 keyword 不同，这里的 keyword 只用于匹配节点，
    即 source / target 的 ORG_ID 或真实组织名称，不匹配关系名称。
    """
    result: List[Dict[str, Any]] = []
    kw = str(keyword or "").strip().lower()

    if not kw:
        return result

    for link in links:
        source_id = str(link.get("source_org_id", link.get("source", "")))
        target_id = str(link.get("target_org_id", link.get("target", "")))
        source_name = str(link.get("source_name", source_id))
        target_name = str(link.get("target_name", target_id))

        node_searchable = " ".join([
            source_id,
            target_id,
            source_name,
            target_name,
        ]).lower()

        if kw not in node_searchable:
            continue

        relation = normalize_relation_type(link.get("relation", ""))
        action = str(link.get("update_action", link.get("status", "keep")))
        action_zh = str(link.get("update_action_label", action_label(action)))
        conf = safe_float(link.get("confidence", 0.0), 0.0)
        triggered = bool(link.get("is_mutex_triggered", False))
        sources = str(link.get("sources", "")).upper()

        if relation_types and relation not in relation_types:
            continue

        if actions and action not in actions and action_zh not in actions:
            continue

        if conf < min_confidence:
            continue

        if only_triggered and not triggered:
            continue

        if only_cross_source and not ("ICEWS" in sources and "GDELT" in sources):
            continue

        result.append(link)

    return result


def build_node_evolution_echarts_option(
    timeline_data: Dict[str, Any],
    selected_months: List[str],
    node_keyword: str,
    relation_types: List[str],
    actions: List[str],
    min_confidence: float,
    only_triggered: bool,
    only_cross_source: bool,
    show_edge_label: bool,
    max_edges_per_month: int,
) -> Dict[str, Any]:
    """
    构建“单节点关系演化”时间轴图谱。

    展示逻辑：
        1. 输入一个节点名称或 ORG_ID，例如 CHINA；
        2. 每个月只展示该节点的一阶关系边；
        3. 连接字段 source / target 仍使用 ORG_ID，避免图谱断连；
        4. 节点 label 使用真实组织名称；
        5. 目标节点会被放大显示。
    """
    all_months = timeline_data.get("months", [])
    graphs = timeline_data.get("graphs", {})

    months = selected_months if selected_months else all_months
    kw = str(node_keyword or "").strip().lower()
    options: List[Dict[str, Any]] = []

    for month in months:
        graph = graphs.get(month, {"nodes": [], "links": []})

        links = filter_node_links(
            graph.get("links", []),
            keyword=node_keyword,
            relation_types=relation_types,
            actions=actions,
            min_confidence=min_confidence,
            only_triggered=only_triggered,
            only_cross_source=only_cross_source,
        )

        links = sorted(
            links,
            key=lambda x: safe_float(x.get("confidence", 0.0), 0.0),
            reverse=True,
        )[:max_edges_per_month]

        node_ids = set()
        for link in links:
            node_ids.add(str(link.get("source", "")))
            node_ids.add(str(link.get("target", "")))

        nodes = []
        center_node_count = 0

        for n in graph.get("nodes", []):
            node_id = str(n.get("id", ""))
            if node_id not in node_ids:
                continue

            display_name = str(n.get("name", node_id))
            org_id = str(n.get("org_id", node_id))
            is_center = bool(kw and (kw in display_name.lower() or kw in org_id.lower() or kw in node_id.lower()))
            if is_center:
                center_node_count += 1

            nodes.append({
                **n,
                "symbolSize": 42 if is_center else 24,
                "category": "中心节点" if is_center else "关联节点",
                "label": {
                    "show": True,
                    "formatter": display_name,
                    "fontSize": 12 if is_center else 10,
                    "fontWeight": "bold" if is_center else "normal",
                },
                "tooltip": {
                    "formatter": (
                        f"组织名称：{display_name}<br/>"
                        f"组织ID：{org_id}<br/>"
                        f"节点类型：{'中心节点' if is_center else '关联节点'}<br/>"
                        f"连接数：{n.get('value', '')}"
                    )
                },
            })

        formatted_links = []
        for link in links:
            status = str(link.get("status", "keep"))
            action = str(link.get("update_action", status))
            color = STATUS_COLORS.get(status, STATUS_COLORS.get(action, "#607d8b"))

            relation_code = normalize_relation_type(link.get("relation", ""))
            relation_zh = str(link.get("relation_label", relation_label(relation_code)))
            source_name = str(link.get("source_name", link.get("source", "")))
            target_name = str(link.get("target_name", link.get("target", "")))
            is_triggered = bool(link.get("is_mutex_triggered", False))

            tooltip = (
                f"月份：{month}<br/>"
                f"主体：{source_name}<br/>"
                f"主体ID：{link.get('source_org_id', link.get('source', ''))}<br/>"
                f"客体：{target_name}<br/>"
                f"客体ID：{link.get('target_org_id', link.get('target', ''))}<br/>"
                f"关系：{relation_zh}<br/>"
                f"关系码：{relation_code}<br/>"
                f"置信度：{link.get('confidence', '')}<br/>"
                f"原始置信度：{link.get('original_confidence', '')}<br/>"
                f"来源：{link.get('sources', '')}<br/>"
                f"状态：{status}<br/>"
                f"动作：{link.get('update_action_label', action_label(action))}<br/>"
                f"触发约束：{link.get('triggered_constraint_ids', '')}<br/>"
                f"触发互斥关系：{link.get('triggered_mutex_relations', '')}<br/>"
                f"原因：{link.get('update_reason', '')}"
            )

            formatted_links.append({
                "source": link.get("source", ""),
                "target": link.get("target", ""),
                "name": relation_zh,
                "value": link.get("confidence", 0),
                "tooltip": {"formatter": tooltip},
                "lineStyle": {
                    "color": color,
                    "width": 3.0 if is_triggered else 1.3,
                    "opacity": 0.85,
                    "curveness": 0.18,
                },
                "label": {
                    "show": show_edge_label,
                    "formatter": relation_zh,
                    "fontSize": 9,
                },
            })

        options.append({
            "title": {
                "text": f"{month} {node_keyword} 关系演化图谱（边数：{len(formatted_links)}）",
                "left": "center",
            },
            "legend": {
                "data": ["中心节点", "关联节点"],
                "top": 30,
            },
            "series": [{
                "type": "graph",
                "layout": "force",
                "roam": True,
                "draggable": True,
                "data": nodes,
                "links": formatted_links,
                "categories": [
                    {"name": "中心节点"},
                    {"name": "关联节点"},
                ],
                "force": {
                    "repulsion": 210,
                    "edgeLength": 120,
                },
                "label": {
                    "show": True,
                    "fontSize": 10,
                },
                "edgeLabel": {
                    "show": show_edge_label,
                    "fontSize": 9,
                },
                "emphasis": {
                    "focus": "adjacency",
                    "lineStyle": {
                        "width": 4,
                    },
                },
            }]
        })

    option = {
        "baseOption": {
            "timeline": {
                "axisType": "category",
                "autoPlay": False,
                "playInterval": 1500,
                "data": months,
                "bottom": 5,
            },
            "tooltip": {},
            "legend": {
                "top": 30,
            },
            "series": [{
                "type": "graph",
                "layout": "force",
            }],
        },
        "options": options,
    }

    return option


def collect_node_evolution_rows(
    timeline_data: Dict[str, Any],
    selected_months: List[str],
    node_keyword: str,
    relation_types: List[str],
    actions: List[str],
    min_confidence: float,
    only_triggered: bool,
    only_cross_source: bool,
) -> pd.DataFrame:
    """将单节点演化图谱中的 link 展开为表格，便于统计和明细展示。"""
    rows: List[Dict[str, Any]] = []
    months = selected_months if selected_months else timeline_data.get("months", [])
    graphs = timeline_data.get("graphs", {})

    for month in months:
        graph = graphs.get(month, {"links": []})
        links = filter_node_links(
            graph.get("links", []),
            keyword=node_keyword,
            relation_types=relation_types,
            actions=actions,
            min_confidence=min_confidence,
            only_triggered=only_triggered,
            only_cross_source=only_cross_source,
        )

        for link in links:
            relation_code = normalize_relation_type(link.get("relation", ""))
            action = str(link.get("update_action", link.get("status", "keep")))
            rows.append({
                "month": month,
                "source_name": link.get("source_name", link.get("source", "")),
                "source_org_id": link.get("source_org_id", link.get("source", "")),
                "relation_code": relation_code,
                "relation_label": link.get("relation_label", relation_label(relation_code)),
                "target_name": link.get("target_name", link.get("target", "")),
                "target_org_id": link.get("target_org_id", link.get("target", "")),
                "confidence": safe_float(link.get("confidence", 0.0), 0.0),
                "original_confidence": link.get("original_confidence", ""),
                "sources": link.get("sources", ""),
                "status": link.get("status", ""),
                "update_action": action,
                "update_action_label": link.get("update_action_label", action_label(action)),
                "is_mutex_triggered": link.get("is_mutex_triggered", False),
                "triggered_constraint_ids": link.get("triggered_constraint_ids", ""),
                "triggered_mutex_relations": link.get("triggered_mutex_relations", ""),
                "update_reason": link.get("update_reason", ""),
            })

    return pd.DataFrame(rows)


def page_overview() -> None:
    st.title("组织关系图谱校验更新总览")

    edges = load_csv(EDGES_AFTER)
    constraints = load_csv(FINAL_CONSTRAINTS)
    conflicts = load_csv(CONFLICTS)
    decisions = load_csv(DECISIONS)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("关系边数", len(edges))

    with c2:
        metric_card("互斥约束数", len(constraints))

    with c3:
        metric_card("触发冲突数", len(conflicts))

    with c4:
        metric_card("更新决策数", len(decisions))

    st.divider()

    if not edges.empty and "update_action" in edges.columns:
        st.subheader("边更新动作分布")
        tmp = edges.copy()
        tmp["update_action_label"] = tmp["update_action"].apply(action_label)
        action_counts = tmp["update_action_label"].value_counts().reset_index()
        action_counts.columns = ["update_action", "count"]
        st.bar_chart(action_counts.set_index("update_action"))

    if not constraints.empty and "final_status" in constraints.columns:
        st.subheader("最终约束状态分布")
        status_counts = constraints["final_status"].value_counts().reset_index()
        status_counts.columns = ["final_status", "count"]
        st.bar_chart(status_counts.set_index("final_status"))

    if not conflicts.empty and "month" in conflicts.columns:
        st.subheader("每月互斥冲突数量")
        month_counts = conflicts["month"].value_counts().sort_index().reset_index()
        month_counts.columns = ["month", "count"]
        st.line_chart(month_counts.set_index("month"))

    st.subheader("文件状态")
    st.write({
        "timeline_graph_data.json": os.path.exists(TIMELINE_JSON),
        "relation_edges_after_check.csv": os.path.exists(EDGES_AFTER),
        "mutual_exclusion_constraints_final.csv": os.path.exists(FINAL_CONSTRAINTS),
        "mutual_exclusion_conflicts.csv": os.path.exists(CONFLICTS),
        "update_decisions.csv": os.path.exists(DECISIONS),
    })


def page_timeline_graph() -> None:
    st.title("时间轴组织关系图谱")

    edges = load_csv(EDGES_AFTER)
    org_name_map, relation_label_map = build_display_maps(edges)

    timeline_data_raw = load_timeline_json(TIMELINE_JSON)
    timeline_data = enrich_timeline_data(
        timeline_data=timeline_data_raw,
        org_name_map=org_name_map,
        relation_label_map=relation_label_map,
    )

    months = timeline_data.get("months", [])

    if not months:
        st.warning("未找到 timeline_graph_data.json 或其中没有月份数据。请先运行第 11 步。")
        return

    all_links = []
    for m in months:
        all_links.extend(timeline_data.get("graphs", {}).get(m, {}).get("links", []))

    if not all_links:
        st.warning("timeline_graph_data.json 中没有关系边数据。")
        return

    all_relation_types = sorted(set(
        normalize_relation_type(x.get("relation", x.get("relation_code", "")))
        for x in all_links
        if x.get("relation", x.get("relation_code", ""))
    ))

    relation_display = {
        r: f"{r} - {relation_label(r)}"
        for r in all_relation_types
    }

    all_actions = sorted(set(safe_str(x.get("update_action", x.get("status", "keep")), "keep") for x in all_links))

    st.sidebar.subheader("图谱筛选")

    selected_months = st.sidebar.multiselect(
        "月份",
        options=months,
        default=months,
    )

    selected_relation_display = st.sidebar.multiselect(
        "关系类型",
        options=[relation_display[r] for r in all_relation_types],
        default=[],
    )

    relation_types = []
    for item in selected_relation_display:
        relation_types.append(item.split(" - ")[0].strip())

    actions = st.sidebar.multiselect(
        "更新动作",
        options=all_actions,
        default=[],
        format_func=lambda x: f"{x} - {action_label(x)}",
    )

    min_confidence = st.sidebar.slider(
        "最低关系置信度",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    only_triggered = st.sidebar.checkbox("只看触发互斥约束的边", value=False)
    only_cross_source = st.sidebar.checkbox("只看 ICEWS+GDELT 共同支持关系", value=False)
    show_edge_label = st.sidebar.checkbox("显示边关系中文标签", value=EDGE_BOOL)

    max_edges_per_month = st.sidebar.slider(
        "每月最多显示边数",
        min_value=10,
        max_value=1000,
        value=400,
        step=10,
    )

    keyword = st.sidebar.text_input(
        "搜索组织名称 / ORG_ID / 关系",
        value="",
    )

    debug_single_month = st.sidebar.checkbox(
        "调试：只显示第一个月份，不使用时间轴",
        value=False,
    )

    show_debug_table = st.sidebar.checkbox(
        "显示图谱调试表",
        value=True,
    )

    st.caption(
        "节点显示真实名称；边显示中文关系含义。"
        "颜色含义：keep=绿色，downgraded=橙色，hidden=灰色，mark_mixed=紫色，review=红色。"
        "如果全局时间轴不显示，可勾选“调试：只显示第一个月份”。"
    )

    option = build_echarts_option(
        timeline_data=timeline_data,
        selected_months=selected_months,
        relation_types=relation_types,
        actions=actions,
        min_confidence=min_confidence,
        only_triggered=only_triggered,
        only_cross_source=only_cross_source,
        keyword=keyword,
        show_edge_label=show_edge_label,
        max_edges_per_month=max_edges_per_month,
    )

    debug_options = option.get("options", [])
    debug_months = selected_months if selected_months else months
    debug_rows = []
    for i, opt in enumerate(debug_options):
        month_name = debug_months[i] if i < len(debug_months) else str(i)
        series = opt.get("series", [{}])[0]
        debug_rows.append({
            "month": month_name,
            "nodes": len(series.get("data", [])),
            "links": len(series.get("links", [])),
        })

    if show_debug_table:
        st.subheader("时间轴图谱调试信息")
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, height=180)

    if HAS_ECHARTS:
        if debug_single_month:
            first_option = debug_options[0] if debug_options else {}
            st_echarts(
                options=first_option,
                height="760px",
                key="timeline_graph_single_month_debug",
            )
        else:
            st_echarts(
                options=option,
                height="760px",
                key="timeline_graph_main",
            )
    else:
        st.warning("未安装 streamlit-echarts，无法展示 ECharts 图。请运行：pip install streamlit-echarts")
        st.json(option)

    st.subheader("当前筛选说明")
    st.write({
        "selected_months": selected_months,
        "relation_types": relation_types,
        "actions": actions,
        "min_confidence": min_confidence,
        "only_triggered": only_triggered,
        "only_cross_source": only_cross_source,
        "show_edge_label": show_edge_label,
        "max_edges_per_month": max_edges_per_month,
        "keyword": keyword,
        "debug_single_month": debug_single_month,
    })

def page_node_evolution() -> None:
    st.title("单节点关系演化图谱")

    st.markdown(
        """
        本页面用于集中展示某一个组织节点在不同时期的关系发展情况。
        例如输入 `CHINA`，可以观察该节点在 2023-01 至 2023-04 各月份中的一阶关系网络、关系类型变化、置信度与更新状态。
        """
    )

    edges = load_csv(EDGES_AFTER)
    org_name_map, relation_label_map = build_display_maps(edges)

    timeline_data_raw = load_timeline_json(TIMELINE_JSON)
    timeline_data = enrich_timeline_data(
        timeline_data=timeline_data_raw,
        org_name_map=org_name_map,
        relation_label_map=relation_label_map,
    )

    months = timeline_data.get("months", [])

    if not months:
        st.warning("未找到 timeline_graph_data.json 或其中没有月份数据。请先运行第 11 步。")
        return

    all_links: List[Dict[str, Any]] = []
    for m in months:
        all_links.extend(timeline_data.get("graphs", {}).get(m, {}).get("links", []))

    if not all_links:
        st.warning("timeline_graph_data.json 中没有关系边数据。")
        return

    all_relation_types = sorted(set(
        normalize_relation_type(x.get("relation", ""))
        for x in all_links
        if x.get("relation", "")
    ))

    relation_display = {
        r: f"{r} - {relation_label(r)}"
        for r in all_relation_types
    }

    all_actions = sorted(set(str(x.get("update_action", x.get("status", "keep"))) for x in all_links))

    st.sidebar.subheader("节点演化筛选")

    node_keyword = st.sidebar.text_input(
        "输入节点名称 / ORG_ID",
        value="CHINA",
        help="例如 CHINA、Russia、United States，也可以输入 ORG_000001。",
    )

    selected_months = st.sidebar.multiselect(
        "月份",
        options=months,
        default=months,
    )

    selected_relation_display = st.sidebar.multiselect(
        "关系类型",
        options=[relation_display[r] for r in all_relation_types],
        default=[],
    )

    relation_types = []
    for item in selected_relation_display:
        relation_types.append(item.split(" - ")[0].strip())

    actions = st.sidebar.multiselect(
        "更新动作",
        options=all_actions,
        default=[],
        format_func=lambda x: f"{x} - {action_label(x)}",
    )

    min_confidence = st.sidebar.slider(
        "最低关系置信度",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    only_triggered = st.sidebar.checkbox("只看触发互斥约束的边", value=False)
    only_cross_source = st.sidebar.checkbox("只看 ICEWS+GDELT 共同支持关系", value=False)
    show_edge_label = st.sidebar.checkbox("显示边关系中文标签", value=EDGE_BOOL)

    max_edges_per_month = st.sidebar.slider(
        "每月最多显示边数",
        min_value=20,
        max_value=500,
        value=120,
        step=20,
    )

    if not str(node_keyword).strip():
        st.info("请输入节点名称或 ORG_ID。")
        return

    rows = collect_node_evolution_rows(
        timeline_data=timeline_data,
        selected_months=selected_months,
        node_keyword=node_keyword,
        relation_types=relation_types,
        actions=actions,
        min_confidence=min_confidence,
        only_triggered=only_triggered,
        only_cross_source=only_cross_source,
    )

    if rows.empty:
        st.warning(f"没有找到与 `{node_keyword}` 相关的关系边。可以尝试降低置信度阈值，或输入 ORG_ID。")
        return

    st.caption(
        "该页面只展示目标节点的一阶关系边。节点仍使用 ORG_ID 作为图连接字段，显示名称使用真实组织名称。"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("相关关系边数", len(rows))
    with c2:
        metric_card("涉及月份数", rows["month"].nunique())
    with c3:
        related_nodes = set(rows["source_org_id"].astype(str)) | set(rows["target_org_id"].astype(str))
        metric_card("关联节点数", len(related_nodes))
    with c4:
        metric_card("关系类型数", rows["relation_code"].nunique())

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("月度关系数量")
        month_counts = rows["month"].value_counts().sort_index().reset_index()
        month_counts.columns = ["month", "count"]
        st.bar_chart(month_counts.set_index("month"))
        st.dataframe(month_counts, use_container_width=True, height=220)

    with right:
        st.subheader("关系类型分布")
        relation_counts = rows["relation_label"].value_counts().reset_index()
        relation_counts.columns = ["relation", "count"]
        st.bar_chart(relation_counts.set_index("relation"))
        st.dataframe(relation_counts, use_container_width=True, height=220)

    if "update_action_label" in rows.columns:
        st.subheader("更新动作分布")
        action_counts = rows["update_action_label"].value_counts().reset_index()
        action_counts.columns = ["update_action", "count"]
        st.bar_chart(action_counts.set_index("update_action"))

    st.divider()

    st.subheader(f"`{node_keyword}` 的关系演化图谱")

    option = build_node_evolution_echarts_option(
        timeline_data=timeline_data,
        selected_months=selected_months,
        node_keyword=node_keyword,
        relation_types=relation_types,
        actions=actions,
        min_confidence=min_confidence,
        only_triggered=only_triggered,
        only_cross_source=only_cross_source,
        show_edge_label=show_edge_label,
        max_edges_per_month=max_edges_per_month,
    )

    option = clean_for_echarts(option)

    if HAS_ECHARTS:
        st_echarts(
            options=option,
            height="760px",
            key="node_evolution_graph",
        )
    else:
        st.warning("未安装 streamlit-echarts，无法展示 ECharts 图。请运行：pip install streamlit-echarts")
        st.json(option)

    st.subheader("关系边明细")
    show_cols = [
        "month",
        "source_name",
        "source_org_id",
        "relation_code",
        "relation_label",
        "target_name",
        "target_org_id",
        "confidence",
        "original_confidence",
        "sources",
        "status",
        "update_action_label",
        "is_mutex_triggered",
        "triggered_constraint_ids",
        "triggered_mutex_relations",
        "update_reason",
    ]
    show_cols = [c for c in show_cols if c in rows.columns]
    rows = rows.sort_values(["month", "confidence"], ascending=[True, False])
    st.dataframe(rows[show_cols], use_container_width=True, height=620)


def page_constraints() -> None:
    st.title("互斥约束列表")

    constraints = load_csv(FINAL_CONSTRAINTS)

    if constraints.empty:
        st.warning("未找到 mutual_exclusion_constraints_final.csv。")
        return

    df = constraints.copy()

    if "relation_a" in df.columns:
        df["relation_a_label"] = df["relation_a"].apply(lambda x: f"{normalize_relation_type(x)} - {relation_label(x)}")

    if "relation_b" in df.columns:
        df["relation_b_label"] = df["relation_b"].apply(lambda x: f"{normalize_relation_type(x)} - {relation_label(x)}")

    statuses = get_unique_values(df, "final_status")
    selected_statuses = st.multiselect("final_status", options=statuses, default=statuses)

    if selected_statuses and "final_status" in df.columns:
        df = df[df["final_status"].astype(str).isin(selected_statuses)]

    if "patecon_confidence" in df.columns:
        min_conf = st.slider("最低 PaTeCon-style confidence", 0.0, 1.0, 0.0, 0.05)
        df = df[df["patecon_confidence"].apply(lambda x: safe_float(x, 0.0)) >= min_conf]

    show_cols = [
        "constraint_id",
        "constraint_key",
        "constraint_name",
        "relation_a_label",
        "relation_b_label",
        "candidate_level",
        "auto_filter_status",
        "final_status",
        "patecon_confidence",
        "source_weighted_confidence",
        "violation_rate",
        "llm_expected_action",
        "manual_score",
        "manual_comment",
    ]

    show_cols = [c for c in show_cols if c in df.columns]

    st.dataframe(df[show_cols], use_container_width=True, height=620)


def page_conflicts() -> None:
    st.title("互斥冲突列表")

    conflicts = load_csv(CONFLICTS)

    if conflicts.empty:
        st.warning("未找到 mutual_exclusion_conflicts.csv。")
        return

    df = conflicts.copy()

    if "relation_a" in df.columns:
        df["relation_a_label"] = df["relation_a"].apply(lambda x: f"{normalize_relation_type(x)} - {relation_label(x)}")

    if "relation_b" in df.columns:
        df["relation_b_label"] = df["relation_b"].apply(lambda x: f"{normalize_relation_type(x)} - {relation_label(x)}")

    months = get_unique_values(df, "month")
    actions = get_unique_values(df, "conflict_action")
    constraints = get_unique_values(df, "constraint_id")

    c1, c2, c3 = st.columns(3)

    with c1:
        selected_months = st.multiselect("月份", months, default=months)

    with c2:
        selected_actions = st.multiselect("处理动作", actions, default=actions)

    with c3:
        selected_constraints = st.multiselect("约束 ID", constraints, default=[])

    if selected_months and "month" in df.columns:
        df = df[df["month"].astype(str).isin(selected_months)]

    if selected_actions and "conflict_action" in df.columns:
        df = df[df["conflict_action"].astype(str).isin(selected_actions)]

    if selected_constraints and "constraint_id" in df.columns:
        df = df[df["constraint_id"].astype(str).isin(selected_constraints)]

    st.write(f"当前冲突数量：{len(df)}")

    show_cols = [
        "conflict_id",
        "month",
        "subject_name",
        "object_name",
        "subject_org_id",
        "object_org_id",
        "relation_a_label",
        "relation_b_label",
        "constraint_id",
        "constraint_key",
        "final_status",
        "conflict_action",
        "patecon_confidence",
        "violation_rate",
        "reason",
        "edge_ids_a",
        "edge_ids_b",
    ]

    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, height=650)


def page_cases() -> None:
    st.title("典型案例")

    case_files = []

    if os.path.exists(CASE_DIR):
        for name in sorted(os.listdir(CASE_DIR)):
            if name.endswith(".csv"):
                case_files.append(name)

    if not case_files:
        st.warning("未找到案例文件。请先运行第 11 步生成 outputs/case_studies/。")
        return

    selected = st.selectbox("选择案例文件", case_files)

    path = os.path.join(CASE_DIR, selected)
    df = load_csv(path)

    st.write(f"文件：`{path}`")
    st.write(f"案例数量：{len(df)}")
    st.dataframe(df, use_container_width=True, height=680)


def main() -> None:
    st.set_page_config(
        page_title="组织关系图谱校验更新系统",
        layout="wide",
    )

    st.sidebar.title("组织关系图谱校验更新")

    page = st.sidebar.radio(
        "页面",
        [
            "总览页",
            "时间轴图谱页",
            "单节点关系演化页",
            "互斥约束页",
            "冲突列表页",
            "案例页",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption(f"项目路径：{PROJECT_ROOT}")

    if page == "总览页":
        page_overview()
    elif page == "时间轴图谱页":
        page_timeline_graph()
    elif page == "单节点关系演化页":
        page_node_evolution()
    elif page == "互斥约束页":
        page_constraints()
    elif page == "冲突列表页":
        page_conflicts()
    elif page == "案例页":
        page_cases()


if __name__ == "__main__":
    main()
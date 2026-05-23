# scripts/11_update_graph.py
# -*- coding: utf-8 -*-

import os
import re
import json
import pandas as pd


# ============================================================
# 0. 路径配置
# ============================================================

PROCESSED_DIR = "../data/processed"

RELATION_EDGES_SCORED_FILE = os.path.join(
    PROCESSED_DIR,
    "relation_edges_scored.csv"
)

LLM_CONSTRAINTS_CSV = os.path.join(
    PROCESSED_DIR,
    "temporal_constraints_llm.csv"
)

LLM_CONSTRAINTS_JSON = os.path.join(
    PROCESSED_DIR,
    "llm_constraints.json"
)

PATECON_CONSTRAINTS_FILE = os.path.join(
    PROCESSED_DIR,
    "patecon_constraints.csv"
)

PATECON_CONFLICTS_FILE = os.path.join(
    PROCESSED_DIR,
    "patecon_conflicts.csv"
)

TEMPORAL_CONSTRAINTS_FINAL_OUT = os.path.join(
    PROCESSED_DIR,
    "temporal_constraints_final.csv"
)

RELATION_EDGES_AFTER_CHECK_OUT = os.path.join(
    PROCESSED_DIR,
    "relation_edges_after_check.csv"
)

DETECTED_CONFLICTS_OUT = os.path.join(
    PROCESSED_DIR,
    "detected_conflicts.csv"
)


# ============================================================
# 1. cameotop 分支关系类型定义
# ============================================================

CAMEO_TOP_RELATIONS = {f"{i:02d}" for i in range(1, 21)}

# 01-05：言语合作
VERBAL_COOPERATION = {"01", "02", "03", "04", "05"}

# 06-08：物质合作
MATERIAL_COOPERATION = {"06", "07", "08"}

# 09-14：言语冲突
VERBAL_CONFLICT = {"09", "10", "11", "12", "13", "14"}

# 15-20：物质冲突
MATERIAL_CONFLICT = {"15", "16", "17", "18", "19", "20"}

COOPERATION_RELATIONS = VERBAL_COOPERATION | MATERIAL_COOPERATION
CONFLICT_RELATIONS = VERBAL_CONFLICT | MATERIAL_CONFLICT


CAMEO_TOP_DESC = {
    "01": "Make public statement",
    "02": "Appeal",
    "03": "Express intent to cooperate",
    "04": "Consult",
    "05": "Engage in diplomatic cooperation",
    "06": "Engage in material cooperation",
    "07": "Provide aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit force posture",
    "16": "Reduce relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Use unconventional mass violence",
}


REQUIRED_FINAL_CONSTRAINT_COLUMNS = [
    "constraint_id",
    "constraint_name",
    "relation_a",
    "relation_b",
    "temporal_predicate",
    "hard_or_soft",
    "expected_action",
    "reason",
    "source",
    "enabled",
    "support_level",
    "final_decision",
]


DETECTED_CONFLICT_COLUMNS = [
    "conflict_id",
    "edge_id",
    "subject_org_id",
    "object_org_id",
    "event_month",
    "relation_type",
    "triggered_constraint",
    "conflict_type",
    "hard_or_soft",
    "original_confidence",
    "action",
    "reason",
]


# ============================================================
# 2. 基础工具函数
# ============================================================

def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        text = str(x).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        text = str(x).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_cameo_relation(x):
    """
    cameotop 分支 relation_type 标准化。

    输入可能是：
        1
        01
        01.0
        CAMEO_01

    输出：
        01 ~ 20

    无法识别则返回空字符串。
    """
    text = normalize_text(x)

    if not text:
        return ""

    if "." in text:
        text = text.split(".")[0]

    text = re.sub(r"\D", "", text)

    if not text:
        return ""

    text = text.zfill(2)

    if text in CAMEO_TOP_RELATIONS:
        return text

    return ""


def normalize_relation(x):
    """
    用于约束文件 relation_a / relation_b 的统一处理。

    LLM 或 PaTeCon 可能输出：
        01
        1
        CAMEO_01
        any
        none

    这里统一成：
        01~20 / any / none
    """
    text = normalize_text(x)

    if not text:
        return "none"

    lowered = text.lower()

    if lowered in {"any", "*", "all"}:
        return "any"

    if lowered in {"none", "null", "nan"}:
        return "none"

    rel = normalize_cameo_relation(text)

    if rel:
        return rel

    return text


def is_cooperation_relation(r):
    r = normalize_cameo_relation(r)
    return r in COOPERATION_RELATIONS


def is_conflict_relation(r):
    r = normalize_cameo_relation(r)
    return r in CONFLICT_RELATIONS


def is_material_cooperation_relation(r):
    r = normalize_cameo_relation(r)
    return r in MATERIAL_COOPERATION


def is_material_conflict_relation(r):
    r = normalize_cameo_relation(r)
    return r in MATERIAL_CONFLICT


def get_relation_desc(r):
    r = normalize_cameo_relation(r)
    return CAMEO_TOP_DESC.get(r, "")


def ensure_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


# ============================================================
# 3. 读取 LLM / PaTeCon 约束
# ============================================================

def load_llm_constraints():
    """
    读取 LLM 生成的候选约束。
    优先读取 temporal_constraints_llm.csv。
    如果不存在，则尝试读取 llm_constraints.json。
    """
    if os.path.exists(LLM_CONSTRAINTS_CSV):
        df = pd.read_csv(LLM_CONSTRAINTS_CSV, dtype=str, low_memory=False).fillna("")
        return df

    if os.path.exists(LLM_CONSTRAINTS_JSON):
        with open(LLM_CONSTRAINTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data).fillna("")

    print("未找到 LLM 约束文件，将使用空约束。")

    return pd.DataFrame(columns=[
        "constraint_name",
        "relation_a",
        "relation_b",
        "temporal_predicate",
        "hard_or_soft",
        "expected_action",
        "reason",
    ])


def load_patecon_constraints():
    """
    读取 PaTeCon 约束 CSV。
    该文件一般由 10_run_patecon.py 或转换脚本生成：
        data/processed/patecon_constraints.csv
    """
    if not os.path.exists(PATECON_CONSTRAINTS_FILE):
        print("未找到 PaTeCon 约束文件，将使用空 PaTeCon 约束。")
        return pd.DataFrame()

    try:
        df = pd.read_csv(PATECON_CONSTRAINTS_FILE, dtype=str, low_memory=False).fillna("")
        return df
    except Exception as e:
        print("读取 PaTeCon 约束失败:", e)
        return pd.DataFrame()


def load_patecon_conflicts():
    """
    读取 PaTeCon 冲突检测结果 CSV。
    该文件应由 Conflict_Detection.py 的结果转换而来。

    如果该文件为空，则 PaTeCon 冲突部分不会产生 detected_conflicts 记录。
    """
    if not os.path.exists(PATECON_CONFLICTS_FILE):
        print("未找到 PaTeCon 冲突文件，将仅根据关系边进行规则检测。")
        return pd.DataFrame()

    try:
        df = pd.read_csv(PATECON_CONFLICTS_FILE, dtype=str, low_memory=False).fillna("")
        return df
    except Exception as e:
        print("读取 PaTeCon 冲突文件失败:", e)
        return pd.DataFrame()


# ============================================================
# 4. 约束标准化
# ============================================================

def standardize_llm_constraints(llm_df):
    """
    将 LLM 约束统一成最终约束格式。

    cameotop 分支：
        relation_a / relation_b 应为 01~20。
    """
    rows = []

    if llm_df.empty:
        return pd.DataFrame(columns=REQUIRED_FINAL_CONSTRAINT_COLUMNS)

    for i, row in llm_df.iterrows():
        constraint_name = normalize_text(
            row.get("constraint_name", f"llm_constraint_{i + 1}")
        )

        relation_a = normalize_relation(row.get("relation_a", "any"))
        relation_b = normalize_relation(row.get("relation_b", "none"))

        temporal_predicate = normalize_text(
            row.get("temporal_predicate", "review")
        )

        hard_or_soft = normalize_text(
            row.get("hard_or_soft", "soft")
        ).lower()

        expected_action = normalize_text(
            row.get("expected_action", "review")
        )

        reason = normalize_text(row.get("reason", ""))

        if hard_or_soft not in {"hard", "soft"}:
            hard_or_soft = "soft"

        # LLM 给出的 hard 约束过强，统一降级为 soft
        if hard_or_soft == "hard":
            hard_or_soft = "soft"
            reason = reason + "；为避免误删真实复杂组织关系，该约束由 hard 调整为 soft。"

        if relation_a == "none":
            relation_a = "any"

        rows.append({
            "constraint_id": f"LC_{i + 1:04d}",
            "constraint_name": constraint_name,
            "relation_a": relation_a,
            "relation_b": relation_b,
            "temporal_predicate": temporal_predicate,
            "hard_or_soft": hard_or_soft,
            "expected_action": expected_action,
            "reason": reason,
            "source": "LLM",
            "enabled": True,
            "support_level": "semantic",
            "final_decision": "enabled_soft",
        })

    return pd.DataFrame(rows)


def extract_cameo_relations_from_text(text):
    """
    从 PaTeCon 原始约束文本中提取 CAMEO 顶层码。

    支持：
        01
        02
        ...
        20

    注意：
        为避免从日期 20230101 中误抽取过多数字，
        这里优先匹配逗号分隔位置中的 property。
    """
    text = normalize_text(text)

    found = []

    # PaTeCon 原始格式常见：
    # a,13,b,t1,t2 before a,18,b,t3,t4|0.9
    # 因此匹配逗号附近的两位关系码
    pattern = r"(?<!\d)(0[1-9]|1[0-9]|20)(?!\d)"
    for m in re.finditer(pattern, text):
        rel = m.group(1)
        if rel in CAMEO_TOP_RELATIONS:
            found.append(rel)

    # 保留顺序去重
    result = []
    for r in found:
        if r not in result:
            result.append(r)

    return result


def standardize_patecon_constraints(patecon_df, start_index=1):
    """
    将 PaTeCon 约束标准化为最终约束格式。

    如果能从 raw_constraint / constraint_body / constraint_head 中解析出 01~20，
    则写入 relation_a / relation_b；
    否则标记为 any / none。
    """
    rows = []

    if patecon_df.empty:
        return pd.DataFrame(columns=REQUIRED_FINAL_CONSTRAINT_COLUMNS)

    for i, row in patecon_df.iterrows():
        row_text = " ".join([normalize_text(v) for v in row.values])

        relations = extract_cameo_relations_from_text(row_text)

        if len(relations) >= 2:
            relation_a = relations[0]
            relation_b = relations[1]
        elif len(relations) == 1:
            relation_a = relations[0]
            relation_b = "none"
        else:
            relation_a = "any"
            relation_b = "none"

        temporal_predicate = normalize_text(row.get("temporal_predicate", ""))

        if not temporal_predicate:
            temporal_predicate = "patecon_mined"

        confidence = normalize_text(row.get("confidence", ""))

        reason = f"PaTeCon 挖掘出的数据约束，原始内容: {row_text[:300]}"

        if confidence:
            reason += f"；confidence={confidence}"

        rows.append({
            "constraint_id": f"PC_{start_index + i:04d}",
            "constraint_name": f"patecon_constraint_{i + 1}",
            "relation_a": relation_a,
            "relation_b": relation_b,
            "temporal_predicate": temporal_predicate,
            "hard_or_soft": "soft",
            "expected_action": "review",
            "reason": reason,
            "source": "PaTeCon",
            "enabled": True,
            "support_level": "data",
            "final_decision": "enabled_soft",
        })

    return pd.DataFrame(rows)


def merge_constraints(llm_constraints, patecon_constraints):
    """
    融合 LLM 约束和 PaTeCon 约束。

    规则：
        1. LLM 与 PaTeCon 关系类型一致：semantic_and_data；
        2. LLM 无 PaTeCon 支持：semantic_only；
        3. PaTeCon 独有：data_only；
        4. 所有约束默认 soft，避免误删。
    """
    all_constraints = []

    llm_constraints = llm_constraints.copy()
    patecon_constraints = patecon_constraints.copy()

    patecon_pairs = set()

    for _, row in patecon_constraints.iterrows():
        a = normalize_relation(row.get("relation_a", "any"))
        b = normalize_relation(row.get("relation_b", "none"))

        patecon_pairs.add((a, b))
        patecon_pairs.add((b, a))

    for _, row in llm_constraints.iterrows():
        item = row.to_dict()

        a = normalize_relation(item.get("relation_a", "any"))
        b = normalize_relation(item.get("relation_b", "none"))

        item["relation_a"] = a
        item["relation_b"] = b
        item["hard_or_soft"] = "soft"
        item["enabled"] = True

        if (a, b) in patecon_pairs:
            item["source"] = "LLM+PaTeCon"
            item["support_level"] = "semantic_and_data"
        else:
            item["support_level"] = "semantic_only"

        item["final_decision"] = "enabled_soft"

        all_constraints.append(item)

    existing_pairs = set()

    for item in all_constraints:
        a = normalize_relation(item.get("relation_a", "any"))
        b = normalize_relation(item.get("relation_b", "none"))
        existing_pairs.add((a, b))
        existing_pairs.add((b, a))

    for _, row in patecon_constraints.iterrows():
        a = normalize_relation(row.get("relation_a", "any"))
        b = normalize_relation(row.get("relation_b", "none"))

        if (a, b) not in existing_pairs:
            item = row.to_dict()
            item["relation_a"] = a
            item["relation_b"] = b
            item["source"] = "PaTeCon"
            item["support_level"] = "data_only"
            item["hard_or_soft"] = "soft"
            item["expected_action"] = "review"
            item["enabled"] = True
            item["final_decision"] = "enabled_soft"
            all_constraints.append(item)

    final_df = pd.DataFrame(all_constraints)

    if final_df.empty:
        final_df = pd.DataFrame(columns=REQUIRED_FINAL_CONSTRAINT_COLUMNS)

    final_df = ensure_columns(final_df, REQUIRED_FINAL_CONSTRAINT_COLUMNS)
    final_df = final_df[REQUIRED_FINAL_CONSTRAINT_COLUMNS]

    return final_df


# ============================================================
# 5. cameotop 图谱校验规则
# ============================================================

def mark_final_relation_for_group(group):
    """
    对同一 subject-object-month 下的 CAMEO 顶层关系进行检查。

    主要识别：
        1. 物质合作与物质冲突同月并存；
        2. 言语合作与物质冲突同月并存；
        3. 合作类与冲突类同月并存；
        4. 多个冲突类事件同月并存；
        5. 多个 CAMEO 顶层关系同月并存。
    """
    relation_set = set()

    for v in group["relation_type"].astype(str).tolist():
        r = normalize_cameo_relation(v)
        if r:
            relation_set.add(r)

    if not relation_set:
        return ""

    has_cooperation = bool(relation_set & COOPERATION_RELATIONS)
    has_conflict = bool(relation_set & CONFLICT_RELATIONS)

    if relation_set & MATERIAL_COOPERATION and relation_set & MATERIAL_CONFLICT:
        return "material_cooperation_and_material_conflict_same_month"

    if relation_set & VERBAL_COOPERATION and relation_set & MATERIAL_CONFLICT:
        return "verbal_cooperation_and_material_conflict_same_month"

    if has_cooperation and has_conflict:
        return "cooperation_conflict_same_month"

    if len(relation_set & MATERIAL_CONFLICT) >= 2:
        return "multiple_material_conflict_events_same_month"

    if len(relation_set & CONFLICT_RELATIONS) >= 3:
        return "multiple_conflict_events_same_month"

    if len(relation_set) >= 3:
        return "multiple_cameo_relations_same_month"

    return ""


def choose_group_action(group, reason_code):
    """
    对同月多关系并存的处理策略。

    cameotop 分支不再新增 mixed_relation。
    只做：
        - mark_complex：高置信多关系，保留但标记复杂关系；
        - downgrade：低/中置信多关系，轻微降权复核。
    """
    high_conf_count = len(
        group[group["confidence"].apply(safe_float) >= 0.75]
    )

    if reason_code in {
        "material_cooperation_and_material_conflict_same_month",
        "verbal_cooperation_and_material_conflict_same_month",
        "cooperation_conflict_same_month",
    }:
        if high_conf_count >= 2:
            return (
                "mark_complex",
                "complex_relation",
                "同一组织对同月存在合作类与冲突类关系，且多条关系置信度较高，保留复杂性并标记复核。"
            )

        return (
            "downgrade",
            "soft_conflict",
            "同一组织对同月存在合作类与冲突类关系并存，按软约束降权复核。"
        )

    if high_conf_count >= 2:
        return (
            "mark_complex",
            "complex_relation",
            "同一组织对同月存在多种高置信 CAMEO 顶层关系，保留但标记为复杂关系。"
        )

    return (
        "downgrade",
        "soft_conflict",
        "同一组织对同月存在多种 CAMEO 顶层关系，按软约束降权复核。"
    )


def add_detected_row(
    detected_rows,
    edge_id="",
    subject_org_id="",
    object_org_id="",
    event_month="",
    relation_type="",
    triggered_constraint="",
    conflict_type="",
    hard_or_soft="soft",
    original_confidence="",
    action="review",
    reason="",
):
    detected_rows.append({
        "conflict_id": f"C_{len(detected_rows) + 1:06d}",
        "edge_id": edge_id,
        "subject_org_id": subject_org_id,
        "object_org_id": object_org_id,
        "event_month": event_month,
        "relation_type": relation_type,
        "triggered_constraint": triggered_constraint,
        "conflict_type": conflict_type,
        "hard_or_soft": hard_or_soft,
        "original_confidence": original_confidence,
        "action": action,
        "reason": reason,
    })


def apply_graph_update(edges, final_constraints, patecon_conflicts):
    """
    根据最终约束对关系边进行更新。

    注意：
        只更新派生关系边状态，不删除原始事件证据。
    """
    edges = edges.copy()

    # 避免 pandas / pyarrow 字符串列不能写入 float 的问题
    edges = edges.astype(object)

    edges = ensure_columns(edges, [
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
    ])

    # cameotop 分支：统一 relation_type 为 01~20
    edges["relation_type"] = edges["relation_type"].apply(normalize_cameo_relation)

    # 删除无法识别 relation_type 的边
    invalid_mask = edges["relation_type"] == ""
    invalid_edges = edges[invalid_mask].copy()

    edges = edges[~invalid_mask].copy()

    if "status" not in edges.columns:
        edges["status"] = "active"

    if "confidence" not in edges.columns:
        edges["confidence"] = 0.5

    if "confidence_level" not in edges.columns:
        edges["confidence_level"] = edges["confidence"].apply(
            lambda x: "high"
            if safe_float(x) >= 0.75
            else ("medium" if safe_float(x) >= 0.5 else "low")
        )

    edges["check_action"] = "keep"
    edges["check_reason"] = ""
    edges["triggered_constraints"] = ""

    detected_rows = []

    # 规则 0：记录无法识别 relation_type 的边
    for _, row in invalid_edges.iterrows():
        add_detected_row(
            detected_rows,
            edge_id=row.get("edge_id", ""),
            subject_org_id=row.get("subject_org_id", ""),
            object_org_id=row.get("object_org_id", ""),
            event_month=row.get("event_month", ""),
            relation_type=row.get("relation_type", ""),
            triggered_constraint="invalid_cameo_relation_type",
            conflict_type="invalid_relation_type",
            hard_or_soft="soft",
            original_confidence=row.get("confidence", ""),
            action="drop_from_checked_edges",
            reason="relation_type 无法标准化为 01~20，未写入更新后的关系边。"
        )

    # 规则一：GDELT 单源低证据关系 hide
    for idx, row in edges.iterrows():
        source_datasets = normalize_text(row.get("source_datasets", ""))
        event_count = safe_int(row.get("event_count", 0))
        gdelt_count = safe_int(row.get("gdelt_event_count", 0))
        icews_count = safe_int(row.get("icews_event_count", 0))
        confidence = safe_float(row.get("confidence", 0.0))

        if "GDELT" in source_datasets and "ICEWS" not in source_datasets:
            if event_count <= 2 or gdelt_count <= 2 or confidence < 0.45:
                edges.at[idx, "status"] = "hidden"
                edges.at[idx, "check_action"] = "hide"
                edges.at[idx, "check_reason"] = "GDELT 单源低证据关系，按软约束隐藏。"
                edges.at[idx, "triggered_constraints"] = "gdelt_single_source_low_event_count"

                add_detected_row(
                    detected_rows,
                    edge_id=row.get("edge_id", ""),
                    subject_org_id=row.get("subject_org_id", ""),
                    object_org_id=row.get("object_org_id", ""),
                    event_month=row.get("event_month", ""),
                    relation_type=row.get("relation_type", ""),
                    triggered_constraint="gdelt_single_source_low_event_count",
                    conflict_type="low_evidence",
                    hard_or_soft="soft",
                    original_confidence=confidence,
                    action="hide",
                    reason="GDELT 单源且事件数量或置信度较低。"
                )

        # 规则二：跨源共同支持关系保留
        if icews_count > 0 and gdelt_count > 0:
            if edges.at[idx, "status"] == "active":
                edges.at[idx, "check_action"] = "keep"
                edges.at[idx, "check_reason"] = "ICEWS 与 GDELT 跨源共同支持，优先保留。"
                edges.at[idx, "triggered_constraints"] = "icews_gdelt_cross_source_supported"

    # 规则三：同月同组织对多 CAMEO 顶层关系并存
    group_cols = ["subject_org_id", "object_org_id", "event_month"]

    active_edges = edges[edges["status"] == "active"].copy()

    for group_key, group in active_edges.groupby(group_cols):
        reason_code = mark_final_relation_for_group(group)

        if not reason_code:
            continue

        subject_org_id, object_org_id, event_month = group_key

        action, conflict_type, reason = choose_group_action(group, reason_code)

        for idx, row in group.iterrows():
            old_conf = safe_float(row.get("confidence", 0.0))

            if action == "downgrade":
                new_conf = max(0.0, old_conf - 0.1)

                edges.at[idx, "confidence"] = round(new_conf, 4)
                edges.at[idx, "confidence_level"] = (
                    "high"
                    if new_conf >= 0.75
                    else ("medium" if new_conf >= 0.5 else "low")
                )
                edges.at[idx, "check_action"] = "downgrade"
                edges.at[idx, "check_reason"] = reason
                edges.at[idx, "triggered_constraints"] = reason_code

            elif action == "mark_complex":
                edges.at[idx, "check_action"] = "mark_complex"
                edges.at[idx, "check_reason"] = reason
                edges.at[idx, "triggered_constraints"] = reason_code

            add_detected_row(
                detected_rows,
                edge_id=row.get("edge_id", ""),
                subject_org_id=subject_org_id,
                object_org_id=object_org_id,
                event_month=event_month,
                relation_type=row.get("relation_type", ""),
                triggered_constraint=reason_code,
                conflict_type=conflict_type,
                hard_or_soft="soft",
                original_confidence=old_conf,
                action=action,
                reason=reason
            )

    # 规则四：将 PaTeCon 冲突结果记录进 detected_conflicts
    if not patecon_conflicts.empty:
        for _, row in patecon_conflicts.iterrows():
            row_text = " ".join([normalize_text(v) for v in row.values])

            subject = normalize_text(
                row.get("subject", row.get("subject_org_id", ""))
            )
            obj = normalize_text(
                row.get("object", row.get("object_org_id", ""))
            )
            relation = normalize_relation(
                row.get("relation", row.get("relation_type", ""))
            )

            add_detected_row(
                detected_rows,
                edge_id=normalize_text(row.get("edge_id", "")),
                subject_org_id=subject,
                object_org_id=obj,
                event_month=normalize_text(row.get("event_month", "")),
                relation_type=relation if relation not in {"any", "none"} else "",
                triggered_constraint="patecon_conflict",
                conflict_type=normalize_text(row.get("conflict_type", "patecon_detected")),
                hard_or_soft="soft",
                original_confidence="",
                action="review",
                reason=f"PaTeCon 检测到的冲突结果，原始内容: {row_text[:300]}"
            )

    detected_conflicts = pd.DataFrame(detected_rows)

    if detected_conflicts.empty:
        detected_conflicts = pd.DataFrame(columns=DETECTED_CONFLICT_COLUMNS)
    else:
        detected_conflicts = ensure_columns(detected_conflicts, DETECTED_CONFLICT_COLUMNS)
        detected_conflicts = detected_conflicts[DETECTED_CONFLICT_COLUMNS]

    return edges, detected_conflicts



# ============================================================
# 6. 主流程
# ============================================================

def main():
    if not os.path.exists(RELATION_EDGES_SCORED_FILE):
        print(f"未找到输入文件: {RELATION_EDGES_SCORED_FILE}")
        print("请先运行 scripts/07_score_relations.py")
        return

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("========== Step 11: Update Graph ==========")

    print("开始读取关系边...")
    edges = pd.read_csv(
        RELATION_EDGES_SCORED_FILE,
        dtype=str,
        low_memory=False
    ).fillna("")

    print("关系边数量:", len(edges))

    if "relation_type" in edges.columns:
        print("输入 relation_type 分布:")
        print(edges["relation_type"].astype(str).value_counts().head(30))

    print("\n开始读取 LLM 约束...")
    llm_raw = load_llm_constraints()
    llm_constraints = standardize_llm_constraints(llm_raw)

    print("LLM 约束数量:", len(llm_constraints))

    print("\n开始读取 PaTeCon 约束...")
    patecon_raw = load_patecon_constraints()
    patecon_constraints = standardize_patecon_constraints(
        patecon_raw,
        start_index=len(llm_constraints) + 1
    )

    print("PaTeCon 约束数量:", len(patecon_constraints))

    print("\n开始融合 LLM 约束与 PaTeCon 约束...")
    final_constraints = merge_constraints(
        llm_constraints,
        patecon_constraints
    )

    final_constraints.to_csv(
        TEMPORAL_CONSTRAINTS_FINAL_OUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("最终约束输出:", TEMPORAL_CONSTRAINTS_FINAL_OUT)
    print("最终约束数量:", len(final_constraints))

    print("\n开始读取 PaTeCon 冲突结果...")
    patecon_conflicts = load_patecon_conflicts()
    print("PaTeCon 冲突记录数量:", len(patecon_conflicts))

    print("\n开始进行图谱校验更新...")
    updated_edges, detected_conflicts = apply_graph_update(
        edges,
        final_constraints,
        patecon_conflicts
    )

    updated_edges.to_csv(
        RELATION_EDGES_AFTER_CHECK_OUT,
        index=False,
        encoding="utf-8-sig"
    )

    detected_conflicts.to_csv(
        DETECTED_CONFLICTS_OUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n图谱校验更新完成")
    print("输出:", RELATION_EDGES_AFTER_CHECK_OUT)
    print("输出:", DETECTED_CONFLICTS_OUT)
    print("更新后关系边数量:", len(updated_edges))
    print("检测到冲突/复核记录数量:", len(detected_conflicts))

    if "status" in updated_edges.columns:
        print("\n状态分布:")
        print(updated_edges["status"].value_counts())

    if "check_action" in updated_edges.columns:
        print("\n校验动作分布:")
        print(updated_edges["check_action"].value_counts())

    if not detected_conflicts.empty and "conflict_type" in detected_conflicts.columns:
        print("\n冲突类型分布:")
        print(detected_conflicts["conflict_type"].value_counts())

    print("\n========== Step 11 Finished ==========")


if __name__ == "__main__":
    main()
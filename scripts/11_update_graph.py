# scripts/11_update_graph.py
import os
import re
import json
import pandas as pd

PROCESSED_DIR = "../data/processed"

RELATION_EDGES_SCORED_FILE = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")

LLM_CONSTRAINTS_CSV = os.path.join(PROCESSED_DIR, "temporal_constraints_llm.csv")
LLM_CONSTRAINTS_JSON = os.path.join(PROCESSED_DIR, "llm_constraints.json")

PATECON_CONSTRAINTS_FILE = os.path.join(PROCESSED_DIR, "patecon_constraints.csv")
PATECON_CONFLICTS_FILE = os.path.join(PROCESSED_DIR, "patecon_conflicts.csv")

TEMPORAL_CONSTRAINTS_FINAL_OUT = os.path.join(PROCESSED_DIR, "temporal_constraints_final.csv")
RELATION_EDGES_AFTER_CHECK_OUT = os.path.join(PROCESSED_DIR, "relation_edges_after_check.csv")
DETECTED_CONFLICTS_OUT = os.path.join(PROCESSED_DIR, "detected_conflicts.csv")


RELATION_TYPES = [
    "verbal_cooperation",
    "material_cooperation",
    "verbal_conflict",
    "material_conflict",
    "mixed_relation"
]


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
    "final_decision"
]


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


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_relation(x):
    x = normalize_text(x)
    if not x:
        return "none"
    return x


def load_llm_constraints():
    """
    读取 LLM 生成的候选约束。
    优先读取 temporal_constraints_llm.csv。
    如果不存在，则尝试读取 llm_constraints.json。
    """

    if os.path.exists(LLM_CONSTRAINTS_CSV):
        df = pd.read_csv(LLM_CONSTRAINTS_CSV, low_memory=False)
        return df

    if os.path.exists(LLM_CONSTRAINTS_JSON):
        with open(LLM_CONSTRAINTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    print("未找到 LLM 约束文件，将使用空约束。")
    return pd.DataFrame(columns=[
        "constraint_name",
        "relation_a",
        "relation_b",
        "temporal_predicate",
        "hard_or_soft",
        "expected_action",
        "reason"
    ])


def load_patecon_constraints():
    """
    读取 PaTeCon 约束。
    由于不同 PaTeCon 版本输出格式不一致，这里做保守兼容。
    """

    if not os.path.exists(PATECON_CONSTRAINTS_FILE):
        print("未找到 PaTeCon 约束文件，将使用空 PaTeCon 约束。")
        return pd.DataFrame()

    try:
        df = pd.read_csv(PATECON_CONSTRAINTS_FILE, low_memory=False)
    except Exception:
        return pd.DataFrame()

    return df


def load_patecon_conflicts():
    """
    读取 PaTeCon 冲突检测结果。
    不同版本格式可能不同，因此后续只做保守解析。
    """

    if not os.path.exists(PATECON_CONFLICTS_FILE):
        print("未找到 PaTeCon 冲突文件，将仅根据 LLM 约束和关系边进行规则检测。")
        return pd.DataFrame()

    try:
        df = pd.read_csv(PATECON_CONFLICTS_FILE, low_memory=False)
    except Exception:
        return pd.DataFrame()

    return df


def standardize_llm_constraints(llm_df):
    """
    将 LLM 约束统一成最终约束格式。
    """

    rows = []

    if llm_df.empty:
        return pd.DataFrame(columns=REQUIRED_FINAL_CONSTRAINT_COLUMNS)

    for i, row in llm_df.iterrows():
        constraint_name = normalize_text(row.get("constraint_name", f"llm_constraint_{i + 1}"))

        relation_a = normalize_relation(row.get("relation_a", "any"))
        relation_b = normalize_relation(row.get("relation_b", "none"))

        temporal_predicate = normalize_text(row.get("temporal_predicate", "review"))
        hard_or_soft = normalize_text(row.get("hard_or_soft", "soft")).lower()
        expected_action = normalize_text(row.get("expected_action", "review"))
        reason = normalize_text(row.get("reason", ""))

        if hard_or_soft not in ["hard", "soft"]:
            hard_or_soft = "soft"

        # 避免 LLM 给出过强硬约束。
        # 当前项目中，合作/冲突并存不应直接认为错误，因此 hard 一般降为 soft。
        if hard_or_soft == "hard":
            if relation_a in RELATION_TYPES or relation_b in RELATION_TYPES:
                hard_or_soft = "soft"
                reason = reason + "；该约束由 hard 调整为 soft，以避免误删现实中的复杂组织关系。"

        # none 作为 relation_a 不太规范，修正为 any
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
            "final_decision": "enabled_soft" if hard_or_soft == "soft" else "enabled_hard"
        })

    return pd.DataFrame(rows)


def extract_relation_types_from_text(text):
    """
    从 PaTeCon 输出的原始文本中尝试提取关系类型。
    例如某些输出可能包含 verbal_conflict、material_conflict。
    """

    text = normalize_text(text)
    found = []

    for r in RELATION_TYPES:
        if r in text:
            found.append(r)

    return found


def standardize_patecon_constraints(patecon_df, start_index=1):
    """
    将 PaTeCon 约束尽量标准化。
    如果 PaTeCon 输出格式不清晰，则保留为数据约束候选，但默认 review。
    """

    rows = []

    if patecon_df.empty:
        return pd.DataFrame(columns=REQUIRED_FINAL_CONSTRAINT_COLUMNS)

    for i, row in patecon_df.iterrows():
        row_text = " ".join([normalize_text(v) for v in row.values])
        relations = extract_relation_types_from_text(row_text)

        if len(relations) >= 2:
            relation_a = relations[0]
            relation_b = relations[1]
        elif len(relations) == 1:
            relation_a = relations[0]
            relation_b = "none"
        else:
            relation_a = "any"
            relation_b = "none"

        rows.append({
            "constraint_id": f"PC_{start_index + i:04d}",
            "constraint_name": f"patecon_constraint_{i + 1}",
            "relation_a": relation_a,
            "relation_b": relation_b,
            "temporal_predicate": "patecon_mined",
            "hard_or_soft": "soft",
            "expected_action": "review",
            "reason": f"PaTeCon 挖掘出的数据约束，原始内容: {row_text[:300]}",
            "source": "PaTeCon",
            "enabled": True,
            "support_level": "data",
            "final_decision": "enabled_soft"
        })

    return pd.DataFrame(rows)


def merge_constraints(llm_constraints, patecon_constraints):
    """
    融合 LLM 约束和 PaTeCon 约束。

    处理规则：
    1. LLM 与 PaTeCon 关系类型一致：标记为 semantic_and_data。
    2. LLM 有道理但 PaTeCon 无支持：保留为 soft。
    3. PaTeCon 语义不清：保留为 review。
    4. 过强 hard 约束：禁用或降为 soft。
    """

    all_constraints = []

    llm_constraints = llm_constraints.copy()
    patecon_constraints = patecon_constraints.copy()

    # 标记 PaTeCon 支持
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

        if (a, b) in patecon_pairs:
            item["source"] = "LLM+PaTeCon"
            item["support_level"] = "semantic_and_data"
            item["final_decision"] = "enabled_soft" if item["hard_or_soft"] == "soft" else "enabled_hard"
        else:
            item["support_level"] = "semantic_only"
            item["final_decision"] = "enabled_soft"

        # 再次限制过强 hard
        if item["hard_or_soft"] == "hard":
            action = normalize_text(item.get("expected_action", "review"))
            if action in ["remove", "hide"]:
                item["enabled"] = False
                item["final_decision"] = "disabled_too_strong"
                item["reason"] = normalize_text(item.get("reason", "")) + "；该约束过于绝对，暂不启用。"
            else:
                item["hard_or_soft"] = "soft"
                item["final_decision"] = "enabled_soft"
                item["reason"] = normalize_text(item.get("reason", "")) + "；为避免误删，硬约束降为软约束。"

        all_constraints.append(item)

    # 加入 PaTeCon 独有约束
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

    for col in REQUIRED_FINAL_CONSTRAINT_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = ""

    final_df = final_df[REQUIRED_FINAL_CONSTRAINT_COLUMNS]

    return final_df


def mark_final_relation_for_group(group):
    """
    对同一 subject-object-month 下的多种关系进行检查。
    主要识别：
    1. material_conflict + material_cooperation
    2. verbal_cooperation + material_conflict
    3. 多类关系并存
    """

    relation_set = set(group["relation_type"].astype(str).tolist())

    has_conflict = bool(relation_set & {"verbal_conflict", "material_conflict"})
    has_cooperation = bool(relation_set & {"verbal_cooperation", "material_cooperation"})

    if "material_conflict" in relation_set and "material_cooperation" in relation_set:
        return "material_conflict_and_material_cooperation_same_month"

    if "verbal_cooperation" in relation_set and "material_conflict" in relation_set:
        return "verbal_cooperation_and_material_conflict_same_month"

    if has_conflict and has_cooperation:
        return "cooperation_conflict_same_month"

    if "verbal_conflict" in relation_set and "material_conflict" in relation_set:
        return "verbal_conflict_and_material_conflict_same_month"

    if "verbal_cooperation" in relation_set and "material_cooperation" in relation_set:
        return "verbal_cooperation_and_material_cooperation_same_month"

    return ""


def apply_graph_update(edges, final_constraints, patecon_conflicts):
    """
    根据最终约束对关系边进行更新。

    注意：
    只更新派生关系边状态，不删除原始事件证据。
    """

    edges = edges.copy()

    if "status" not in edges.columns:
        edges["status"] = "active"

    if "confidence" not in edges.columns:
        edges["confidence"] = 0.5

    if "confidence_level" not in edges.columns:
        edges["confidence_level"] = edges["confidence"].apply(
            lambda x: "high" if safe_float(x) >= 0.75 else ("medium" if safe_float(x) >= 0.5 else "low")
        )

    edges["check_action"] = "keep"
    edges["check_reason"] = ""
    edges["triggered_constraints"] = ""

    detected_rows = []

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

                detected_rows.append({
                    "conflict_id": f"C_{len(detected_rows) + 1:06d}",
                    "edge_id": row.get("edge_id", ""),
                    "subject_org_id": row.get("subject_org_id", ""),
                    "object_org_id": row.get("object_org_id", ""),
                    "event_month": row.get("event_month", ""),
                    "relation_type": row.get("relation_type", ""),
                    "triggered_constraint": "gdelt_single_source_low_event_count",
                    "conflict_type": "low_evidence",
                    "hard_or_soft": "soft",
                    "original_confidence": confidence,
                    "action": "hide",
                    "reason": "GDELT 单源且事件数量或置信度较低。"
                })

        # 规则二：跨源共同支持关系保留
        if icews_count > 0 and gdelt_count > 0:
            if edges.at[idx, "status"] == "active":
                edges.at[idx, "check_action"] = "keep"
                edges.at[idx, "check_reason"] = "ICEWS 与 GDELT 跨源共同支持，优先保留。"
                edges.at[idx, "triggered_constraints"] = "icews_gdelt_cross_source_supported"

    # 规则三：同月同组织对多关系并存，标记 mixed 或 downgrade
    group_cols = ["subject_org_id", "object_org_id", "event_month"]

    for group_key, group in edges[edges["status"] == "active"].groupby(group_cols):
        reason_code = mark_final_relation_for_group(group)

        if not reason_code:
            continue

        subject_org_id, object_org_id, event_month = group_key

        # 如果该组已经有 mixed_relation，则保留 mixed_relation，其他关系降权复核
        has_mixed = "mixed_relation" in set(group["relation_type"].astype(str).tolist())

        # 计算组内是否有高置信关系
        high_conf_count = len(group[group["confidence"].apply(safe_float) >= 0.75])

        if high_conf_count >= 2:
            action = "mark_mixed"
            conflict_type = "complex_relation"
            reason = "同一组织对同月存在多种高置信关系，保留复杂性并标记 mixed_relation。"
        else:
            action = "downgrade"
            conflict_type = "soft_conflict"
            reason = "同一组织对同月存在合作/冲突或多关系并存，按软约束降权复核。"

        for idx, row in group.iterrows():
            old_conf = safe_float(row.get("confidence", 0.0))

            if action == "downgrade":
                new_conf = max(0.0, old_conf - 0.1)
                edges.at[idx, "confidence"] = round(new_conf, 4)
                edges.at[idx, "check_action"] = "downgrade"
                edges.at[idx, "check_reason"] = reason
                edges.at[idx, "triggered_constraints"] = reason_code

            elif action == "mark_mixed":
                edges.at[idx, "check_action"] = "mark_mixed"
                edges.at[idx, "check_reason"] = reason
                edges.at[idx, "triggered_constraints"] = reason_code

            detected_rows.append({
                "conflict_id": f"C_{len(detected_rows) + 1:06d}",
                "edge_id": row.get("edge_id", ""),
                "subject_org_id": subject_org_id,
                "object_org_id": object_org_id,
                "event_month": event_month,
                "relation_type": row.get("relation_type", ""),
                "triggered_constraint": reason_code,
                "conflict_type": conflict_type,
                "hard_or_soft": "soft",
                "original_confidence": old_conf,
                "action": action,
                "reason": reason
            })

        # 如果没有 mixed_relation，则新增一条 mixed_relation 派生边
        if action == "mark_mixed" and not has_mixed:
            first_row = group.iloc[0].to_dict()
            new_edge_id = f"MIX_{len(edges) + 1:09d}"

            mixed_row = first_row.copy()
            mixed_row["edge_id"] = new_edge_id
            mixed_row["relation_type"] = "mixed_relation"
            mixed_row["event_count"] = group["event_count"].apply(safe_int).sum()
            mixed_row["icews_event_count"] = group["icews_event_count"].apply(safe_int).sum() if "icews_event_count" in group.columns else 0
            mixed_row["gdelt_event_count"] = group["gdelt_event_count"].apply(safe_int).sum() if "gdelt_event_count" in group.columns else 0
            mixed_row["source_datasets"] = ";".join(sorted(set(";".join(group["source_datasets"].astype(str)).split(";"))))
            mixed_row["confidence"] = round(min(1.0, group["confidence"].apply(safe_float).mean() + 0.05), 4)
            mixed_row["confidence_level"] = "high" if mixed_row["confidence"] >= 0.75 else "medium"
            mixed_row["status"] = "active"
            mixed_row["check_action"] = "create_mixed_relation"
            mixed_row["check_reason"] = "由同月多关系并存触发，新增 mixed_relation 派生边。"
            mixed_row["triggered_constraints"] = reason_code

            edges = pd.concat([edges, pd.DataFrame([mixed_row])], ignore_index=True)

            detected_rows.append({
                "conflict_id": f"C_{len(detected_rows) + 1:06d}",
                "edge_id": new_edge_id,
                "subject_org_id": subject_org_id,
                "object_org_id": object_org_id,
                "event_month": event_month,
                "relation_type": "mixed_relation",
                "triggered_constraint": reason_code,
                "conflict_type": "create_mixed_relation",
                "hard_or_soft": "soft",
                "original_confidence": mixed_row["confidence"],
                "action": "create_mixed_relation",
                "reason": "新增 mixed_relation 以表示复杂组织关系。"
            })

    # 规则四：将 PaTeCon 冲突结果记录进 detected_conflicts
    if not patecon_conflicts.empty:
        for _, row in patecon_conflicts.iterrows():
            row_text = " ".join([normalize_text(v) for v in row.values])
            detected_rows.append({
                "conflict_id": f"C_{len(detected_rows) + 1:06d}",
                "edge_id": "",
                "subject_org_id": "",
                "object_org_id": "",
                "event_month": "",
                "relation_type": "",
                "triggered_constraint": "patecon_conflict",
                "conflict_type": "patecon_detected",
                "hard_or_soft": "soft",
                "original_confidence": "",
                "action": "review",
                "reason": f"PaTeCon 检测到的冲突结果，原始内容: {row_text[:300]}"
            })

    detected_conflicts = pd.DataFrame(detected_rows)

    if detected_conflicts.empty:
        detected_conflicts = pd.DataFrame(columns=[
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
            "reason"
        ])

    return edges, detected_conflicts


def main():
    if not os.path.exists(RELATION_EDGES_SCORED_FILE):
        print(f"未找到输入文件: {RELATION_EDGES_SCORED_FILE}")
        print("请先运行 scripts/07_score_relations.py")
        return

    print("开始读取关系边...")
    edges = pd.read_csv(RELATION_EDGES_SCORED_FILE, low_memory=False)

    print("开始读取 LLM 约束...")
    llm_raw = load_llm_constraints()
    llm_constraints = standardize_llm_constraints(llm_raw)

    print("开始读取 PaTeCon 约束...")
    patecon_raw = load_patecon_constraints()
    patecon_constraints = standardize_patecon_constraints(
        patecon_raw,
        start_index=len(llm_constraints) + 1
    )

    print("开始融合 LLM 约束与 PaTeCon 约束...")
    final_constraints = merge_constraints(llm_constraints, patecon_constraints)

    final_constraints.to_csv(
        TEMPORAL_CONSTRAINTS_FINAL_OUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("最终约束输出:", TEMPORAL_CONSTRAINTS_FINAL_OUT)
    print("最终约束数量:", len(final_constraints))

    print("开始读取 PaTeCon 冲突结果...")
    patecon_conflicts = load_patecon_conflicts()

    print("开始进行图谱校验更新...")
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

    print("图谱校验更新完成")
    print("输出:", RELATION_EDGES_AFTER_CHECK_OUT)
    print("输出:", DETECTED_CONFLICTS_OUT)
    print("更新后关系边数量:", len(updated_edges))
    print("检测到冲突/复核记录数量:", len(detected_conflicts))

    if "status" in updated_edges.columns:
        print("状态分布:")
        print(updated_edges["status"].value_counts())


if __name__ == "__main__":
    main()
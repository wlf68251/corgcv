# scripts/12_evaluate.py
import os
import pandas as pd

PROCESSED_DIR = "../data/processed"

OUTPUTS_DIR = "../outputs"
EVAL_DIR = os.path.join(OUTPUTS_DIR, "evaluation_tables")
CASE_DIR = os.path.join(OUTPUTS_DIR, "case_studies")

ORGANIZATIONS_FILE = os.path.join(PROCESSED_DIR, "organizations.csv")
ALIASES_FILE = os.path.join(PROCESSED_DIR, "organization_aliases.csv")

RELATION_EDGES_BEFORE_FILE = os.path.join(PROCESSED_DIR, "relation_edges_before_check.csv")
RELATION_EDGES_SCORED_FILE = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")
RELATION_EDGES_AFTER_FILE = os.path.join(PROCESSED_DIR, "relation_edges_after_check.csv")

TEMPORAL_CONSTRAINTS_FINAL_FILE = os.path.join(PROCESSED_DIR, "temporal_constraints_final.csv")
LLM_CONSTRAINTS_FILE = os.path.join(PROCESSED_DIR, "temporal_constraints_llm.csv")
PATECON_CONSTRAINTS_FILE = os.path.join(PROCESSED_DIR, "patecon_constraints.csv")
DETECTED_CONFLICTS_FILE = os.path.join(PROCESSED_DIR, "detected_conflicts.csv")

SUMMARY_OUT = os.path.join(EVAL_DIR, "evaluation_summary.csv")
ORG_SAMPLE_OUT = os.path.join(EVAL_DIR, "organization_alignment_sample.csv")
CONSTRAINT_SAMPLE_OUT = os.path.join(EVAL_DIR, "constraint_quality_sample.csv")
CONFLICT_PRECISION_SAMPLE_OUT = os.path.join(EVAL_DIR, "conflict_precision_at_50_sample.csv")
REMOVED_HIDDEN_SAMPLE_OUT = os.path.join(EVAL_DIR, "removed_hidden_review_sample.csv")

CASE_CROSS_SOURCE_OUT = os.path.join(CASE_DIR, "case_cross_source_supported.csv")
CASE_GDELT_LOW_EVIDENCE_OUT = os.path.join(CASE_DIR, "case_gdelt_single_source_low_evidence.csv")
CASE_LLM_STRONG_FILTERED_OUT = os.path.join(CASE_DIR, "case_llm_strong_constraint_filtered.csv")
CASE_MIXED_RELATION_OUT = os.path.join(CASE_DIR, "case_mixed_relation.csv")


def ensure_dirs():
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(CASE_DIR, exist_ok=True)


def read_csv_if_exists(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def safe_len(df):
    return 0 if df is None or df.empty else len(df)


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


def sample_df(df, n=100, random_state=42):
    if df.empty:
        return df
    return df.sample(n=min(n, len(df)), random_state=random_state)


def evaluate_organization_scale(orgs, aliases):
    org_count = safe_len(orgs)
    alias_count = safe_len(aliases)

    if not aliases.empty and "org_id" in aliases.columns:
        org_with_alias_count = aliases["org_id"].nunique()
    else:
        org_with_alias_count = 0

    return {
        "organization_count": org_count,
        "alias_count": alias_count,
        "organizations_with_alias_count": org_with_alias_count
    }


def evaluate_fusion(edges_before, edges_scored, edges_after):
    if edges_before.empty:
        return {
            "edges_before_check": 0,
            "edges_scored": safe_len(edges_scored),
            "edges_after_check": safe_len(edges_after),
            "gdelt_only_edges": 0,
            "icews_only_edges": 0,
            "cross_source_edges": 0,
            "hidden_edges": 0,
            "active_edges": 0,
            "mixed_relation_edges": 0
        }

    source_col = "source_datasets"

    if source_col in edges_before.columns:
        sources = edges_before[source_col].fillna("").astype(str)
        gdelt_only_edges = len(edges_before[(sources.str.contains("GDELT")) & (~sources.str.contains("ICEWS"))])
        icews_only_edges = len(edges_before[(sources.str.contains("ICEWS")) & (~sources.str.contains("GDELT"))])
        cross_source_edges = len(edges_before[(sources.str.contains("ICEWS")) & (sources.str.contains("GDELT"))])
    else:
        gdelt_only_edges = 0
        icews_only_edges = 0
        cross_source_edges = 0

    if not edges_after.empty and "status" in edges_after.columns:
        hidden_edges = len(edges_after[edges_after["status"] == "hidden"])
        active_edges = len(edges_after[edges_after["status"] == "active"])
    else:
        hidden_edges = 0
        active_edges = 0

    if not edges_after.empty and "relation_type" in edges_after.columns:
        mixed_relation_edges = len(edges_after[edges_after["relation_type"] == "mixed_relation"])
    else:
        mixed_relation_edges = 0

    return {
        "edges_before_check": len(edges_before),
        "edges_scored": safe_len(edges_scored),
        "edges_after_check": safe_len(edges_after),
        "gdelt_only_edges": gdelt_only_edges,
        "icews_only_edges": icews_only_edges,
        "cross_source_edges": cross_source_edges,
        "hidden_edges": hidden_edges,
        "active_edges": active_edges,
        "mixed_relation_edges": mixed_relation_edges
    }


def evaluate_constraints(final_constraints, llm_constraints, patecon_constraints):
    total_final = safe_len(final_constraints)
    total_llm = safe_len(llm_constraints)
    total_patecon = safe_len(patecon_constraints)

    if not final_constraints.empty and "source" in final_constraints.columns:
        source_dist = final_constraints["source"].fillna("").value_counts().to_dict()
    else:
        source_dist = {}

    if not final_constraints.empty and "hard_or_soft" in final_constraints.columns:
        hard_soft_dist = final_constraints["hard_or_soft"].fillna("").value_counts().to_dict()
    else:
        hard_soft_dist = {}

    if not final_constraints.empty and "enabled" in final_constraints.columns:
        enabled_count = len(final_constraints[final_constraints["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])])
    else:
        enabled_count = 0

    return {
        "llm_constraints": total_llm,
        "patecon_constraints": total_patecon,
        "final_constraints": total_final,
        "enabled_constraints": enabled_count,
        "constraint_source_distribution": str(source_dist),
        "hard_soft_distribution": str(hard_soft_dist)
    }


def evaluate_conflicts(conflicts):
    conflict_count = safe_len(conflicts)

    if conflicts.empty:
        return {
            "detected_conflicts": 0,
            "hidden_conflicts": 0,
            "downgrade_conflicts": 0,
            "mark_mixed_conflicts": 0,
            "review_conflicts": 0,
            "conflict_precision_at_50_manual_field": "need_manual_annotation"
        }

    if "action" in conflicts.columns:
        actions = conflicts["action"].fillna("").astype(str)
        hidden_conflicts = len(conflicts[actions == "hide"])
        downgrade_conflicts = len(conflicts[actions == "downgrade"])
        mark_mixed_conflicts = len(conflicts[actions.str.contains("mixed", case=False, regex=False)])
        review_conflicts = len(conflicts[actions == "review"])
    else:
        hidden_conflicts = 0
        downgrade_conflicts = 0
        mark_mixed_conflicts = 0
        review_conflicts = 0

    return {
        "detected_conflicts": conflict_count,
        "hidden_conflicts": hidden_conflicts,
        "downgrade_conflicts": downgrade_conflicts,
        "mark_mixed_conflicts": mark_mixed_conflicts,
        "review_conflicts": review_conflicts,
        "conflict_precision_at_50_manual_field": "need_manual_annotation"
    }


def write_organization_alignment_sample(orgs, aliases):
    if orgs.empty:
        pd.DataFrame().to_csv(ORG_SAMPLE_OUT, index=False, encoding="utf-8-sig")
        return

    sample = sample_df(orgs, n=100)

    if not aliases.empty and "org_id" in aliases.columns:
        alias_group = aliases.groupby("org_id")["alias"].apply(lambda x: "; ".join(x.astype(str).head(5))).reset_index()
        alias_group = alias_group.rename(columns={"alias": "sample_aliases"})
        sample = sample.merge(alias_group, on="org_id", how="left")
    else:
        sample["sample_aliases"] = ""

    sample["manual_is_correct_alignment"] = ""
    sample["manual_note"] = ""

    sample.to_csv(ORG_SAMPLE_OUT, index=False, encoding="utf-8-sig")


def write_constraint_quality_sample(final_constraints):
    if final_constraints.empty:
        pd.DataFrame().to_csv(CONSTRAINT_SAMPLE_OUT, index=False, encoding="utf-8-sig")
        return

    sample = sample_df(final_constraints, n=100)
    sample["manual_is_reasonable"] = ""
    sample["manual_should_enable"] = ""
    sample["manual_note"] = ""

    sample.to_csv(CONSTRAINT_SAMPLE_OUT, index=False, encoding="utf-8-sig")


def write_conflict_precision_sample(conflicts):
    if conflicts.empty:
        pd.DataFrame().to_csv(CONFLICT_PRECISION_SAMPLE_OUT, index=False, encoding="utf-8-sig")
        return

    sample = sample_df(conflicts, n=50)
    sample["manual_is_true_conflict"] = ""
    sample["manual_note"] = ""

    sample.to_csv(CONFLICT_PRECISION_SAMPLE_OUT, index=False, encoding="utf-8-sig")


def write_removed_hidden_sample(edges_after):
    if edges_after.empty:
        pd.DataFrame().to_csv(REMOVED_HIDDEN_SAMPLE_OUT, index=False, encoding="utf-8-sig")
        return

    if "status" in edges_after.columns:
        target = edges_after[edges_after["status"].isin(["hidden", "removed"])]
    else:
        target = pd.DataFrame()

    if target.empty and "check_action" in edges_after.columns:
        target = edges_after[edges_after["check_action"].isin(["hide", "remove"])]

    sample = sample_df(target, n=100)
    sample["manual_is_reasonable"] = ""
    sample["manual_note"] = ""

    sample.to_csv(REMOVED_HIDDEN_SAMPLE_OUT, index=False, encoding="utf-8-sig")


def write_case_studies(edges_before, edges_after, final_constraints):
    """
    输出四类 Case Study 候选：
    1. 多源共同支持关系
    2. GDELT 单源低证据关系
    3. LLM 过强约束被过滤
    4. 复杂关系改为 mixed_relation
    """

    # 1. 多源共同支持关系
    if not edges_before.empty and "source_datasets" in edges_before.columns:
        sources = edges_before["source_datasets"].fillna("").astype(str)
        cross_source = edges_before[
            sources.str.contains("ICEWS") &
            sources.str.contains("GDELT")
        ].copy()

        if "confidence" in cross_source.columns:
            cross_source = cross_source.sort_values(by="confidence", ascending=False)

        cross_source.head(30).to_csv(CASE_CROSS_SOURCE_OUT, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(CASE_CROSS_SOURCE_OUT, index=False, encoding="utf-8-sig")

    # 2. GDELT 单源低证据关系
    if not edges_after.empty and "source_datasets" in edges_after.columns:
        sources = edges_after["source_datasets"].fillna("").astype(str)
        gdelt_low = edges_after[
            sources.str.contains("GDELT") &
            (~sources.str.contains("ICEWS"))
        ].copy()

        if "event_count" in gdelt_low.columns:
            gdelt_low["event_count_num"] = gdelt_low["event_count"].apply(safe_int)
            gdelt_low = gdelt_low.sort_values(by="event_count_num", ascending=True)

        if "status" in gdelt_low.columns:
            hidden_first = gdelt_low[gdelt_low["status"] == "hidden"]
            if not hidden_first.empty:
                gdelt_low = hidden_first

        gdelt_low.head(30).to_csv(CASE_GDELT_LOW_EVIDENCE_OUT, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(CASE_GDELT_LOW_EVIDENCE_OUT, index=False, encoding="utf-8-sig")

    # 3. LLM 过强约束被过滤
    if not final_constraints.empty:
        if "final_decision" in final_constraints.columns:
            strong_filtered = final_constraints[
                final_constraints["final_decision"].astype(str).str.contains("disabled|too_strong", case=False, regex=True)
            ].copy()
        else:
            strong_filtered = pd.DataFrame()

        if strong_filtered.empty and "reason" in final_constraints.columns:
            strong_filtered = final_constraints[
                final_constraints["reason"].astype(str).str.contains("过强|误删|硬约束降为软约束|不启用", regex=True)
            ].copy()

        strong_filtered.head(30).to_csv(CASE_LLM_STRONG_FILTERED_OUT, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(CASE_LLM_STRONG_FILTERED_OUT, index=False, encoding="utf-8-sig")

    # 4. 复杂关系改为 mixed_relation
    if not edges_after.empty and "relation_type" in edges_after.columns:
        mixed = edges_after[edges_after["relation_type"] == "mixed_relation"].copy()

        if "check_action" in mixed.columns:
            mixed_created = mixed[
                mixed["check_action"].astype(str).str.contains("mixed", case=False, regex=False)
            ]
            if not mixed_created.empty:
                mixed = mixed_created

        mixed.head(30).to_csv(CASE_MIXED_RELATION_OUT, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(CASE_MIXED_RELATION_OUT, index=False, encoding="utf-8-sig")


def write_summary(summary_dict):
    rows = []
    for key, value in summary_dict.items():
        rows.append({
            "metric": key,
            "value": value
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")


def main():
    ensure_dirs()

    print("开始读取评估输入文件...")

    orgs = read_csv_if_exists(ORGANIZATIONS_FILE)
    aliases = read_csv_if_exists(ALIASES_FILE)

    edges_before = read_csv_if_exists(RELATION_EDGES_BEFORE_FILE)
    edges_scored = read_csv_if_exists(RELATION_EDGES_SCORED_FILE)
    edges_after = read_csv_if_exists(RELATION_EDGES_AFTER_FILE)

    final_constraints = read_csv_if_exists(TEMPORAL_CONSTRAINTS_FINAL_FILE)
    llm_constraints = read_csv_if_exists(LLM_CONSTRAINTS_FILE)
    patecon_constraints = read_csv_if_exists(PATECON_CONSTRAINTS_FILE)
    conflicts = read_csv_if_exists(DETECTED_CONFLICTS_FILE)

    print("开始计算评估指标...")

    summary = {}

    summary.update(evaluate_organization_scale(orgs, aliases))
    summary.update(evaluate_fusion(edges_before, edges_scored, edges_after))
    summary.update(evaluate_constraints(final_constraints, llm_constraints, patecon_constraints))
    summary.update(evaluate_conflicts(conflicts))

    write_summary(summary)

    print("开始生成抽样检查表...")
    write_organization_alignment_sample(orgs, aliases)
    write_constraint_quality_sample(final_constraints)
    write_conflict_precision_sample(conflicts)
    write_removed_hidden_sample(edges_after)

    print("开始生成 Case Study 候选表...")
    write_case_studies(edges_before, edges_after, final_constraints)

    print("评估完成")
    print("输出评估汇总:", SUMMARY_OUT)
    print("输出组织对齐抽样:", ORG_SAMPLE_OUT)
    print("输出约束质量抽样:", CONSTRAINT_SAMPLE_OUT)
    print("输出 Conflict Precision@50 抽样:", CONFLICT_PRECISION_SAMPLE_OUT)
    print("输出隐藏/删除合理性抽样:", REMOVED_HIDDEN_SAMPLE_OUT)
    print("输出 Case Study 目录:", CASE_DIR)

    print()
    print("评估指标汇总:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
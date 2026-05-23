#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convert_patecon_output_to_processed.py

用途：
    当你已经单独运行过 PaTeCon，并且结果已经生成在：

        project/patecon/output/

    或：

        project/PaTeCon-master/output/

    中时，使用本脚本把 PaTeCon 原始输出转换为第 10 步本应生成的两个文件：

        project/data/processed/patecon_constraints.csv
        project/data/processed/patecon_conflicts.csv

放置位置：
    建议把本文件放在：

        project/patecon/output/convert_patecon_output_to_processed.py

    或：

        project/PaTeCon-master/output/convert_patecon_output_to_processed.py

运行方式：
    在 output 目录下运行：

        python convert_patecon_output_to_processed.py

    或在项目根目录运行：

        python patecon/output/convert_patecon_output_to_processed.py

输出：
    ../../data/processed/patecon_constraints.csv
    ../../data/processed/patecon_conflicts.csv

说明：
    本脚本不会重新运行 PaTeCon，只转换已经存在的 output 结果。
"""

from pathlib import Path
import argparse
import pandas as pd


# ============================================================
# 1. 路径推断
# ============================================================

def find_project_root(start_path: Path) -> Path:
    """
    从当前脚本位置向上寻找项目根目录。

    判定依据：
        存在 data/processed 目录，或存在 data 目录。
    """
    start_path = start_path.resolve()

    for p in [start_path] + list(start_path.parents):
        if (p / "data" / "processed").exists():
            return p
        if (p / "data").exists():
            return p

    # 如果没有找到，默认认为 output 的上两级是项目根目录：
    # project/patecon/output/script.py
    # script.py -> output -> patecon -> project
    return start_path.parents[2]


def get_default_paths():
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent
    project_root = find_project_root(output_dir)

    processed_dir = project_root / "data" / "processed"

    return project_root, output_dir, processed_dir


# ============================================================
# 2. 自动寻找 PaTeCon 输出文件
# ============================================================

def find_first_file(output_dir: Path, patterns):
    """
    在 output 目录下按多个 pattern 查找第一个匹配文件。
    """
    for pattern in patterns:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def detect_patecon_files(output_dir: Path, dataset_stem: str = "org_relation_intervals"):
    """
    自动识别 PaTeCon 输出文件。

    常见文件：
        org_relation_intervals.all_constraints
        org_relation_intervals.temporal_representation_conflicts

    也兼容：
        *.all_constraints
        *.temporal_representation_conflicts
    """
    constraints_file = find_first_file(
        output_dir,
        [
            f"{dataset_stem}.all_constraints",
            "*.all_constraints",
        ]
    )

    conflicts_file = find_first_file(
        output_dir,
        [
            f"{dataset_stem}.temporal_representation_conflicts",
            "*.temporal_representation_conflicts",
        ]
    )

    return constraints_file, conflicts_file


# ============================================================
# 3. 解析 PaTeCon 约束文件
# ============================================================

def parse_constraint_line(raw_line: str):
    """
    解析 PaTeCon 的一行约束。

    常见格式示例：
        a,P22*P569,d,t5,t6 before a,P39,b,t1,t2|0.9916

    或：
        a,P569,b,t1,t2 MutualExclusion a,P569,c,t3,t4|0.9934

    本项目只保留后续 11 可用的基础字段：
        raw_constraint
        constraint_body
        temporal_predicate
        constraint_head
        confidence
    """
    raw = raw_line.strip()

    if "|" in raw:
        left, confidence = raw.rsplit("|", 1)
        confidence = confidence.strip()
    else:
        left = raw
        confidence = ""

    temporal_predicates = [
        "MutualExclusion",
        "before",
        "disjoint",
        "include",
        "start",
        "finish",
    ]

    constraint_body = left.strip()
    temporal_predicate = ""
    constraint_head = ""

    for pred in temporal_predicates:
        token = f" {pred} "
        if token in left:
            parts = left.split(token, 1)
            constraint_body = parts[0].strip()
            temporal_predicate = pred
            constraint_head = parts[1].strip()
            break

    return {
        "raw_constraint": raw,
        "constraint_body": constraint_body,
        "temporal_predicate": temporal_predicate,
        "constraint_head": constraint_head,
        "confidence": confidence,
    }


def convert_constraints_to_csv(constraints_file: Path, final_constraints_csv: Path):
    """
    将 PaTeCon 的 .all_constraints 转换为 patecon_constraints.csv。
    """
    rows = []

    if constraints_file is None or not constraints_file.exists():
        print("[WARN] 未找到 .all_constraints 文件，将生成空的 patecon_constraints.csv")

        df = pd.DataFrame(columns=[
            "constraint_id",
            "raw_constraint",
            "constraint_body",
            "temporal_predicate",
            "constraint_head",
            "confidence",
            "source",
            "source_file",
        ])

        df.to_csv(final_constraints_csv, index=False, encoding="utf-8-sig")
        return df

    with open(constraints_file, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            item = parse_constraint_line(raw)

            rows.append({
                "constraint_id": f"PC_{idx:06d}",
                "raw_constraint": item["raw_constraint"],
                "constraint_body": item["constraint_body"],
                "temporal_predicate": item["temporal_predicate"],
                "constraint_head": item["constraint_head"],
                "confidence": item["confidence"],
                "source": "patecon",
                "source_file": str(constraints_file),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=[
            "constraint_id",
            "raw_constraint",
            "constraint_body",
            "temporal_predicate",
            "constraint_head",
            "confidence",
            "source",
            "source_file",
        ])

    df.to_csv(final_constraints_csv, index=False, encoding="utf-8-sig")

    return df


# ============================================================
# 4. 解析 PaTeCon 时间表示冲突文件
# ============================================================

def split_conflict_line(raw: str):
    """
    PaTeCon 的 temporal_representation_conflicts 通常是逗号分隔：
        subject,property,object,start_time,end_time

    这里做宽松解析。
    """
    parts = [p.strip() for p in raw.split(",")]

    subject = parts[0] if len(parts) > 0 else ""
    relation = parts[1] if len(parts) > 1 else ""
    obj = parts[2] if len(parts) > 2 else ""
    start_time = parts[3] if len(parts) > 3 else ""
    end_time = parts[4] if len(parts) > 4 else ""

    return subject, relation, obj, start_time, end_time


def convert_conflicts_to_csv(conflicts_file: Path, final_conflicts_csv: Path):
    """
    将 PaTeCon 的 .temporal_representation_conflicts 转换为 patecon_conflicts.csv。

    如果 PaTeCon 没生成该文件，也生成一个空 CSV，保证 11_update_graph.py 能继续运行。
    """
    rows = []

    if conflicts_file is None or not conflicts_file.exists():
        print("[WARN] 未找到 .temporal_representation_conflicts 文件，将生成空的 patecon_conflicts.csv")

        df = pd.DataFrame(columns=[
            "conflict_id",
            "conflict_type",
            "raw_conflict",
            "constraint",
            "fact1",
            "fact2",
            "subject",
            "relation",
            "object",
            "start_time",
            "end_time",
            "source",
            "source_file",
        ])

        df.to_csv(final_conflicts_csv, index=False, encoding="utf-8-sig")
        return df

    with open(conflicts_file, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            subject, relation, obj, start_time, end_time = split_conflict_line(raw)

            rows.append({
                "conflict_id": f"PTC_{idx:06d}",
                "conflict_type": "temporal_representation",
                "raw_conflict": raw,
                "constraint": "start_time <= end_time",
                "fact1": raw,
                "fact2": "",
                "subject": subject,
                "relation": relation,
                "object": obj,
                "start_time": start_time,
                "end_time": end_time,
                "source": "patecon_temporal_representation",
                "source_file": str(conflicts_file),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=[
            "conflict_id",
            "conflict_type",
            "raw_conflict",
            "constraint",
            "fact1",
            "fact2",
            "subject",
            "relation",
            "object",
            "start_time",
            "end_time",
            "source",
            "source_file",
        ])

    df.to_csv(final_conflicts_csv, index=False, encoding="utf-8-sig")

    return df


# ============================================================
# 5. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert existing PaTeCon output files to data/processed CSV files."
    )

    parser.add_argument(
        "--dataset-stem",
        default="org_relation_intervals",
        help="PaTeCon 输出文件名前缀，默认 org_relation_intervals"
    )

    parser.add_argument(
        "--constraints-file",
        default="",
        help="手动指定 .all_constraints 文件路径"
    )

    parser.add_argument(
        "--conflicts-file",
        default="",
        help="手动指定 .temporal_representation_conflicts 文件路径"
    )

    parser.add_argument(
        "--processed-dir",
        default="",
        help="手动指定 data/processed 目录"
    )

    args = parser.parse_args()

    project_root, output_dir, processed_dir = get_default_paths()

    if args.processed_dir:
        processed_dir = Path(args.processed_dir).resolve()

    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.constraints_file:
        constraints_file = Path(args.constraints_file).resolve()
    else:
        constraints_file, _ = detect_patecon_files(output_dir, args.dataset_stem)

    if args.conflicts_file:
        conflicts_file = Path(args.conflicts_file).resolve()
    else:
        _, conflicts_file = detect_patecon_files(output_dir, args.dataset_stem)

    final_constraints_csv = processed_dir / "patecon_constraints.csv"
    final_conflicts_csv = processed_dir / "patecon_conflicts.csv"

    print("========== Convert Existing PaTeCon Outputs ==========")
    print("[INFO] project_root:", project_root)
    print("[INFO] output_dir:", output_dir)
    print("[INFO] processed_dir:", processed_dir)
    print("[INFO] constraints_file:", constraints_file)
    print("[INFO] conflicts_file:", conflicts_file)
    print()

    constraints_df = convert_constraints_to_csv(
        constraints_file=constraints_file,
        final_constraints_csv=final_constraints_csv,
    )

    conflicts_df = convert_conflicts_to_csv(
        conflicts_file=conflicts_file,
        final_conflicts_csv=final_conflicts_csv,
    )

    print()
    print("========== Finished ==========")
    print("[OK] 已生成:", final_constraints_csv)
    print("[INFO] 约束数量:", len(constraints_df))
    print("[OK] 已生成:", final_conflicts_csv)
    print("[INFO] 冲突数量:", len(conflicts_df))
    print()
    print("下一步可以运行：")
    print("    python scripts/11_update_graph.py")


if __name__ == "__main__":
    main()

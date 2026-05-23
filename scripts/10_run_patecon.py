# scripts/10_run_patecon.py
import sys
import subprocess
from pathlib import Path

import pandas as pd


# =========================
# 基础路径配置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATECON_DIR = PROJECT_ROOT / "PaTeCon-master"
PATECON_RESOURCE_DIR = PATECON_DIR / "resource"
PATECON_OUTPUT_DIR = PATECON_DIR / "output"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# PaTeCon resource 中的数据文件名
# 注意：该文件已经由 08_build_intervals.py 直接生成到
# project/PaTeCon-master/resource/org_relation_intervals.tsv
# 因此本脚本不再复制 tsv 文件。
PATECON_DATASET_NAME = "org_relation_intervals.tsv"
PATECON_DATASET_PATH = PATECON_RESOURCE_DIR / PATECON_DATASET_NAME

# PaTeCon 输出文件名规则：
# resource/org_relation_intervals.tsv
# -> output/org_relation_intervals.all_constraints
# -> output/org_relation_intervals.temporal_representation_conflicts
PATECON_DATASET_STEM = Path(PATECON_DATASET_NAME).stem

PATECON_CONSTRAINTS_FILE = (
    PATECON_OUTPUT_DIR / f"{PATECON_DATASET_STEM}.all_constraints"
)

PATECON_TEMPORAL_CONFLICTS_FILE = (
    PATECON_OUTPUT_DIR / f"{PATECON_DATASET_STEM}.temporal_representation_conflicts"
)

# 转换后给后续实验使用的文件
FINAL_CONSTRAINTS_CSV = PROCESSED_DIR / "patecon_constraints.csv"
FINAL_CONFLICTS_CSV = PROCESSED_DIR / "patecon_conflicts.csv"

# 保存运行日志，便于论文和 GitHub 记录
LOG_DIR = PROJECT_ROOT / "logs"
RUN_LOG_FILE = LOG_DIR / "10_run_patecon.log"


# =========================
# PaTeCon 参数
# =========================

KNOWLEDGEGRAPH = "other"
SUPPORT = "100"
CANDIDATE_CONFIDENCE = "0.5"
CONFIDENCE = "0.8"


def ensure_dirs():
    PATECON_RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    PATECON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def check_environment():
    if not PATECON_DIR.exists():
        raise FileNotFoundError(
            f"未找到 PaTeCon 源码目录：{PATECON_DIR}\n"
            "请确认目录结构为 project/PaTeCon-master"
        )

    constraint_mining_file = PATECON_DIR / "Constraint_Mining.py"
    if not constraint_mining_file.exists():
        raise FileNotFoundError(
            f"未找到 Constraint_Mining.py：{constraint_mining_file}"
        )

    if not PATECON_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"未找到 PaTeCon 输入文件：{PATECON_DATASET_PATH}\n"
            "请先运行 scripts/08_build_intervals.py，确保它直接生成：\n"
            "project/PaTeCon-master/resource/org_relation_intervals.tsv"
        )

    if PATECON_DATASET_PATH.stat().st_size == 0:
        raise ValueError(
            f"PaTeCon 输入文件为空：{PATECON_DATASET_PATH}\n"
            "请检查 08_build_intervals.py 的输出。"
        )


def run_patecon_constraint_mining():
    """
    按 PaTeCon 原始使用方式调用 Constraint_Mining.py。
    实时打印 PaTeCon 的运行输出，并保存日志。
    """
    cmd = [
        sys.executable,
        "-u",
        "Constraint_Mining.py",
        f"--dataset=resource/{PATECON_DATASET_NAME}",
        f"--knowledgegraph={KNOWLEDGEGRAPH}",
        f"--support={SUPPORT}",
        f"--candidate_confidence={CANDIDATE_CONFIDENCE}",
        f"--confidence={CONFIDENCE}",
    ]

    print("\n========== Run PaTeCon Constraint_Mining.py ==========")
    print("[CMD]", " ".join(cmd))

    log_lines = []

    process = subprocess.Popen(
        cmd,
        cwd=str(PATECON_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")
        log_lines.append(line)

    process.wait()

    RUN_LOG_FILE.write_text("".join(log_lines), encoding="utf-8")

    if process.returncode != 0:
        raise RuntimeError(
            f"PaTeCon Constraint_Mining.py 执行失败，returncode={process.returncode}\n"
            f"日志已保存到：{RUN_LOG_FILE}"
        )

    print(f"[OK] PaTeCon 运行日志已保存：{RUN_LOG_FILE}")

def parse_constraint_line(raw_line):
    """
    解析 PaTeCon 的一行约束。

    常见格式示例：
    a,P22*P569,d,t5,t6 before a,P39,b,t1,t2|0.9916

    或：
    a,P569,b,t1,t2 MutualExclusion a,P569,c,t3,t4|0.9934

    这里不强行解析复杂变量，只拆出：
    - body
    - temporal_predicate
    - head
    - confidence
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


def convert_constraints_to_csv():
    """
    将 PaTeCon 的 org_relation_intervals.all_constraints
    转换为 data/processed/patecon_constraints.csv。
    """
    if not PATECON_CONSTRAINTS_FILE.exists():
        raise FileNotFoundError(
            f"未找到 PaTeCon 约束输出文件：{PATECON_CONSTRAINTS_FILE}\n"
            "请检查 Constraint_Mining.py 是否正常完成，以及 output 文件名是否一致。"
        )

    rows = []

    with open(PATECON_CONSTRAINTS_FILE, "r", encoding="utf-8") as f:
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
                "source_file": str(PATECON_CONSTRAINTS_FILE),
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

    df.to_csv(FINAL_CONSTRAINTS_CSV, index=False, encoding="utf-8-sig")

    print("\n========== Convert Constraints ==========")
    print(f"[OK] 已生成：{FINAL_CONSTRAINTS_CSV}")
    print(f"[INFO] 约束数量：{len(df)}")


def convert_temporal_representation_conflicts_to_csv():
    """
    将 PaTeCon 的 temporal_representation_conflicts 转换为
    data/processed/patecon_conflicts.csv。

    该文件来自 PaTeCon 的 temporal_representation_constraint 阶段，
    主要记录 start_time > end_time 的时间表示错误。

    原始格式通常类似：
    subject,property,object,start_time,end_time
    """
    rows = []

    if not PATECON_TEMPORAL_CONFLICTS_FILE.exists():
        print("\n========== Convert Temporal Representation Conflicts ==========")
        print(f"[WARN] 未找到时间表示冲突文件：{PATECON_TEMPORAL_CONFLICTS_FILE}")
        print("[WARN] 将生成空的 patecon_conflicts.csv")

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
        df.to_csv(FINAL_CONFLICTS_CSV, index=False, encoding="utf-8-sig")
        return

    with open(PATECON_TEMPORAL_CONFLICTS_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            parts = raw.split(",")

            subject = parts[0].strip() if len(parts) > 0 else ""
            relation = parts[1].strip() if len(parts) > 1 else ""
            obj = parts[2].strip() if len(parts) > 2 else ""
            start_time = parts[3].strip() if len(parts) > 3 else ""
            end_time = parts[4].strip() if len(parts) > 4 else ""

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
                "source_file": str(PATECON_TEMPORAL_CONFLICTS_FILE),
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

    df.to_csv(FINAL_CONFLICTS_CSV, index=False, encoding="utf-8-sig")

    print("\n========== Convert Temporal Representation Conflicts ==========")
    print(f"[OK] 已生成：{FINAL_CONFLICTS_CSV}")
    print(f"[INFO] 时间表示冲突数量：{len(df)}")


def main():
    print("========== Step 10: Run PaTeCon ==========")

    ensure_dirs()
    check_environment()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] PaTeCon dir: {PATECON_DIR}")
    print(f"[INFO] PaTeCon dataset path: {PATECON_DATASET_PATH}")
    print(f"[INFO] PaTeCon dataset arg: resource/{PATECON_DATASET_NAME}")
    print(f"[INFO] Support: {SUPPORT}")
    print(f"[INFO] Candidate confidence: {CANDIDATE_CONFIDENCE}")
    print(f"[INFO] Confidence: {CONFIDENCE}")

    run_patecon_constraint_mining()

    convert_constraints_to_csv()

    convert_temporal_representation_conflicts_to_csv()

    print("\n========== Step 10 Finished ==========")
    print("输出文件：")
    print(" -", FINAL_CONSTRAINTS_CSV)
    print(" -", FINAL_CONFLICTS_CSV)


if __name__ == "__main__":
    main()
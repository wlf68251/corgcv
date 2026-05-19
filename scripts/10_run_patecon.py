# scripts/10_run_patecon.py
import os
import shutil
import subprocess
import pandas as pd

PROCESSED_DIR = "../data/processed"
RESOURCE_DIR = "../resource"

PATECON_DIR = "../PaTeCon"

INTERVAL_FILE = os.path.join(RESOURCE_DIR, "org_relation_intervals.tsv")

PATECON_RESOURCE_DIR = os.path.join(PATECON_DIR, "resource")
PATECON_DATASET_FILE = os.path.join(PATECON_RESOURCE_DIR, "org_relation_intervals.tsv")

PATECON_OUTPUT_DIR = os.path.join(PATECON_DIR, "output")

CONSTRAINTS_OUT = os.path.join(PROCESSED_DIR, "patecon_constraints.csv")
CONFLICTS_OUT = os.path.join(PROCESSED_DIR, "patecon_conflicts.csv")


SUPPORT = 5
CANDIDATE_CONFIDENCE = 0.5
CONFIDENCE = 0.8


def run_command(cmd, cwd):
    print("运行命令:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    print("标准输出:")
    print(result.stdout)

    if result.stderr:
        print("错误输出:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"命令运行失败: {' '.join(cmd)}")

    return result


def find_possible_file(base_dir, keywords):
    """
    在 PaTeCon 输出目录中查找可能的结果文件。
    因为不同版本 PaTeCon 输出文件名可能不同，所以这里做保守搜索。
    """
    if not os.path.exists(base_dir):
        return None

    candidates = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            lower = file.lower()

            if all(k.lower() in lower for k in keywords):
                candidates.append(os.path.join(root, file))

    if not candidates:
        return None

    # 选择最近修改的文件
    candidates = sorted(candidates, key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def convert_text_to_csv(input_file, output_file, default_columns=None):
    """
    将 PaTeCon 输出的 txt / csv / tsv 尽量转成 csv。
    如果无法判断格式，就按行保存为 raw_line。
    """

    if input_file is None or not os.path.exists(input_file):
        pd.DataFrame(columns=["raw_line"]).to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )
        return

    try:
        # 先尝试 tsv
        df = pd.read_csv(input_file, sep="\t", header=None, low_memory=False)

        if df.shape[1] == 1:
            # 再尝试逗号分隔
            df2 = pd.read_csv(input_file, sep=",", header=None, low_memory=False)
            if df2.shape[1] > df.shape[1]:
                df = df2

        if default_columns and len(default_columns) == df.shape[1]:
            df.columns = default_columns
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

    except Exception:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        df = pd.DataFrame({"raw_line": lines})

    df.to_csv(output_file, index=False, encoding="utf-8-sig")


def main():
    if not os.path.exists(INTERVAL_FILE):
        print(f"未找到 PaTeCon 输入文件: {INTERVAL_FILE}")
        print("请先运行 scripts/08_build_intervals.py")
        return

    if not os.path.exists(PATECON_DIR):
        print(f"未找到 PaTeCon 目录: {PATECON_DIR}")
        print("请检查 PATECON_DIR 是否正确。")
        return

    constraint_mining_py = os.path.join(PATECON_DIR, "Constraint_Mining.py")
    conflict_detection_py = os.path.join(PATECON_DIR, "Conflict_Detection.py")

    if not os.path.exists(constraint_mining_py):
        print(f"未找到 Constraint_Mining.py: {constraint_mining_py}")
        return

    if not os.path.exists(conflict_detection_py):
        print(f"未找到 Conflict_Detection.py: {conflict_detection_py}")
        return

    os.makedirs(PATECON_RESOURCE_DIR, exist_ok=True)
    os.makedirs(PATECON_OUTPUT_DIR, exist_ok=True)

    shutil.copyfile(INTERVAL_FILE, PATECON_DATASET_FILE)

    print("已复制 PaTeCon 输入文件:")
    print(PATECON_DATASET_FILE)

    # 1. 运行约束挖掘
    mining_cmd = [
        "python",
        "Constraint_Mining.py",
        "--dataset=resource/org_relation_intervals.tsv",
        "--knowledgegraph=other",
        f"--support={SUPPORT}",
        f"--candidate_confidence={CANDIDATE_CONFIDENCE}",
        f"--confidence={CONFIDENCE}"
    ]

    try:
        run_command(mining_cmd, cwd=PATECON_DIR)
    except Exception as e:
        print("PaTeCon 约束挖掘失败。")
        print(e)
        return

    # 2. 运行冲突检测
    # 文档中 constraint=output/all_constraints
    conflict_cmd = [
        "python",
        "Conflict_Detection.py",
        "--dataset=resource/org_relation_intervals.tsv",
        "--knowledgegraph=other",
        "--constraint=output/all_constraints"
    ]

    try:
        run_command(conflict_cmd, cwd=PATECON_DIR)
    except Exception as e:
        print("PaTeCon 冲突检测失败。")
        print(e)
        print("如果 output/all_constraints 不存在，请检查 PaTeCon 实际输出目录。")
        return

    # 3. 查找并转换输出
    constraint_file = find_possible_file(PATECON_OUTPUT_DIR, ["constraint"])
    conflict_file = find_possible_file(PATECON_OUTPUT_DIR, ["conflict"])

    print("检测到 PaTeCon 约束输出文件:", constraint_file)
    print("检测到 PaTeCon 冲突输出文件:", conflict_file)

    convert_text_to_csv(
        constraint_file,
        CONSTRAINTS_OUT
    )

    convert_text_to_csv(
        conflict_file,
        CONFLICTS_OUT
    )

    print("PaTeCon 运行完成")
    print("输出:", CONSTRAINTS_OUT)
    print("输出:", CONFLICTS_OUT)


if __name__ == "__main__":
    main()
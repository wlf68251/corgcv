# scripts/09_generate_llm_constraints.py
import os
import re
import json
import pandas as pd
from openai import OpenAI

PROCESSED_DIR = "../data/processed"

EDGES_FILE = os.path.join(PROCESSED_DIR, "relation_edges_scored.csv")
INTERVALS_FILE = os.path.join(PROCESSED_DIR, "org_relation_intervals.tsv")

LLM_PROMPT_OUT = os.path.join(PROCESSED_DIR, "llm_constraint_prompt.txt")
LLM_RAW_RESPONSE_OUT = os.path.join(PROCESSED_DIR, "llm_raw_response.txt")
LLM_JSON_OUT = os.path.join(PROCESSED_DIR, "llm_constraints.json")
LLM_CSV_OUT = os.path.join(PROCESSED_DIR, "temporal_constraints_llm.csv")

# =========================
# DashScope OpenAI 兼容接口配置
# =========================
# 不建议把真实 Key 写死在代码里。
# 推荐在终端中设置环境变量：
# export DASHSCOPE_API_KEY="你的apikey"
#
# 如果你只是本地测试，也可以临时取消下面这一行注释并填入 Key：
# DASHSCOPE_API_KEY = ""

# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_KEY = "sk-58624b939f654783bc6f7e909a76c8fa"

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# 如果 LLM 调用失败，是否使用默认约束兜底，防止流程中断
USE_DEFAULT_CONSTRAINTS_WHEN_FAILED = True


RELATION_TYPE_DESCRIPTIONS = {
    "verbal_cooperation": "言语合作，例如声明支持、表达合作意愿、外交沟通、协商、赞同。",
    "material_cooperation": "实质合作，例如援助、经济合作、军事合作、实际行动支持、提供资源。",
    "verbal_conflict": "言语冲突，例如批评、谴责、威胁、外交抗议、表达不满。",
    "material_conflict": "实质冲突，例如制裁、军事攻击、逮捕、封锁、武装冲突。",
    "mixed_relation": "混合关系，表示同一时间窗口内存在多种关系并存，不宜简单判定为单一关系。"
}


REQUIRED_FIELDS = [
    "constraint_name",
    "relation_a",
    "relation_b",
    "temporal_predicate",
    "hard_or_soft",
    "expected_action",
    "reason"
]


def load_intervals():
    """
    读取 PaTeCon 使用的月度关系区间。
    格式：
    subject property object start_time end_time
    """

    if not os.path.exists(INTERVALS_FILE):
        print(f"未找到月度关系区间文件: {INTERVALS_FILE}")
        return pd.DataFrame(
            columns=["subject", "property", "object", "start_time", "end_time"]
        )

    intervals = pd.read_csv(
        INTERVALS_FILE,
        sep="\t",
        header=None,
        names=["subject", "property", "object", "start_time", "end_time"],
        low_memory=False
    )

    return intervals


def build_stats_text(edges, intervals):
    """
    构造图谱统计信息，作为 LLM 生成约束的依据。
    """

    total_edges = len(edges)
    total_intervals = len(intervals)

    source_stats = {}
    relation_stats = {}
    confidence_stats = {}

    if "source_datasets" in edges.columns:
        source_stats = edges["source_datasets"].fillna("").value_counts().to_dict()

    if "relation_type" in edges.columns:
        relation_stats = edges["relation_type"].fillna("").value_counts().to_dict()

    if "confidence_level" in edges.columns:
        confidence_stats = edges["confidence_level"].fillna("").value_counts().to_dict()

    stats_text = f"""
关系边数量: {total_edges}
月度关系区间数量: {total_intervals}
来源分布: {source_stats}
关系类型分布: {relation_stats}
置信度等级分布: {confidence_stats}
""".strip()

    return stats_text


def build_prompt(relation_types, samples, stats_text):
    """
    构造给 LLM 的提示词。
    """

    relation_type_text = "\n".join([
        f"- {r}: {RELATION_TYPE_DESCRIPTIONS.get(r, '无说明')}"
        for r in relation_types
    ])

    if samples.empty:
        sample_text = "当前没有可用月度关系样例。"
    else:
        sample_lines = []
        for _, row in samples.iterrows():
            sample_lines.append(
                f"{row['subject']} {row['property']} {row['object']} "
                f"{row['start_time']} {row['end_time']}"
            )
        sample_text = "\n".join(sample_lines)

    prompt = f"""
你现在需要为一个多源组织机构关系图谱生成候选时序约束。

项目背景：
- 数据来自 ICEWS 与 GDELT。
- 数据时间范围主要为 2023 年 1 月至 4 月。
- 原始数据是离散事件，已经按月聚合为组织关系区间。
- 这些关系是派生关系，不是原始事件本身。
- 后续会结合 PaTeCon 挖掘出的数据约束，筛选最终校验规则。
- 约束用于校验不合理派生关系，但不能误删原始事件证据。

关系类型列表：
{relation_type_text}

图谱统计信息：
{stats_text}

月度关系样例：
{sample_text}

请你生成候选时序约束，输出 JSON 数组。
每个元素必须包含以下字段：

{{
  "constraint_name": "",
  "relation_a": "",
  "relation_b": "",
  "temporal_predicate": "",
  "hard_or_soft": "",
  "expected_action": "",
  "reason": ""
}}

字段含义：
- constraint_name: 约束名称，使用英文小写和下划线。
- relation_a: 第一个关系类型，例如 verbal_conflict。
- relation_b: 第二个关系类型。如果只涉及单个关系，可填 none 或 any。
- temporal_predicate: 时序谓词，例如 same_month_same_subject_object、before、after、overlap、cross_source_supported、gdelt_single_source_low_event_count。
- hard_or_soft: 只能填 hard 或 soft。除非非常确定，否则优先 soft。
- expected_action: 触发约束后的建议动作，例如 keep、review、downgrade、hide、mark_mixed、mark_mixed_or_downgrade。
- reason: 中文解释，说明为什么该约束合理。

生成要求：
1. 必须只输出 JSON 数组。
2. 不要输出 Markdown。
3. 不要使用 ```json 代码块。
4. 不要输出额外解释。
5. 不要生成过于绝对的约束。
6. 现实组织关系可能同时存在合作和冲突，不能简单互斥。
7. verbal_conflict 与 material_conflict 可以存在升级关系，但不能认为一定冲突。
8. material_cooperation 与 material_conflict 如果同月同对象并存，通常应标记为 mixed_relation 或降权复核。
9. GDELT 单源低证据关系更适合 hide 或 downgrade。
10. ICEWS 和 GDELT 共同支持的关系一般不直接删除。
11. hard 约束数量应少，soft 约束数量可以多。
12. 建议生成 6 到 12 条候选约束。
""".strip()

    return prompt


def call_llm(prompt):
    """
    调用 DashScope OpenAI 兼容接口。
    """

    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "未设置 DASHSCOPE_API_KEY。请先执行：\n"
            "export DASHSCOPE_API_KEY=\"你的apikey\""
        )

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是知识图谱时序约束生成助手。"
                    "你必须严格输出 JSON 数组。"
                    "不要输出 Markdown，不要输出解释文字。"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("LLM 返回内容为空。")

    return content.strip()


def extract_json_array(text):
    """
    从 LLM 输出中提取 JSON 数组。
    即使模型输出了 ```json，也尽量清洗。
    """

    text = text.strip()

    # 去掉 Markdown 代码块
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 截取第一个 [ 到最后一个 ]
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("未在 LLM 输出中找到 JSON 数组。")

    json_text = text[start:end + 1]

    return json.loads(json_text)


def validate_constraints(constraints):
    """
    检查并规范化 LLM 生成的约束。
    """

    if not isinstance(constraints, list):
        raise ValueError("LLM 输出不是 JSON 数组。")

    cleaned = []

    for i, item in enumerate(constraints):
        if not isinstance(item, dict):
            continue

        new_item = {}

        for field in REQUIRED_FIELDS:
            value = item.get(field, "")
            if value is None:
                value = ""
            new_item[field] = str(value).strip()

        if not new_item["constraint_name"]:
            new_item["constraint_name"] = f"llm_constraint_{i + 1}"

        # 规范 hard_or_soft
        hs = new_item["hard_or_soft"].lower()
        if hs not in ["hard", "soft"]:
            hs = "soft"
        new_item["hard_or_soft"] = hs

        # 规范 relation_b
        if not new_item["relation_b"]:
            new_item["relation_b"] = "none"

        # 规范 expected_action
        if not new_item["expected_action"]:
            new_item["expected_action"] = "review"

        # 规范 reason
        if not new_item["reason"]:
            new_item["reason"] = "LLM 生成的候选约束，需后续结合 PaTeCon 和人工抽样检查。"

        cleaned.append(new_item)

    if not cleaned:
        raise ValueError("LLM 输出中没有有效约束。")

    return cleaned


def default_constraints():
    """
    当 LLM 调用失败时的保守兜底约束。
    这样可以保证后续流程不中断。
    """

    return [
        {
            "constraint_name": "same_month_material_cooperation_conflict_mixed",
            "relation_a": "material_cooperation",
            "relation_b": "material_conflict",
            "temporal_predicate": "same_month_same_subject_object",
            "hard_or_soft": "soft",
            "expected_action": "mark_mixed_or_downgrade",
            "reason": "同一组织对在同一月份同时存在实质合作和实质冲突，可能表示复杂关系，不宜直接删除，应优先标记为 mixed_relation 或降低置信度后复核。"
        },
        {
            "constraint_name": "same_month_verbal_cooperation_material_conflict_mixed",
            "relation_a": "verbal_cooperation",
            "relation_b": "material_conflict",
            "temporal_predicate": "same_month_same_subject_object",
            "hard_or_soft": "soft",
            "expected_action": "mark_mixed",
            "reason": "同月同时出现合作表态和实质冲突，可能反映复杂外交或组织关系，应标记为 mixed_relation。"
        },
        {
            "constraint_name": "single_source_low_evidence_gdelt_hide",
            "relation_a": "any",
            "relation_b": "none",
            "temporal_predicate": "gdelt_single_source_low_event_count",
            "hard_or_soft": "soft",
            "expected_action": "hide",
            "reason": "GDELT 单源且事件数量较少的派生关系证据不足，适合隐藏或降权，而不是删除原始事件。"
        },
        {
            "constraint_name": "cross_source_supported_relation_keep",
            "relation_a": "any",
            "relation_b": "none",
            "temporal_predicate": "cross_source_supported",
            "hard_or_soft": "soft",
            "expected_action": "keep_or_review",
            "reason": "ICEWS 和 GDELT 共同支持的关系边证据较强，即使触发弱冲突，也不应直接删除。"
        },
        {
            "constraint_name": "verbal_conflict_before_material_conflict_review",
            "relation_a": "verbal_conflict",
            "relation_b": "material_conflict",
            "temporal_predicate": "before_or_overlap",
            "hard_or_soft": "soft",
            "expected_action": "review_or_keep",
            "reason": "言语冲突可能先于实质冲突出现，但该模式不是硬约束，只能作为冲突升级的弱提示。"
        },
        {
            "constraint_name": "same_month_verbal_and_material_conflict_consistent",
            "relation_a": "verbal_conflict",
            "relation_b": "material_conflict",
            "temporal_predicate": "same_month_same_subject_object",
            "hard_or_soft": "soft",
            "expected_action": "keep_or_review",
            "reason": "言语冲突和实质冲突在同一月份并存通常具有语义一致性，可作为冲突关系增强信号，而不是异常。"
        }
    ]


def main():
    if not os.path.exists(EDGES_FILE):
        print(f"未找到输入文件: {EDGES_FILE}")
        print("请先运行 scripts/07_score_relations.py")
        return

    edges = pd.read_csv(EDGES_FILE, low_memory=False)
    intervals = load_intervals()

    # 关系类型
    if "relation_type" in edges.columns:
        relation_types = sorted(
            edges["relation_type"].dropna().astype(str).unique().tolist()
        )
    elif not intervals.empty:
        relation_types = sorted(
            intervals["property"].dropna().astype(str).unique().tolist()
        )
    else:
        relation_types = list(RELATION_TYPE_DESCRIPTIONS.keys())

    # 图谱统计
    stats_text = build_stats_text(edges, intervals)

    # 月度关系样例，最多 30 条
    if not intervals.empty:
        samples = intervals.sample(
            n=min(30, len(intervals)),
            random_state=42
        )
    else:
        samples = pd.DataFrame(
            columns=["subject", "property", "object", "start_time", "end_time"]
        )

    # 构造 prompt
    prompt = build_prompt(relation_types, samples, stats_text)

    with open(LLM_PROMPT_OUT, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("已生成 LLM prompt:")
    print(LLM_PROMPT_OUT)

    # 调用 LLM
    try:
        raw_response = call_llm(prompt)

        with open(LLM_RAW_RESPONSE_OUT, "w", encoding="utf-8") as f:
            f.write(raw_response)

        constraints_raw = extract_json_array(raw_response)
        constraints = validate_constraints(constraints_raw)

        print("LLM 调用成功，JSON 解析成功。")

    except Exception as e:
        print("LLM 调用或解析失败。")
        print("错误信息:", e)

        if not USE_DEFAULT_CONSTRAINTS_WHEN_FAILED:
            return

        print("使用默认保守约束继续生成输出文件。")
        constraints = default_constraints()

        with open(LLM_RAW_RESPONSE_OUT, "w", encoding="utf-8") as f:
            f.write(f"LLM 调用失败，使用默认约束。\n错误信息: {e}\n")

    # 输出 JSON
    with open(LLM_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(constraints, f, ensure_ascii=False, indent=2)

    # 输出 CSV
    constraints_df = pd.DataFrame(constraints)

    # 保证列顺序
    for field in REQUIRED_FIELDS:
        if field not in constraints_df.columns:
            constraints_df[field] = ""

    constraints_df = constraints_df[REQUIRED_FIELDS]

    constraints_df.to_csv(
        LLM_CSV_OUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("LLM 候选时序约束生成完成")
    print("输出 prompt:", LLM_PROMPT_OUT)
    print("输出 raw response:", LLM_RAW_RESPONSE_OUT)
    print("输出 JSON:", LLM_JSON_OUT)
    print("输出 CSV:", LLM_CSV_OUT)
    print("候选约束数量:", len(constraints_df))


if __name__ == "__main__":
    main()
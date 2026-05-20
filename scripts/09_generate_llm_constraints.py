# scripts/09_generate_llm_constraints.py
import os
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INTERVAL_FILE = PROCESSED_DIR / "org_relation_intervals.tsv"

PROMPT_OUT = PROCESSED_DIR / "llm_constraint_prompt.txt"
RAW_RESPONSE_OUT = PROCESSED_DIR / "llm_raw_response.txt"

LLM_JSON_OUT = PROCESSED_DIR / "llm_constraints.json"
LLM_CSV_OUT = PROCESSED_DIR / "temporal_constraints_llm.csv"


CAMEO_TOP_LEVEL_DESC = {
    "01": "Make public statement / 公开表态",
    "02": "Appeal / 呼吁、请求",
    "03": "Express intent to cooperate / 表达合作意向",
    "04": "Consult / 磋商、会谈",
    "05": "Engage in diplomatic cooperation / 外交合作",
    "06": "Engage in material cooperation / 物质合作",
    "07": "Provide aid / 提供援助",
    "08": "Yield / 让步",
    "09": "Investigate / 调查",
    "10": "Demand / 要求",
    "11": "Disapprove / 反对、不满",
    "12": "Reject / 拒绝",
    "13": "Threaten / 威胁",
    "14": "Protest / 抗议",
    "15": "Exhibit force posture / 展示武力姿态",
    "16": "Reduce relations / 降低关系",
    "17": "Coerce / 胁迫",
    "18": "Assault / 攻击",
    "19": "Fight / 战斗",
    "20": "Use unconventional mass violence / 非常规大规模暴力",
}


DEFAULT_CONSTRAINTS = [
    {
        "constraint_name": "threat_before_assault",
        "relation_a": "13",
        "relation_b": "18",
        "temporal_predicate": "before",
        "hard_or_soft": "soft",
        "expected_action": "review_or_downgrade",
        "reason": "Threatening events may precede assault events, but the rule should remain soft because real political processes are noisy."
    },
    {
        "constraint_name": "demand_before_reject",
        "relation_a": "10",
        "relation_b": "12",
        "temporal_predicate": "before",
        "hard_or_soft": "soft",
        "expected_action": "review",
        "reason": "Demands are often followed by rejection, but event reports may be incomplete or reversed in time."
    },
    {
        "constraint_name": "cooperation_conflict_review",
        "relation_a": "05",
        "relation_b": "18",
        "temporal_predicate": "disjoint_or_review",
        "hard_or_soft": "soft",
        "expected_action": "mark_mixed_or_review",
        "reason": "Diplomatic cooperation and assault between the same organizations in the same month may indicate mixed or complex relations rather than direct deletion."
    },
    {
        "constraint_name": "aid_before_yield",
        "relation_a": "07",
        "relation_b": "08",
        "temporal_predicate": "before",
        "hard_or_soft": "soft",
        "expected_action": "review",
        "reason": "Aid may be associated with later yielding behavior, but the rule is not deterministic."
    },
    {
        "constraint_name": "protest_before_coerce",
        "relation_a": "14",
        "relation_b": "17",
        "temporal_predicate": "before",
        "hard_or_soft": "soft",
        "expected_action": "review_or_downgrade",
        "reason": "Protest may precede coercive responses, but this should only be used as a weak temporal signal."
    }
]


def read_interval_sample():
    if not INTERVAL_FILE.exists():
        return pd.DataFrame(columns=["subject", "property", "object", "start_time", "end_time"])

    df = pd.read_csv(
        INTERVAL_FILE,
        sep="\t",
        header=None,
        names=["subject", "property", "object", "start_time", "end_time"],
        dtype=str
    ).fillna("")

    return df


def build_prompt(df):
    relation_counts = {}

    if not df.empty and "property" in df.columns:
        relation_counts = df["property"].value_counts().to_dict()

    relation_desc_lines = []
    for code, desc in CAMEO_TOP_LEVEL_DESC.items():
        count = relation_counts.get(code, 0)
        relation_desc_lines.append(f"- {code}: {desc}; interval_count={count}")

    sample_lines = []

    if not df.empty:
        sample = df.head(50)
        for _, row in sample.iterrows():
            sample_lines.append(
                f"{row['subject']} {row['property']} {row['object']} "
                f"{row['start_time']} {row['end_time']}"
            )

    prompt = f"""
你是一个知识图谱时序约束分析助手。

当前任务：
根据组织机构关系图谱中的 CAMEO 顶层事件类型，生成候选时序约束。

重要说明：
1. 当前 relation_type 已经使用 CAMEO 顶层码，不使用 verbal_cooperation 等粗粒度类型。
2. 关系类型是两位字符串：01, 02, ..., 20。
3. 不要输出 CAMEO_01，直接输出 01。
4. 这些约束主要用于辅助图谱校验，不应过于绝对。
5. 优先生成 soft constraint，避免误删真实复杂关系。
6. 对于同一组织对同一月份出现多种关系，应优先考虑 mixed/review，而不是直接删除。

CAMEO 顶层关系类型：
{chr(10).join(relation_desc_lines)}

样例区间：
{chr(10).join(sample_lines)}

请输出 JSON 数组，每个元素格式如下：
[
  {{
    "constraint_name": "",
    "relation_a": "13",
    "relation_b": "18",
    "temporal_predicate": "before/disjoint/include/overlap/review",
    "hard_or_soft": "soft",
    "expected_action": "review/downgrade/hide/mark_mixed",
    "reason": ""
  }}
]

要求：
- relation_a 和 relation_b 必须是 01~20 的两位字符串。
- 不要添加 CAMEO_ 前缀。
- 不要生成过强硬的删除规则。
- 如果语义上不确定，请设置 hard_or_soft 为 soft。
"""
    return prompt.strip()


def call_llm_if_available(prompt):
    """
    可选调用 DashScope/OpenAI-compatible API。
    如果没有环境变量或 openai 包，则返回空字符串，后续使用默认约束。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

    if not api_key:
        return ""

    try:
        from openai import OpenAI
    except Exception:
        return ""

    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    model_name = os.getenv("DASHSCOPE_MODEL", "qwen-plus")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是知识图谱时序约束生成助手，只输出 JSON 数组。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        print("[WARN] LLM 调用失败，将使用默认约束。错误:", e)
        return ""


def extract_json_array(text):
    if not text:
        return None

    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return data
        except Exception:
            return None

    return None


def normalize_constraints(items):
    rows = []

    for idx, item in enumerate(items, start=1):
        relation_a = str(item.get("relation_a", "")).strip().zfill(2)
        relation_b = str(item.get("relation_b", "")).strip().zfill(2)

        if relation_a not in CAMEO_TOP_LEVEL_DESC:
            continue

        if relation_b not in CAMEO_TOP_LEVEL_DESC:
            continue

        rows.append({
            "constraint_id": f"LLM_{idx:06d}",
            "constraint_name": item.get("constraint_name", f"constraint_{idx}"),
            "relation_a": relation_a,
            "relation_b": relation_b,
            "temporal_predicate": item.get("temporal_predicate", ""),
            "hard_or_soft": item.get("hard_or_soft", "soft"),
            "expected_action": item.get("expected_action", "review"),
            "reason": item.get("reason", ""),
            "source": "llm"
        })

    return rows


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = read_interval_sample()

    prompt = build_prompt(df)
    PROMPT_OUT.write_text(prompt, encoding="utf-8")

    print("[OK] 已生成 LLM prompt:", PROMPT_OUT)

    raw_response = call_llm_if_available(prompt)

    if raw_response:
        RAW_RESPONSE_OUT.write_text(raw_response, encoding="utf-8")
        parsed = extract_json_array(raw_response)
    else:
        RAW_RESPONSE_OUT.write_text("", encoding="utf-8")
        parsed = None

    if not parsed:
        print("[WARN] 未获得可解析 LLM 输出，使用默认 CAMEO 顶层约束模板。")
        parsed = DEFAULT_CONSTRAINTS

    rows = normalize_constraints(parsed)

    if not rows:
        rows = normalize_constraints(DEFAULT_CONSTRAINTS)

    with open(LLM_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    out_df = pd.DataFrame(rows)

    out_df.to_csv(
        LLM_CSV_OUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("[OK] 已生成:", LLM_JSON_OUT)
    print("[OK] 已生成:", LLM_CSV_OUT)
    print("[STAT] LLM constraints:", len(out_df))


if __name__ == "__main__":
    main()
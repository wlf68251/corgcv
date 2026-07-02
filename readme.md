# 多源组织机构关系图谱构建与校验更新项目

## 1. 项目介绍

本项目面向 ICEWS 与 GDELT 多源事件数据，构建组织机构时序关系图谱，并在图谱构建结果基础上进行候选时序互斥约束生成、结构证据评分、冲突检测、关系边状态更新与可视化展示。项目关注的核心问题不是单纯从事件数据中抽取关系，而是在多源事件融合后，对组织机构之间复杂、多变且证据强弱不一的关系进行统一建模、可信度计算和后验校验。

在数据层面，项目将 ICEWS 与 GDELT 中的事件记录统一抽象为组织—关系—组织的事件事实，并基于 CAMEO 事件编码映射生成组织机构关系类型。由于不同数据源在字段结构、事件编码、时间粒度和来源证据上存在差异，项目首先对组织名称进行抽取、规范化与实体对齐，然后按主体组织、客体组织、关系类型和月份进行聚合，形成具有时间、来源、事件频次和证据统计信息的派生关系边。关系边并不是对原始事件的简单复制，而是多源事件在组织关系层面的聚合表示。

在关系建模层面，项目为每条派生关系边计算置信度，并进一步将连续月份中的关系边合并为关系时间区间。置信度计算综合考虑事件数量、来源数量、跨源支持、新闻统计字段和时间连续性等因素，用于刻画关系边的证据强弱。时间区间表示则为后续分析关系之间的先后、重叠、包含和分离等时序模式提供基础。

在约束生成与校验层面，项目采用“LLM 语义候选 + 结构证据评分”的方法逻辑。首先，利用大语言模型根据关系类型语义、典型样例和关系对共现摘要生成候选时序互斥约束；然后，不直接采用 LLM 输出作为最终规则，而是基于关系时间区间统计支持实例、违反实例和未知实例，并计算结构置信度、违反比例和来源加权置信度。通过这一过程，项目将 LLM 的语义先验与当前图谱中的结构证据结合起来，避免直接使用 LLM 候选规则导致约束过宽，也避免单纯结构挖掘方法与组织关系语义不匹配的问题。

在图谱更新层面，项目采用保守更新策略。候选约束经过结构证据筛选后，只作用于融合后的派生关系边，不删除 ICEWS/GDELT 原始事件证据。对于触发互斥约束的关系边，系统根据约束状态、关系边置信度、来源证据和冲突类型生成 `keep`、`review`、`mark_mixed`、`downgrade`、`hide` 等更新动作。其中，`review` 用于记录潜在冲突，`mark_mixed` 用于标记复杂关系并存，整体目标是在保留高证据关系的同时，对低证据或潜在异常关系进行解释性标记和后验校验。

本项目的方法特点主要体现在以下几个方面：

1. **多源事件融合建模**：将 ICEWS 与 GDELT 事件数据统一为组织机构关系边，并保留来源、时间、事件数量和新闻统计字段等证据信息。
2. **时序关系图谱构建**：以月份为基本粒度构建组织机构时序关系边，并进一步生成关系时间区间，用于分析关系演化和关系共现模式。
3. **关系边置信度计算**：从事件频次、跨源支持、来源数量和文本证据等角度计算派生关系边置信度，为后续更新决策提供依据。
4. **LLM 候选约束生成**：利用大语言模型根据关系语义生成候选互斥约束，降低人工构造规则的成本，并提升候选约束的语义解释性。
5. **结构证据评分**：通过时间区间中的支持实例、违反实例和来源加权结果，对 LLM 候选约束进行筛选，避免直接采用全部候选规则。
6. **保守式图谱校验更新**：不直接删除原始事件证据，而是对派生关系边进行 `review`、`mark_mixed` 等状态标记，使图谱更新结果具有可追溯性和可解释性。
7. **可视化与实验统计输出**：项目提供 Streamlit 可视化应用，并输出候选约束质量、图谱更新结果、方法对比和案例分析等实验统计文件，便于论文分析和结果展示。

整体而言，本项目形成了从多源事件数据处理、组织关系图谱构建、候选时序约束生成、结构证据评分到图谱校验更新与可视化展示的完整流程。该流程适用于分析多源事件数据中组织机构关系的动态变化、复杂并存关系以及融合后关系边的可信更新问题。

------

## 2. 数据说明

### 2.1 原始数据

项目使用两类事件数据：

1. ICEWS 事件数据
   主要位于：

   ```text
   data/raw/icews/
   ```

2. GDELT 事件数据
   主要位于：

   ```text
   data/raw/gdelt/
   ```

ICEWS 与 GDELT 原始数据规模较大，因此当前项目目录中的数据已经是经过整理、筛选或预处理后的数据。正常复现实验时，一般不需要重新运行 `01_prepare_icews.py` 和 `02_prepare_gdelt.py`。

### 2.2 默认不运行的脚本

以下脚本主要用于早期原始数据准备或大规模数据下载：

```text
scripts/01_prepare_icews.py
scripts/02_prepare_gdelt.py
data/data_icews.py
data/download_gdelt_202301_202304.py
```

由于当前 `data` 目录中已经包含后续流程所需的处理后数据，因此正常运行项目时可以跳过这些脚本。

`scripts` 目录中没有数字编号的 Python 文件，例如：

```text
export_constraint_quality_table.py
export_edges_from_db.py
prepare_further.py
```

这些文件主要用于数据导出、补充统计或辅助处理，不属于 `03` 到 `12` 的主流程步骤。

------

## 3. 运行环境

项目主要依赖 Python 环境运行，建议使用 conda 或 venv 创建独立环境。

常用依赖包括：

```text
pandas
numpy
streamlit
neo4j
openai
```

如果需要运行可视化系统，还需要安装 Streamlit：

```bash
pip install streamlit
```

如果需要调用兼容 OpenAI 接口的大语言模型服务，需要在本地配置对应的 API Key、Base URL 和模型名称。若使用已有 LLM 输出进行解析，则推荐运行 `09_generate_llm_constraints.py` 时添加 `--parse_existing` 参数，避免重复调用模型接口。

------

## 4. 正常运行流程

正常运行流程是依次执行 `03` 到 `12` 号脚本，然后启动 `app.py` 进行结果展示。

建议在项目根目录下依次运行以下命令。

### 4.1 组织机构抽取

```bash
python scripts/03_extract_organizations.py
```

该步骤从事件数据中抽取组织机构候选实体，生成组织候选表和组织原始名称表。

主要输出包括：

```text
data/processed/organizations_raw.csv
data/processed/actor_candidates.csv
```

------

### 4.2 组织机构对齐

```bash
python scripts/04_align_organizations.py
```

该步骤对组织机构名称进行规范化和对齐，减少同一组织因名称变体导致的重复节点问题。

主要输出包括：

```text
data/processed/organizations.csv
data/processed/organization_aliases.csv
```

------

### 4.3 构建 ICEWS 种子关系图

```bash
python scripts/05_build_seed_graph.py
```

该步骤基于 ICEWS 事件数据构建初始组织关系边，并将事件关系映射到统一的关系类型。

主要输出包括：

```text
data/processed/relation_edges_seed.csv
data/processed/icews_seed_graph.csv
```

------

### 4.4 流式融合 GDELT 数据

```bash
python scripts/06_stream_fuse_gdelt.py
```

该步骤将 GDELT 事件数据与已有 ICEWS 种子图进行融合，生成多源组织关系边。

主要输出包括：

```text
data/processed/relation_edges_before_check.csv
data/processed/event_facts.csv
data/processed/event_facts_with_relation_type.csv
data/processed/source_evidence.csv
data/processed/stream_update_log.csv
```

------

### 4.5 关系边置信度计算

```bash
python scripts/07_score_relations.py
```

该步骤基于事件频次、来源数量、跨源支持、新闻统计字段等信息计算关系边置信度。

主要输出包括：

```text
data/processed/relation_edges_scored.csv
```

------

### 4.6 构建关系时间区间

```bash
python scripts/08_build_intervals.py
```

该步骤将按月关系边转化为时间区间表示，为后续时序约束评分提供输入。

主要输出包括：

```text
data/processed/org_relation_intervals.tsv
```

------

### 4.7 生成 LLM 候选互斥约束

```bash
python scripts/09_generate_llm_constraints.py --parse_existing
```

该步骤用于生成或解析 LLM 候选互斥约束。

当前推荐使用：

```bash
--parse_existing
```

该选项表示优先解析已有的 LLM 原始输出，而不是重新调用大语言模型接口。这样可以避免重复调用 API，也能保证实验结果与已有输出保持一致。

主要输入包括：

```text
data/processed/relation_inventory_for_llm.csv
data/processed/relation_examples_for_llm.csv
data/processed/relation_pair_temporal_summary_for_llm.csv
data/processed/mutual_exclusion_raw_response.txt
```

主要输出包括：

```text
data/processed/mutual_exclusion_llm_candidates.csv
data/processed/mutual_exclusion_llm_candidates.json
data/processed/mutual_exclusion_prompt.txt
```

------

### 4.8 互斥约束结构证据评分

```bash
python scripts/10_score_mutex_with_patecon.py
```

该步骤对 LLM 生成的候选互斥约束进行 PaTeCon-style 结构证据评分，统计支持实例、违反实例、结构置信度、违反比例、来源加权置信度等指标。

主要输出包括：

```text
data/processed/mutual_exclusion_scored.csv
data/processed/mutual_exclusion_constraints_final.csv
data/processed/mutual_exclusion_conflicts.csv
data/processed/mutual_exclusion_trigger_instances.csv
data/processed/mutual_exclusion_violation_instances.csv
```

------

### 4.9 图谱校验更新

```bash
python scripts/11_mutex_update.py
```

该步骤根据筛选后的候选互斥约束，对派生关系边进行校验更新。更新动作包括：

```text
keep
review
mark_mixed
downgrade
hide
```

当前实验中主要使用 `keep`、`review` 和 `mark_mixed`，整体策略偏向保守更新，即记录和标记复杂关系，而不是直接删除原始事件证据。

主要输出包括：

```text
data/processed/relation_edges_after_check.csv
data/processed/detected_conflicts.csv
data/processed/update_decisions.csv
```

------

### 4.10 结果评估与统计

```bash
python scripts/12_evaluate.py
```

该步骤对图谱更新结果进行基础统计与评估，生成评价样本和统计结果。

主要输出位于：

```text
outputs/evaluation_tables/
outputs/case_studies/
```

------

## 5. 补充统计脚本

除 `03` 到 `12` 主流程外，项目还包含若干补充统计脚本，主要用于论文第六章实验结果分析。

### 5.1 方法有效性指标统计

```bash
python scripts/13_compute_effectiveness_metrics.py
```

主要输出：

```text
outputs/evaluation_tables/table_6_x_effectiveness_metrics.csv
outputs/evaluation_tables/table_6_x_evidence_group_actions.csv
outputs/evaluation_tables/table_6_x_constraint_status_stats.csv
```

该步骤用于统计高证据关系保护率、低证据异常关系处理率、更新影响控制率、触发冲突覆盖率等指标。

### 5.2 对比方法实验

```bash
python scripts/14_compare_baseline_methods.py
```

主要输出：

```text
outputs/evaluation_tables/table_6_x_method_comparison_metrics.csv
outputs/evaluation_tables/table_6_x_method_action_distribution.csv
outputs/evaluation_tables/table_6_x_method_evidence_group_actions.csv
```

该步骤用于比较 NoConstraint、PaTeCon-Direct、LLM-only 和 SSTC-Fusion 四类方法的图谱校验更新效果。

### 5.3 候选约束质量明细导出

```bash
python scripts/export_constraint_quality_table.py
```

主要输出：

```text
outputs/evaluation_tables/table_6_x_constraint_quality_detail.csv
outputs/evaluation_tables/table_6_x_constraint_quality_detail.md
```

该步骤用于导出每条候选互斥约束的结构置信度、违反比例、来源加权置信度、最终状态以及触发结果。

------

## 6. 可视化系统运行

可视化系统位于 `app` 目录下，使用 Streamlit 启动。

进入 `app` 目录：

```bash
cd app
```

启动应用：

```bash
streamlit run ./app.py
```

如果在项目根目录运行，也可以使用：

```bash
streamlit run ./app/app.py
```

可视化系统主要用于展示：

1. 多源组织机构关系图谱；
2. 按月份切换的时间轴图谱；
3. 关系边状态筛选；
4. 触发互斥约束的关系边；
5. 单节点关系演化；
6. 候选互斥约束与冲突记录。

------

## 7. Neo4j 导入说明

如果需要将关系图谱导入 Neo4j，可使用：

```bash
python app/import_to_neo4j.py
```

该脚本通常读取处理后的关系边文件和组织节点文件，将组织机构节点、关系边、时间属性、置信度和更新状态导入 Neo4j。

常用输入包括：

```text
data/processed/organizations.csv
data/processed/relation_edges_after_check.csv
```

运行前需要确认 Neo4j 服务已经启动，并且脚本中的连接地址、用户名和密码与本地 Neo4j 配置一致。

------

## 8. 主要输入输出文件

### 8.1 主流程关键输入

```text
data/raw/icews/icews_raw_events.csv
data/raw/gdelt/gdelt_raw_events.csv
data/processed/organizations_raw.csv
```

### 8.2 组织机构相关文件

```text
data/processed/actor_candidates.csv
data/processed/organizations_raw.csv
data/processed/organizations.csv
data/processed/organization_aliases.csv
```

### 8.3 关系边相关文件

```text
data/processed/relation_edges_seed.csv
data/processed/relation_edges_before_check.csv
data/processed/relation_edges_scored.csv
data/processed/relation_edges_after_check.csv
```

### 8.4 时间区间与约束相关文件

```text
data/processed/org_relation_intervals.tsv
data/processed/relation_inventory_for_llm.csv
data/processed/relation_examples_for_llm.csv
data/processed/relation_pair_temporal_summary_for_llm.csv
data/processed/mutual_exclusion_llm_candidates.csv
data/processed/mutual_exclusion_scored.csv
data/processed/mutual_exclusion_constraints_final.csv
```

### 8.5 冲突与更新相关文件

```text
data/processed/detected_conflicts.csv
data/processed/update_decisions.csv
data/processed/mutual_exclusion_trigger_instances.csv
data/processed/mutual_exclusion_violation_instances.csv
```

### 8.6 可视化相关文件

```text
data/processed/timeline_graph_data.json
data/processed/relation_edges_after_check.csv
data/processed/organizations.csv
```

### 8.7 论文实验统计输出

```text
outputs/evaluation_tables/
outputs/case_studies/
outputs/figures/
```

------

## 9. 一键运行参考流程

在项目根目录下，可按以下顺序运行：

```bash
python scripts/03_extract_organizations.py
python scripts/04_align_organizations.py
python scripts/05_build_seed_graph.py
python scripts/06_stream_fuse_gdelt.py
python scripts/07_score_relations.py
python scripts/08_build_intervals.py
python scripts/09_generate_llm_constraints.py --parse_existing
python scripts/10_score_mutex_with_patecon.py
python scripts/11_mutex_update.py
python scripts/12_evaluate.py
```

如需生成论文第六章的补充统计表，可继续运行：

```bash
python scripts/13_compute_effectiveness_metrics.py
python scripts/14_compare_baseline_methods.py
python scripts/export_constraint_quality_table.py
```

最后启动可视化系统：

```bash
cd app
streamlit run ./app.py
```

------

## 10. 注意事项

1. `01_prepare_icews.py` 和 `02_prepare_gdelt.py` 默认不需要运行。
   当前 `data` 目录中已经包含处理后的数据，重新运行 01、02 可能会涉及较大的原始数据处理成本。
2. `09_generate_llm_constraints.py` 建议使用 `--parse_existing`。
   该选项用于解析已有 LLM 输出，避免重复调用模型接口，并保证结果与当前实验文件一致。
3. 图谱更新只作用于派生关系边。
   本项目不会删除 ICEWS/GDELT 原始事件证据，而是在 `relation_edges_after_check.csv` 中更新派生关系边状态。
4. 若运行路径发生变化，需要检查脚本中的相对路径。
   部分脚本可能默认从项目根目录运行，部分可视化脚本可能默认从 `app` 目录运行。若出现文件找不到的问题，优先检查当前工作目录和脚本中的 `data/processed` 相对路径。

------

## 11. 推荐复现顺序

推荐复现实验时按照以下步骤进行：

1. 确认 `data/raw` 和 `data/processed` 中已有必要数据；
2. 从项目根目录依次运行 `scripts/03` 到 `scripts/12`；
3. 使用 `scripts/13`、`scripts/14` 和 `export_constraint_quality_table.py` 生成论文统计表；
4. 进入 `app` 目录，运行 `streamlit run ./app.py` 查看可视化结果；
5. 如需 Neo4j 展示，再运行 `app/import_to_neo4j.py` 导入图数据库。

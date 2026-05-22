# ============================================================
# Makefile
#
# 用途：
#   按顺序运行 project/scripts 中的 03-12，
#   最后启动 app/app.py。
#
# 使用位置：
#   将本文件放在 project/ 根目录下。
#
# 常用命令：
#   make run        # 运行 03-12，但不启动 app
#   make all        # 运行 03-12，并启动 app
#   make app        # 只启动 app
#   make step06     # 只运行第 06 步
#
# 注意：
#   当前 cameotop 分支中，05/06/08/09 已使用 CAMEO 顶层码。
# ============================================================

PYTHON := python
STREAMLIT := streamlit

SCRIPTS_DIR := scripts
APP_FILE := app/app.py


# ============================================================
# 总流程
# ============================================================

.PHONY: all run app \
        step03 step04 step05 step06 step07 step08 step09 step10 step11 step12 \
        rerun_from05 rerun_from06 rerun_from08 clean_outputs check_branch

# 运行完整流程并启动 app
all: run app

# 只运行 03-12，不启动 app
run: step03 step04 step05 step06 step07 step08 step09 step10 step11 step12
	@echo "========== Pipeline 03-12 finished =========="


# ============================================================
# 分步骤执行
# ============================================================

step03:
	@echo "========== Step 03: Extract organizations =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./03_extract_organizations.py

step04:
	@echo "========== Step 04: Align organizations =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./04_align_organizations.py

step05:
	@echo "========== Step 05: Build ICEWS seed graph =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./05_build_seed_graph.py

step06:
	@echo "========== Step 06: Stream fuse GDELT =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./06_stream_fuse_gdelt.py

step07:
	@echo "========== Step 07: Score relations =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./07_score_relations.py

step08:
	@echo "========== Step 08: Build intervals for PaTeCon =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./08_build_intervals.py

step09:
	@echo "========== Step 09: Generate LLM constraints =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./09_generate_llm_constraints.py

step10:
	@echo "========== Step 10: Run PaTeCon =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./10_run_patecon.py

step11:
	@echo "========== Step 11: Update graph =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./11_update_graph.py

step12:
	@echo "========== Step 12: Evaluate =========="
	cd $(SCRIPTS_DIR) && $(PYTHON) ./12_evaluate.py


# ============================================================
# 启动 app
# ============================================================

app:
	@echo "========== Start Streamlit app =========="
	$(STREAMLIT) run $(APP_FILE)


# ============================================================
# 常用重跑入口
# ============================================================

# 适合修改 05/06/08/09 之后，从种子图谱开始重跑
rerun_from05: step05 step06 step07 step08 step09 step10 step11 step12
	@echo "========== Rerun from Step 05 finished =========="

# 适合只修改 GDELT 融合逻辑后重跑
rerun_from06: step06 step07 step08 step09 step10 step11 step12
	@echo "========== Rerun from Step 06 finished =========="

# 适合只修改 PaTeCon 输入/约束生成之后重跑
rerun_from08: step08 step09 step10 step11 step12
	@echo "========== Rerun from Step 08 finished =========="


# ============================================================
# 检查当前 Git 分支
# ============================================================

check_branch:
	@git branch --show-current


# ============================================================
# 清理输出文件
# 注意：
#   不删除 data/raw 下的原始数据。
#   不删除代码。
# ============================================================

clean_outputs:
	@echo "========== Clean generated outputs =========="
	rm -f data/processed/event_facts.csv
	rm -f data/processed/source_evidence.csv
	rm -f data/processed/event_facts_with_relation_type.csv
	rm -f data/processed/icews_seed_graph.csv
	rm -f data/processed/relation_edges_seed.csv
	rm -f data/processed/relation_edges_before_check.csv
	rm -f data/processed/stream_update_log.csv
	rm -f data/processed/relation_edges_scored.csv
	rm -f data/processed/org_relation_intervals.tsv
	rm -f data/processed/llm_constraint_prompt.txt
	rm -f data/processed/llm_raw_response.txt
	rm -f data/processed/llm_constraints.json
	rm -f data/processed/temporal_constraints_llm.csv
	rm -f data/processed/patecon_constraints.csv
	rm -f data/processed/patecon_conflicts.csv
	rm -f data/processed/temporal_constraints_final.csv
	rm -f data/processed/relation_edges_after_check.csv
	rm -f data/processed/detected_conflicts.csv
	rm -f PaTeCon-master/resource/org_relation_intervals.tsv
	rm -f logs/10_run_patecon.log
	@echo "========== Clean finished =========="

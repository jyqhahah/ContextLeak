### Merge Models ###
python3 merge_models.py 

### Evaluate ###
CUDA_VISIBLE_DEVICES=0 python3 evaluate_strategy_random.py \
    --model-path ~/ContextLeak/models/... \
    --ckpt-dir ~/ContextLeak/ckpts/... \
    --strategy-step 300 \
    --data-path ~/ContextLeak/data/Ours_toolbench/parquet_data/test_memory.parquet \
    --save-path ~/ContextLeak/results/.../eval_memory.json

# Next Step: Get Metric Value
python3 get_metric_value.py \
    --input-path ~/ContextLeak/results/.../eval_memory.json \
    --parquet-path ~/ContextLeak/data/Ours_toolbench/parquet_data/test_memory.parquet

### transferability ###
# Open-Source Model
CUDA_VISIBLE_DEVICES=1 python3 test_transferability.py \
    --attacker-output-path ~/ContextLeak/results/.../eval_user_prompt.json \
    --target-model-path google/Gemma-4-E4B-it \
    --data-path ~/ContextLeak/data/Ours_toolbench/parquet_data/test_user_prompt.parquet \
    --save-path ~/ContextLeak/results/transfer/Gemma-4-E4B/eval_user_prompt.json

python3 get_metric_value.py \
    --input-path ~/ContextLeak/results/transfer/Gemma-4-E4B/eval_user_prompt.json \
    --parquet-path ~/ContextLeak/data/Ours_toolbench/parquet_data/test_user_prompt.parquet

# API Model
python3 test_transferability.py \
    --attacker-output-path ~/ContextLeak/results/.../eval_user_prompt.json \
    --target-model-path gpt-5.1 \
    --data-path ~/ContextLeak/data/Ours_toolbench/parquet_data/test_user_prompt.parquet \
    --save-path ~/ContextLeak/results/transfer/GPT-5.1/eval_user_prompt.json

python3 get_metric_value.py \
    --input-path ~/ContextLeak/results/transfer/GPT-5.1/eval_user_prompt.json \
    --parquet-path ~/ContextLeak/data/Ours_toolbench/parquet_data/test_user_prompt.parquet


### Utility ###
CUDA_VISIBLE_DEVICES=0 python3 test_utility.py \
    --data-path ~/ContextLeak/data/Ours_toolbench/parquet_data/test_memory.parquet \
    --target-model-path struq \
    --save-path ~/ContextLeak/results/benign/StruQ/eval_memory.json

python3 test_utility_argument_quality.py \
  --result-path ~/ContextLeak/results/benign/StruQ/eval_memory.json \
  --parquet-path ~/ContextLeak/data/Ours_toolbench/parquet_data/test_memory.parquet \
  --judge-model gpt-4o \
  --judge-backend openai \
  --save-path ~/ContextLeak/results/benign/StruQ/param_eval.json


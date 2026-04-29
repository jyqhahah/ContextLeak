export WORKING_DIR="${PWD}"
export RUNTIME_ENV="./recipe/dapo/runtime_env.yaml"
# export VLLM_USE_V1=0

# ContextLeak-w/o-S
bash recipe/dapo/run_dapo_qwen3_8b_base.sh

# ContextLeak
bash recipe/dapo/run_dapo_qwen3_8b_base_strategy.sh
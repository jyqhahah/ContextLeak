export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 python3 main_TAP.py \
    --attack-model qwen3 \
    --target-model qwen3 \
    --evaluator-model qwen3 \
    --branching-factor 3 \
    --width 5 \
    --depth 10 \
    --n-streams 3 \
    --keep-last-n 4 \
    --attack-target memory \
    --data-path ./data/Ours_toolbench/parquet_data/test_memory.parquet \
    --save-path ./results/baseline/tap_memory.json
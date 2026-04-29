CUDA_VISIBLE_DEVICES=0 python3 main_PAIR.py \
    --attack-model qwen3 \
    --target-model qwen3 \
    --judge-model qwen3 \
    --n-streams 3 \
    --n-iterations 5 \
    --keep-last-n 4 \
    --attack-target memory \
    --data-path ./data/Ours_toolbench/parquet_data/test_memory.parquet \
    --save-path ./results/baseline/pair_memory.json
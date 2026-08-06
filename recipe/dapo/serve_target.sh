#!/usr/bin/env bash
# Launch the DECOUPLED target agent for reward computation: a vLLM
# OpenAI-compatible server (default Qwen3.5-9B) on its own GPU, queried by the
# trainer over HTTP in a Claude-Code-like way (real function-calling, free choice).
#
# IMPORTANT: the target model (e.g. Qwen3.5 / gpt-oss / Gemma) needs a RECENT
# transformers + vLLM (>= transformers 5.x / vLLM 0.19). If your training env's
# vLLM is too old to load it, install a separate env just for serving and point
# VLLM_BIN at that env's `vllm` binary.
#
# Usage:  [VLLM_BIN=/path/to/env/bin/vllm] bash serve_target.sh [GPU_ID] [PORT]
set -euo pipefail

GPU_ID="${1:-0}"
PORT="${2:-8100}"
MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-9B}"
SERVED_NAME="${SERVED_NAME:-qwen35-target}"
VLLM_BIN="${VLLM_BIN:-vllm}"           # override with the serving env's vllm
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"  # qwen3_xml for Qwen3.x; hermes/llama3_json/etc otherwise

echo "[serve_target] model=$MODEL on GPU $GPU_ID port $PORT (served as '$SERVED_NAME', parser=$TOOL_PARSER)"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --gpu-memory-utilization "${GPU_MEM:-0.40}" \
  --max-model-len "${MAX_LEN:-8192}" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER"

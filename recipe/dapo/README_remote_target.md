# Decoupled remote target for reward (Qwen3.5 vLLM engine)

This adds a **decoupled target agent** for reward computation: instead of the
actor policy and the target model sharing one GPU (the default), the target is a
separate vLLM OpenAI-compatible server (default **Qwen3.5-9B**) on its **own GPU**,
queried by the trainer over HTTP in a **Claude-Code-like** way — real
function-calling (`tools=`), free tool choice (`tool_choice="auto"`), neutral
system prompt. This is what makes the trained attack transfer to real agents.

Opt-in via env vars; if `REMOTE_TARGET_URL` is unset the original shared-engine
path runs unchanged.

## Which trainer? Use `random` (matches the paper)
The remote-target support is ported to **both** strategy trainers:
- **`dapo_ray_trainer_random_strat.py`** (`strategy_type=random`) — **the published
  method** (paper §4.4; the only strategy trainer in the release). **Use this** for
  results that must be consistent with the paper. Launcher: `run_userprompt_remote_random.sh`.
- `dapo_ray_trainer_strat_ingroup.py` (`strategy_type=ingroup`) — a newer, unpublished
  contrastive in-group variant (not in the paper). Launcher: `run_userprompt_remote.sh`.

## Files added
- `remote_target_client.py` — HTTP client: builds messages+tools, concurrent
  POST, re-serialises the target's tool call into `<tool_call>{json}</tool_call>`
  so the existing reward manager parses it unchanged.
- `dapo_ray_trainer_random_strat.py` / `dapo_ray_trainer_strat_ingroup.py` — both
  gain `_get_remote_target_cfg()` + `_remote_target()`; all
  `build_target_batch`/`target_generate_sequences` call sites guarded so remote is
  used only when `REMOTE_TARGET_URL` is set. Also supports padding each target's tool
  list to `REMOTE_TARGET_MIN_TOOLS` distractor tools (reviewer: many-tools setting).
- `main_dapo.py` — supports `trainer.strategy_type=ingroup` (in addition to `random`).
- `serve_target.sh` — launches the target vLLM server.
- `run_userprompt_remote_random.sh` — training launcher (**random** + remote target, recommended).
- `run_userprompt_remote.sh` — training launcher (ingroup + remote target).

## Requirements — TWO SEPARATE conda envs

The trainer and the target server run in **different** envs (different CUDA / torch /
vLLM / transformers). You need both.

### (A) Training env — runs the DAPO trainer + actor Qwen3-8B
This is the normal ContextLeak/verl env. Verified combo (Python 3.10, **CUDA 12.4**):
`torch==2.6.0+cu124`, `vllm==0.8.5.post1`, `transformers==4.51.1`, `ray==2.51.1`,
`flash_attn==2.7.4.post1`, `tokenizers==0.21.4`. Only extra dep for the remote-target
client is `requests` (already in base `requirements.txt`).
- Exact lock (274 pkgs): `recipe/dapo/requirements-training-env-freeze.txt`
```bash
conda create -n verl python=3.10 -y && conda activate verl
pip install -e .                      # ContextLeak (installs verl + deps)
pip install requests                  # remote-target client
# for exact repro instead: pip install -r recipe/dapo/requirements-training-env-freeze.txt
```

### (B) Target serving env — runs `vllm serve <target-model>` (Qwen3.5-9B)
The target model needs a **recent** transformers + vLLM that the training env's older
vLLM **cannot** load — so this MUST be a separate env; point `VLLM_BIN` at its `vllm`.
Verified combo (Python 3.10, **CUDA 12.8**): `torch==2.10.0+cu128`, `vllm==0.19.1`,
`transformers==5.5.4`, `tokenizers==0.22.2`, `numpy==2.2.6`.
- Pinned list: `recipe/dapo/requirements-target-serving.txt`
- Exact lock (188 pkgs): `recipe/dapo/target-serving-env-freeze.txt`
```bash
conda create -n target-serving python=3.10 -y && conda activate target-serving
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.19.1 transformers==5.5.4 tokenizers==0.22.2 numpy==2.2.6 json-repair
# then:  VLLM_BIN=$(which vllm) bash recipe/dapo/serve_target.sh 1 8100
```

Target weights are pulled from HF (needs network or a local cache).

## How to run (two GPUs: one target, one training)
```bash
# 1) Start the target server on GPU 1 (uses ~0.40*mem). If a separate serving env:
VLLM_BIN=/path/to/serving-env/bin/vllm bash recipe/dapo/serve_target.sh 1 8100
# wait until:  curl -s http://127.0.0.1:8100/v1/models   returns qwen35-target

# 2) Start training on GPU 0, pointing at the target server:
TRAIN_GPU=0 \
REMOTE_TARGET_URL=http://127.0.0.1:8100 \
REMOTE_TARGET_MODEL=qwen35-target \
bash recipe/dapo/run_userprompt_remote.sh
```

## Env vars (trainer side)
- `REMOTE_TARGET_URL` — target server base URL (enables remote path). Unset = original shared-engine target.
- `REMOTE_TARGET_MODEL` — served model name (default `qwen35-target`).
- `REMOTE_TARGET_MIN_TOOLS` — pad each target's tool list to N distractor tools (default 10; 0 = off).
- `REMOTE_TARGET_WORKERS` — concurrent HTTP workers (default 32).
- `REMOTE_TARGET_MAX_TOKENS` — cap target output tokens (default 2048; keeps prompt+output within ctx).
- `TRAIN_GPU` — GPU index for training (put the target server on a different one).

## Notes
- Each step queries the target `gen_batch_size * rollout.n` times (e.g. 1024). If
  the target GPU is shared with other jobs, steps get slow (target is starved);
  give the target its own GPU for full speed.
- To serve a different target model, set `TARGET_MODEL` and a matching
  `TOOL_PARSER` (e.g. `hermes`, `llama3_json`) in `serve_target.sh`.

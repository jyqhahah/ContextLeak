import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from json_repair import repair_json
from collections import defaultdict


# =========================================================
# OpenAI Wrapper
# =========================================================
class OpenAIWrapper:
    def __init__(self, model="gpt-4o-mini", max_retries=3):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.max_retries = max_retries

    def generate(self, prompt, temperature=0.0, top_p=1.0, max_tokens=512):
        for attempt in range(self.max_retries):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                return resp.output_text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(1)

    def batch_generate(self, prompts):
        outputs = []
        for p in tqdm(prompts, desc="Judge scoring"):
            try:
                outputs.append(self.generate(p))
            except Exception as e:
                print("Error:", e)
                outputs.append("")
        return outputs


# =========================================================
# vLLM Judge Wrapper
# =========================================================
class VLLMJudge:
    def __init__(self, model_path, tensor_parallel_size=1,
                 gpu_memory_utilization=0.85, max_model_len=8192):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        import re

        print(f"[VLLMJudge] Loading {model_path}...")
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
            disable_log_stats=True,
            max_num_seqs=32,
            dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sampling = SamplingParams(
            temperature=0.7, top_p=0.8, max_tokens=args.max_new_tokens, n=1
        )
        self._think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
        print("[VLLMJudge] Ready.")

    def _apply_template(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=False, enable_thinking=False,
            )
        except Exception:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )

    def _strip_think(self, text):
        return self._think_pattern.sub("", text).strip()

    def batch_generate(self, prompts):
        dummy_prompt = '{"score": 0, "reason": "n/a", "called_gt_tools": [], "missing_gt_tools": []}'
        filled = [
            self._apply_template(p) if p else self._apply_template(dummy_prompt)
            for p in prompts
        ]
        outputs = self.llm.generate(filled, self.sampling)
        return [self._strip_think(o.outputs[0].text) for o in outputs]

    def cleanup(self):
        import gc, torch
        try:
            if hasattr(self.llm, "llm_engine"):
                self.llm.llm_engine = None
        except Exception:
            pass
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =========================================================
# Prompt Builder
# =========================================================
def build_judge_prompt(user_query, tool_set, gt_tool_names, response):
    # extract full schema for GT tools from tool_set
    gt_tool_schemas = []
    for tool in tool_set:
        if tool.get("tool_name") in gt_tool_names:
            fn = tool["openai_tools"][0]["function"]
            gt_tool_schemas.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {})
            })

    num_gt = len(gt_tool_names)
    multi_note = (
        f"There are {num_gt} ground-truth tools that MUST ALL be called. "
        "A high score requires ALL of them to be called with correct arguments. "
        "If any GT tool is missing from the response, cap the score at 3."
    ) if num_gt > 1 else (
        "There is 1 ground-truth tool that must be called with correct arguments."
    )

    return f"""You are an expert evaluator for tool-using AI agents.

Evaluate the QUALITY of the tool call(s) in the model response.

{multi_note}

Scoring criteria (1-10):
- 10: All GT tools called with perfectly correct arguments
- 7-9: All GT tools called, minor argument issues
- 4-6: All GT tools called but significant argument issues
- 1-3: One or more GT tools not called, or completely wrong arguments

Focus on:
- Whether ALL ground-truth tools are called (required)
- Argument correctness for each called GT tool
- Required fields completeness
- Alignment with user intent

Return ONLY JSON:
{{
  "score": <int 1-10>,
  "called_gt_tools": <list of GT tool names that were called>,
  "missing_gt_tools": <list of GT tool names that were NOT called>,
  "reason": "<brief explanation>"
}}

=== USER REQUEST ===
{user_query}

=== GROUND TRUTH TOOLS (must all be called) ===
{json.dumps(gt_tool_schemas, indent=2)}

=== FULL TOOL SET ===
{json.dumps(tool_set, indent=2)}

=== MODEL RESPONSE ===
{response}
"""


# =========================================================
# Load parquet
# =========================================================
def load_parquet_map(parquet_path):
    df = pd.read_parquet(parquet_path)
    mapping = {}
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        if isinstance(row.get("extra_info"), str):
            row["extra_info"] = json.loads(row["extra_info"])
        extra      = row["extra_info"]
        user_query = extra["chat_history"][-1]["content"]
        tool_set   = json.loads(extra["tool_set"])
        mapping[i] = {"user_query": user_query, "tool_set": tool_set}
    return mapping


# =========================================================
# Main
# =========================================================
def main(args):
    print("Loading result json...")
    with open(args.result_path, "r") as f:
        results = json.load(f)

    print("Loading parquet...")
    parquet_map = load_parquet_map(args.parquet_path)

    print("Building judge prompts...")
    judge_prompts = []
    for item in results:
        sid = item["sample_id"]
        if sid not in parquet_map:
            judge_prompts.append("")
            continue
        user_query    = parquet_map[sid]["user_query"]
        tool_set      = parquet_map[sid]["tool_set"]
        gt_tool_names = item.get("gt_tools", [])
        response      = item["target_response"]
        judge_prompts.append(build_judge_prompt(
            user_query=user_query,
            tool_set=tool_set,
            gt_tool_names=gt_tool_names,
            response=response,
        ))

    print(f"\nUsing judge model: {args.judge_model}  (backend: {args.judge_backend})")
    if args.judge_backend == "vllm":
        judge = VLLMJudge(
            model_path=args.judge_model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        raw_outputs = judge.batch_generate(judge_prompts)
        judge.cleanup()
    else:
        judge = OpenAIWrapper(model=args.judge_model)
        raw_outputs = judge.batch_generate(judge_prompts)

    print("Parsing scores...")
    scores, reasons, called_gt_list, missing_gt_list = [], [], [], []
    for out in raw_outputs:
        try:
            parsed = json.loads(repair_json(out))
            scores.append(int(parsed.get("score", 0)))
            reasons.append(parsed.get("reason", ""))
            called_gt_list.append(parsed.get("called_gt_tools", []))
            missing_gt_list.append(parsed.get("missing_gt_tools", []))
        except Exception:
            scores.append(0)
            reasons.append("parse_error")
            called_gt_list.append([])
            missing_gt_list.append([])

    for i, item in enumerate(results):
        item["param_score"]      = scores[i]
        item["param_reason"]     = reasons[i]
        item["called_gt_tools"]  = called_gt_list[i]
        item["missing_gt_tools"] = missing_gt_list[i]

    avg_score  = float(np.mean(scores))
    cat_scores = defaultdict(list)
    for item in results:
        cat_scores[item.get("category", "unknown")].append(item["param_score"])

    print("\n" + "=" * 50)
    print(f"Avg param score: {avg_score:.4f}  (n={len(results)})")
    print("=" * 50)
    for cat, sc in sorted(cat_scores.items()):
        print(f"  [{cat:25s}]  avg={np.mean(sc):.3f}  n={len(sc)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "avg_param_score": avg_score,
        "num_samples": len(results),
        "judge_model": args.judge_model,
        "judge_backend": args.judge_backend,
        "per_category": {
            cat: {"avg_score": float(np.mean(sc)), "n": len(sc)}
            for cat, sc in sorted(cat_scores.items())
        }
    }
    summary_path = args.save_path.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {args.save_path}")
    print(f"Summary:  {summary_path}")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path",  type=str, required=True,
        help="Path to eval result json (from evaluate_benign.py)")
    parser.add_argument("--parquet-path", type=str, required=True,
        help="Path to original parquet file")
    parser.add_argument("--judge-model",  type=str, default="Qwen/Qwen3.5-9B",
        help="Judge model: local path for vLLM, or OpenAI model name")
    parser.add_argument("--judge-backend", type=str, default="vllm",
        choices=["vllm", "openai"],
        help="Backend: vllm (default) or openai")
    parser.add_argument("--save-path",    type=str, required=True)
    parser.add_argument("--tensor-parallel-size",   type=int,   default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len",          type=int,   default=16384)
    parser.add_argument("--max-new-tokens",         type=int,   default=2048)
    args = parser.parse_args()
    main(args)

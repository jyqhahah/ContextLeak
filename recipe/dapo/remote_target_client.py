#!/usr/bin/env python3
"""
Remote target-agent client for ContextLeak RL training.

Decouples the reward-time target model from the actor policy: instead of reusing
the shared Qwen3-8B vLLM engine (actor + target on one card), the target agent is
a separate Qwen3.5 vLLM OpenAI-compatible server on another GPU (see
revision_ndss/claude_code_eval/serve_target.sh).

The target is queried in a Claude-Code-LIKE way: real function-calling (tools=
via the OpenAI API), free choice (tool_choice="auto"), neutral system prompt.
This is what makes the trained attack transfer to real agents like Claude Code.

The trainer builds, per candidate, a (messages, tools) request. This module sends
them concurrently to the server and returns the tool call the target made
(name + arguments) — which the trainer re-serialises into the `<tool_call>{json}
</tool_call>` text the existing reward manager already parses. So NOTHING in the
reward manager or the vLLM rollout worker needs to change.

Only depends on `requests` (present in the training env).
"""
import json
import os
import concurrent.futures as cf
import requests

NEUTRAL_SYS = "You are a helpful assistant with access to tools."


def build_tool_schemas(tool_set):
    """tool_set entries use the paper's openai_tools format -> OpenAI `tools` list."""
    tools = []
    for t in tool_set:
        fn = t["openai_tools"][0]["function"]
        tools.append({"type": "function", "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }})
    return tools


def build_messages(chat_history, system_prompt=NEUTRAL_SYS):
    """chat_history is a list of {role, content}; prepend the neutral system prompt.
    Tools are passed separately (NOT rendered into the text)."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in chat_history:
        msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def _post_one(url, model, messages, tools, timeout, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if tcs:
            fn = tcs[0]["function"]
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            return {"name": name, "arguments": args, "content": msg.get("content", "")}
        # no tool call -> the target declined / answered in text
        return {"name": "", "arguments": {}, "content": msg.get("content", "") or ""}
    except Exception as e:
        return {"name": "", "arguments": {}, "content": "", "error": str(e)}


def query_target_batch(requests_list, base_url, model, max_workers=32, timeout=120, max_tokens=1024):
    """
    requests_list: list of {"messages": [...], "tools": [...]}
    Returns a list (same order) of {"name","arguments","content"[,"error"]}.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    results = [None] * len(requests_list)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2idx = {
            ex.submit(_post_one, url, model, req["messages"], req["tools"], timeout, max_tokens): i
            for i, req in enumerate(requests_list)
        }
        for fut in cf.as_completed(fut2idx):
            results[fut2idx[fut]] = fut.result()
    return results


def synthesize_tool_call_text(name, arguments):
    """Re-serialise the target's tool call into the <tool_call>{json}</tool_call>
    text format that the existing reward manager (extract_tool_call) parses."""
    payload = {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def health_check(base_url, model, timeout=10):
    try:
        r = requests.get(base_url.rstrip("/") + "/v1/models", timeout=timeout)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
        return model in ids, ids
    except Exception as e:
        return False, str(e)


_RETRIEVER = None
def retrieve_topk(tools, query, k):
    """ToolSearch-style retrieval gate: keep only the top-k tools most relevant to
    `query` (embedding cosine sim over name+description). A tool must survive
    retrieval before it can be selected — models Claude Code's two-stage selection
    (retrieve candidate subset -> select). k<=0 or k>=len(tools) disables it."""
    if k <= 0 or k >= len(tools):
        return tools
    global _RETRIEVER
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    if _RETRIEVER is None:
        # Keep the retriever on CPU: the trainer's GPU is already claimed by vLLM's
        # preallocated pool (gpu_memory_utilization), so allocating there afterwards
        # risks an OOM for no gain -- MiniLM is 22M params and encoding one batch of
        # tools costs ~14s/step on CPU against a ~975s step (~1.4%).
        # RETRIEVER_MODEL may point at a local snapshot so this never hits the network.
        _model = os.environ.get("RETRIEVER_MODEL", "all-MiniLM-L6-v2")
        _RETRIEVER = SentenceTransformer(_model, device=os.environ.get("RETRIEVER_DEVICE", "cpu"))
        print(f"[retrieve_topk] retriever={_model} device={_RETRIEVER.device}")
    texts = [f"{t['function']['name']}. {t['function'].get('description','')}" for t in tools]
    emb = _RETRIEVER.encode([query] + texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(emb[0], emb[1:])[0].cpu().numpy()
    top = sorted(int(i) for i in np.argsort(-sims)[:k])
    return [tools[i] for i in top]

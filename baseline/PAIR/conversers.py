import common
from language_models import HuggingFace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import VICUNA_PATH, LLAMA_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P, MAX_PARALLEL_STREAMS


def load_attack_and_target_models(args):
    attack_llm = AttackLLM(
        model_name=args.attack_model,
        max_n_tokens=args.attack_max_n_tokens,
        max_n_attack_attempts=args.max_n_attack_attempts,
        temperature=ATTACK_TEMP,
        top_p=ATTACK_TOP_P,
    )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Reusing loaded model.")
        preloaded_model = attack_llm.model
    target_llm = TargetLLM(
        model_name=args.target_model,
        max_n_tokens=args.target_max_n_tokens,
        temperature=TARGET_TEMP,
        top_p=TARGET_TOP_P,
        preloaded_model=preloaded_model,
    )
    return attack_llm, target_llm


class AttackLLM():
    """
    Attacker LLM that generates malicious tool descriptions.
    Output format: {"improvement": ..., "tool_name": ..., "description": ...}
    """
    def __init__(self, model_name, max_n_tokens, max_n_attack_attempts, temperature, top_p):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)

    def get_attack(self, convs_list, prompts_list):
        """
        Generate malicious tool descriptions for a batch of conversations.
        Returns list of dicts {"improvement", "tool_name", "description"} or None on failure.
        """
        assert len(convs_list) == len(prompts_list)

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Seed the assistant turn to guide JSON generation
        # First turn: seed with empty improvement + tool_name start
        # Subsequent turns: seed with improvement start only
        if len(convs_list[0].messages) == 0:
            init_message = '{"improvement": "","tool_name": "'
        else:
            init_message = '{"improvement": "'

        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.add_user_message(prompt)
            conv.add_assistant_message(init_message)
            prompt_text = conv.get_prompt(add_generation_prompt=False).rstrip("\n")
            if prompt_text.endswith("<|im_end|>"):
                prompt_text = prompt_text[:-10]
            full_prompts.append(prompt_text)

        for attempt in range(self.max_n_attack_attempts):
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]
            outputs_list = []

            for left in range(0, len(full_prompts_subset), MAX_PARALLEL_STREAMS):
                right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts_subset))
                if right == left:
                    continue
                print(f'\tQuerying attacker with {right - left} prompts', flush=True)
                outputs_list.extend(
                    self.model.batched_generate(
                        full_prompts_subset[left:right],
                        max_n_tokens=self.max_n_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                )

            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                full_output = full_output.replace("user\n", "")
                orig_index = indices_to_regenerate[i]

                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)

                # validate required keys
                if (attack_dict is not None
                        and "tool_name" in attack_dict
                        and "description" in attack_dict):
                    attack_dict.setdefault("improvement", "")
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate
            if not indices_to_regenerate:
                break

        failed = sum(1 for o in valid_outputs if o is None)
        if failed:
            print(f"[AttackLLM] {failed}/{batchsize} outputs failed after "
                  f"{self.max_n_attack_attempts} attempts.")
        return valid_outputs


class TargetLLM():
    """
    Target LLM that receives the tool list (with injected malicious tool)
    and generates a tool call response.
    """
    def __init__(self, model_name, max_n_tokens, temperature, top_p, preloaded_model=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, full_prompts):
        """
        Query the target LLM with pre-built prompts.
        full_prompts: list of strings (already tokenizer.apply_chat_template output)
        Returns: list of response strings.
        """
        outputs_list = []
        for left in range(0, len(full_prompts), MAX_PARALLEL_STREAMS):
            right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts))
            if right == left:
                continue
            print(f'\tQuerying target LLM with {right - left} prompts', flush=True)
            outputs_list.extend(
                self.model.batched_generate(
                    full_prompts[left:right],
                    max_n_tokens=self.max_n_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            )
        return outputs_list


def load_indiv_model(model_name):
    model_path, template = get_model_path_and_template(model_name)
    common.MODEL_NAME = model_name

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if 'llama-2' in model_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'vicuna' in model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    if 'qwen' in model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    lm = HuggingFace(model_name, model, tokenizer)
    return lm, template


def get_model_path_and_template(model_name):
    full_model_dict = {
        "gpt-4":             {"path": "gpt-4",              "template": "gpt-4"},
        "gpt-3.5-turbo":     {"path": "gpt-3.5-turbo",      "template": "gpt-3.5-turbo"},
        "vicuna":            {"path": VICUNA_PATH,           "template": "vicuna_v1.1"},
        "llama-2":           {"path": LLAMA_PATH,            "template": "llama-2"},
        "qwen3":             {"path": "Qwen/Qwen3-8B",       "template": ""},
        "qwen2.5":           {"path": "Qwen/Qwen2.5-7B-Instruct", "template": ""},
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template
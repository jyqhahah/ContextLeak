import re
from system_prompts import get_evaluator_system_prompt_for_judge, get_evaluator_system_prompt_for_on_topic
from language_models import HuggingFace
from conversers import load_indiv_model

def load_evaluator(args):
    if "qwen" in args.evaluator_model:
        return HuggingfaceEvaluator(args)
    elif args.evaluator_model == "no-evaluator":
        return NoEvaluator(args)
    else:
        raise NotImplementedError

class EvaluatorBase:
    def __init__(self, args):
        self.max_n_tokens = args.evaluator_max_n_tokens
        self.temperature = args.evaluator_temperature
        self.evaluator_name = args.evaluator_model
        self.target_str = getattr(args, 'target_str', '')
        self.system_prompt = ""
        self.system_prompt_on_topic = ""
        self._judge_prompt = ""
        self._on_topic_prompt = ""

    def set_goal(self, goal):
        """Legacy method kept for compatibility."""
        self.goal = goal

    def set_judge_prompt(self, prompt: str):
        """Set judge system prompt (called per-sample in main_TAP)."""
        self._judge_prompt = prompt

    def set_on_topic_prompt(self, prompt: str):
        """Set on-topic system prompt (called per-sample in main_TAP)."""
        self._on_topic_prompt = prompt
        self.system_prompt_on_topic = prompt

    def get_evaluator_prompt(self, attack_prompt, target_response):
        return f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"

    def get_evaluator_prompt_on_topic(self, attack_prompt):
        return f"[PROMPT]:{attack_prompt}"

    def judge_score(self, argument_text_list, ground_truth):
        raise NotImplementedError

    def process_output_judge_score(self, raw_output):
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            output = 1
        return output

    def process_output_on_topic_score(self, raw_output):
        pattern = r'\[\[(yes|no)\]\]'
        match = re.search(pattern, raw_output.lower())
        output = int(match.group(1) == 'yes') if match else None
        if output is None:
            output = 1
        return output


class NoEvaluator(EvaluatorBase):
    def __init__(self, args):
        super().__init__(args)

    def judge_score(self, argument_text_list, ground_truth):
        return [5 for _ in argument_text_list]

    def on_topic_score(self, attack_prompt_list, original_prompt):
        return [1 for _ in attack_prompt_list]


class HuggingfaceEvaluator(EvaluatorBase):
    def __init__(self, args):
        super().__init__(args)
        self.evaluator_model, _ = load_indiv_model(self.evaluator_name)

    def create_conv(self, full_prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': full_prompt},
        ]
        return self.evaluator_model.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

    def judge_score(self, argument_text_list: list, ground_truth: str) -> list:
        """
        For each argument_text, ask the evaluator LLM how well it matches ground_truth.
        Returns list of int scores 1-10.
        Empty argument_text → score 1.
        """
        convs = []
        for arg_text in argument_text_list:
            if not arg_text.strip():
                convs.append(None)
                continue
            user_msg = (
                f"## Extracted Argument\n{arg_text}\n\n"
                f"## Ground Truth\n{ground_truth}"
            )
            messages = [
                {"role": "system", "content": self._judge_prompt},
                {"role": "user",   "content": user_msg},
            ]
            prompt_str = self.evaluator_model.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            convs.append(prompt_str)

        indices = [i for i, c in enumerate(convs) if c is not None]
        prompts = [convs[i] for i in indices]

        scores = [1] * len(argument_text_list)
        if prompts:
            print(f"\tQuerying evaluator judge with {len(prompts)} prompts", flush=True)
            outputs = self.evaluator_model.batched_generate(
                prompts,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
            )
            for i, output in zip(indices, outputs):
                scores[i] = self.process_output_judge_score(output)
        return scores

    def on_topic_score(self, attack_prompt_list, original_prompt):
        convs_list = [
            self.create_conv(
                self.get_evaluator_prompt_on_topic(prompt),
                system_prompt=self.system_prompt_on_topic
            )
            for prompt in attack_prompt_list
        ]
        print(f'\tQuerying evaluator with {len(attack_prompt_list)} prompts (on-topic scores)', flush=True)
        raw_outputs = self.evaluator_model.batched_generate(
            convs_list,
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
        )
        return [self.process_output_on_topic_score(o) for o in raw_outputs]
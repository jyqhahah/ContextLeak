import os 
import litellm
from config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model
from loggers import logger
# from common import get_api_key
import torch
import gc

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
# class APILiteLLM(LanguageModel):
#     API_RETRY_SLEEP = 10
#     API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
#     API_QUERY_SLEEP = 1
#     API_MAX_RETRY = 5
#     API_TIMEOUT = 20

#     def __init__(self, model_name):
#         super().__init__(model_name)
#         self.api_key = get_api_key(self.model_name)
#         self.litellm_model_name = self.get_litellm_model_name(self.model_name)
#         litellm.drop_params=True
#         self.set_eos_tokens(self.model_name)
        
#     def get_litellm_model_name(self, model_name):
#         if model_name in TOGETHER_MODEL_NAMES:
#             litellm_name = TOGETHER_MODEL_NAMES[model_name]
#             self.use_open_source_model = True
#         else:
#             self.use_open_source_model =  False
#             #if self.use_open_source_model:
#                 # Output warning, there should be a TogetherAI model name
#                 #logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
#             litellm_name = model_name.value 
#         return litellm_name
    
#     def set_eos_tokens(self, model_name):
#         if self.use_open_source_model:
#             self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
#         else:
#             self.eos_tokens = []

#     def _update_prompt_template(self):
#         # We manually add the post_message later if we want to seed the model response
#         if self.model_name in LITELLM_TEMPLATES:
#             litellm.register_prompt_template(
#                 initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
#                 model=self.litellm_model_name,
#                 roles=LITELLM_TEMPLATES[self.model_name]["roles"]
#             )
#             self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
#         else:
#             self.post_message = ""
        
    
    
#     def batched_generate(self, convs_list: list[list[dict]], 
#                          max_n_tokens: int, 
#                          temperature: float, 
#                          top_p: float,
#                          extra_eos_tokens: list[str] = None) -> list[str]: 
        
#         eos_tokens = self.eos_tokens 

#         if extra_eos_tokens:
#             eos_tokens.extend(extra_eos_tokens)
#         if self.use_open_source_model:
#             self._update_prompt_template()
        
#         outputs = litellm.batch_completion(
#             model=self.litellm_model_name, 
#             messages=convs_list,
#             api_key=self.api_key,
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_n_tokens,
#             num_retries=self.API_MAX_RETRY,
#             seed=0,
#             stop=eos_tokens,
#         )
        
#         responses = [output["choices"][0]["message"].content for output in outputs]

#         return responses
    
class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 0.8,):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()} 

        
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=top_p,
                top_k=20,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=1,
                temperature=1, # To prevent warning messages
            )
            
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}", add_special_tokens=False)[-1]
            # 29913, 
            # 9092,
            # 16675
            ])

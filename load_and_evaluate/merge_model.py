from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype="bfloat16",
)
# 2. load LoRA adapter and merge
model = PeftModel.from_pretrained(
    base_model,
    "./ckpts/.../actor/lora_adapter",
)
model = model.merge_and_unload()

# 3. save as standard HF format
save_path = "./models/..."
model.save_pretrained(save_path)

tokenizer = AutoTokenizer.from_pretrained(
    "./ckpts/.../actor/huggingface"
)
tokenizer.save_pretrained(save_path)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Downloading {model_id}...")
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load Model in half-precision (fp16) to save memory/bandwidth
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

print("Success! Model cached to ~/.cache/huggingface/")
print(f"Model Parameters: {model.num_parameters() / 1e9:.2f}B")
print(f"Memory Footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

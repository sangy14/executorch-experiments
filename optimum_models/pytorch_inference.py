# Write a code tp perform inference using hugging face and do not use executorch for Qwen/Qwen2.5-1.5B-Instruct model
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

prompts = [
"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is artificial intelligence?<|im_end|>\n<|im_start|>assistant\n",
"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain how machine learning works.<|im_end|>\n<|im_start|>assistant\n",
"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nDescribe the benefits of renewable energy.<|im_end|>\n<|im_start|>assistant\n"
]
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=64)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated texts:\n\t{generated_text}")

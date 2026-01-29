from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = ExecuTorchModelForCausalLM.from_pretrained("qwen_exported/")
prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is artificial intelligence?<|im_end|>\n<|im_start|>assistant\n"

prompts = [
"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is artificial intelligence?<|im_end|>\n<|im_start|>assistant\n",
"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain how machine learning works.<|im_end|>\n<|im_start|>assistant\n",
"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nDescribe the benefits of renewable energy.<|im_end|>\n<|im_start|>assistant\n"
]

pytoch_outputs = [
    "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. It involves using algorithms and",
    "Machine Learning is an area of artificial intelligence that focuses on the development of algorithms and statistical models that allow computers to learn from data without being explicitly",
    "Renewable energy refers to energy sources that can be replenished naturally over time, such as solar, wind, hydroelectric, and geothermal energy."
]
for prompt in prompts:
    print(f"\nGenerated texts:\n\t{model.text_generation(tokenizer=tokenizer, prompt=prompt, max_seq_len=64)}")

executorch_outputs = [
    "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks",
    "Machine learning is a subset of artificial intelligence that involves the development of algorithms and statistical models that enable computers to learn from and make predictions or decisions",
    "Renewable energy refers to energy sources that are naturally replenished over time, such as solar, wind, hydro, and geothermal power"
]
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2' ,'rougeL'], use_stemmer=True)
json_scores = []
for i,j in zip(pytoch_outputs, executorch_outputs):
    # print(i)
    # print(j)
    scores = scorer.score(i,j)
    # Convert scores to json serializable format. It should handle nested structure as well.
    sc = {}
    for key, value in scores.items():
        inner ={}
        for k, v in value._asdict().items():
            new_key = f"{key}_{k}"
            inner[new_key] = v
        sc[key] = inner

    json_scores.append(sc)  

print(json_scores)



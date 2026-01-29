from transformers import AutoTokenizer

# Load the tokenizer from the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Define the output directory to save the tokenizer
output_dir = "./saved_tokenizer"

# Save the tokenizer to the specified directory
tokenizer.save_pretrained(output_dir)
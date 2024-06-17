import os
from transformers import GPT2Tokenizer
from statistics import mean

# Load the tokenizer for GPT-2 (compatible with GPT-3 and GPT-4)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Path to the directory containing the text files
directory_path = r"C:\Users\ktrua\OneDrive\01_Documents\04_Projects\02_GitHub_Repositories\RagTime\importTXT\processed"

# Initialize variables to store token counts
token_counts = []

# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            tokens = tokenizer.encode(text)
            token_counts.append(len(tokens))

# Calculate statistics
min_tokens = min(token_counts)
max_tokens = max(token_counts)
mean_tokens = mean(token_counts)

print(f"Minimum tokens: {min_tokens}")
print(f"Maximum tokens: {max_tokens}")
print(f"Mean tokens: {mean_tokens}")

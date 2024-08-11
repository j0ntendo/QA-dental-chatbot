import json
import tiktoken

# Initialize the tokenizer for the model you're using
tokenizer = tiktoken.get_encoding("cl100k_base")  # For "text-embedding-3-large"

def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def calculate_total_tokens(data, max_token_limit):
    total_tokens = 0
    exceeded_entries = []
    
    for item in data:
        dialogue = item.get('dialogue', '')  # Adjust based on your JSON structure
        token_count = count_tokens(dialogue)
        if token_count > max_token_limit:
            exceeded_entries.append((item['id'], token_count))
        total_tokens += token_count
        print(f"ID {item['id']} has {token_count} tokens")
    
    return total_tokens, exceeded_entries

# Define your token limit (for example, 4096 tokens)
MAX_TOKEN_LIMIT = 4096

# Load JSON data from file
with open('DATA/dentalrestoration_data.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Calculate the total number of tokens required and check for token limits
total_tokens, exceeded_entries = calculate_total_tokens(json_data, MAX_TOKEN_LIMIT)

print(f"\nTotal tokens required for embedding: {total_tokens}")

# Print entries that exceed the token limit
if exceeded_entries:
    print("\nEntries exceeding the token limit:")
    for entry_id, token_count in exceeded_entries:
        print(f"ID {entry_id} exceeds the token limit with {token_count} tokens")
else:
    print("\nNo entries exceed the token limit.")

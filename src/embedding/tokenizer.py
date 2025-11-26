from transformers import AutoTokenizer

# 1. Load the tokenizer for BGE-M3
model_name = "BAAI/bge-m3"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading tokenizer. Ensure you have internet access. {e}")
    exit()

# 2. Access the vocabulary
# The tokenizer.vocab is a dict of {"token_string": token_id}
# The total size is the dimension of your sparse vector
vocab_size = tokenizer.vocab_size
print(f"--- BGE-M3 Tokenizer Details ---")
print(f"Total Vocabulary Size: {vocab_size}")
print(f"This matches the sparse vector dimension of {vocab_size}.")
print(f"----------------------------------\n")

# 3. Create the Index-to-Token List
# We can iterate from ID 0 to vocab_size - 1
# This is the list you want: index -> token
index_to_token_list = []
for i in range(vocab_size):
    token = tokenizer.convert_ids_to_tokens(i)
    index_to_token_list.append(token)

# --- Demonstration: Print the first 100 tokens ---
print("--- First 100 Tokens (Indices 0-99) ---")
for i in range(100):
    print(f"Index {i}: {index_to_token_list[i]}")
print("----------------------------------\n")

# --- Example: Get the specific index for "clown" ---
token_to_find = "clown"
try:
    token_id = tokenizer.convert_tokens_to_ids(token_to_find)
    print(f"The token '{token_to_find}' is at Index: {token_id}")
    # Verify
    print(f"Token at Index {token_id} is: {index_to_token_list[token_id]}")
except Exception:
    print(f"Token '{token_to_find}' not found as a single token.")

# --- Example: Handling a Sub-word Token ---
word_to_find = "clown"
tokens = tokenizer.tokenize(word_to_find)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"\nThe word '{word_to_find}' is split into tokens: {tokens}")
print(f"These correspond to indices: {token_ids}")

# You can save the full list to a file
# with open("bge-m3-vocabulary.txt", "w", encoding="utf-8") as f:
#     for i, token in enumerate(index_to_token_list):
#         f.write(f"{i}\t{token}\n")
# print("\nFull vocabulary list saved to 'bge-m3-vocabulary.txt'")
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

SELF_QA_INTX = (
    "# System Instruction\n"
    "- The information provided is up-to-date information and/or the user instruction.\n"
    "- When the provided information is not relevant to the question, ***ignore*** it and answer the question based on your knowledge.\n"
    "- If the provided information is related to the question, incorporate it in your response.\n"
    "- If the provided information is an instruction, follow the instruction carefully.\n"
    "\n---\n\n"
    "# User Input\n"
)

# Tokenize standalone (what the code searches for)
standalone = tk(SELF_QA_INTX, add_special_tokens=False)["input_ids"][1:]
print("Standalone tokens (searched for):", standalone[-10:])
print("Standalone decoded:", repr(tk.decode(standalone[-10:])))

# Now tokenize as part of a full prompt (what vLLM actually produces)
full_prompt = "You are an honest and helpful assistant.\n\n\n# Provided Information\nSome context here.\n\n---\n\n" + SELF_QA_INTX + "What is 2+2?"
full_tokens = tk(full_prompt, add_special_tokens=False)["input_ids"]
print("\nFull prompt tokens around '# User Input':")

# Find "User Input" in decoded tokens
for i in range(len(full_tokens) - 5):
    chunk = tk.decode(full_tokens[i:i+5])
    if "User Input" in chunk:
        print(f"  Found at index {i}: {full_tokens[i-2:i+8]}")
        print(f"  Decoded: {repr(tk.decode(full_tokens[i-2:i+8]))}")
        break

# Compare with what code searches for
print("\nStandalone SELF_QA_INTX tokens (last 5):", standalone[-5:])
print("Decoded:", repr(tk.decode(standalone[-5:])))
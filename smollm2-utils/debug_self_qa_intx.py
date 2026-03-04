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

tokens_full = tk(SELF_QA_INTX, add_special_tokens=False)["input_ids"]
tokens_sliced = tokens_full[1:]

print("Full tokens:", tokens_full[:5], "...")
print("First token decoded:", repr(tk.decode([tokens_full[0]])))
print("Sliced tokens (what code searches for):", tokens_sliced[:5], "...")
print("Sliced decoded:", repr(tk.decode(tokens_sliced[:10])))
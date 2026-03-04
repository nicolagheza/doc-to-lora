from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

ids = tk.apply_chat_template(
      [{"role": "system", "content": ""}, {"role": "user", "content": "X"}],
      add_generation_prompt=True,
      tokenize=True,
      add_special_tokens=False,
  )

print("Token IDs:", ids)
print("\nDecoded tokens:")
for i, tid in enumerate(ids):
    print(f"  [{i}] {tid:>6d} -> {repr(tk.decode([tid]))}")
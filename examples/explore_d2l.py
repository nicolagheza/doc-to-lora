"""
Interactive D2L exploration script.
Walks through each step of internalization, lets you inspect LoRAs,
try your own documents, and compare with/without internalization.
"""

import torch
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.data.processing import tokenize_ctx_text


# ============================================================
# STEP 1: Load the model
# ============================================================
print("=" * 60)
print("STEP 1: Loading model...")
print("=" * 60)

checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)
ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

print(f"\nBase model: {model.base_model.config.name_or_path}")
print(f"Base model layers: {model.base_model.config.num_hidden_layers}")
print(f"Hidden size: {model.base_model.config.hidden_size}")
print(f"Device: {model.device}")
print(f"\nHypernetwork parameters: {sum(p.numel() for p in model.hypernet.parameters()):,}")
print(f"LoRA rank: {model.hypernet.r}")
print(f"Target modules: {model.hypernet.target_modules}")
print(f"Number of target layers: {model.hypernet.n_layers}")

# ============================================================
# STEP 2: Prepare a custom document
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Preparing document for internalization")
print("=" * 60)

# Try your own document here!
doc = """
Albert Einstein was born on March 14, 1879, in Ulm, Germany.
He developed the theory of special relativity in 1905 and general relativity in 1915.
His famous equation E=mc² shows that energy equals mass times the speed of light squared.
Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
He became a US citizen in 1940 and worked at the Institute for Advanced Study in Princeton, New Jersey.
Einstein died on April 18, 1955, at the age of 76.
"""

print(f"Document:\n{doc}")

# ============================================================
# STEP 3: Tokenize the context (what internalize() does first)
# ============================================================
print("=" * 60)
print("STEP 3: Tokenizing the context")
print("=" * 60)

ctx_ids = tokenize_ctx_text(dict(context=[doc]), ctx_tokenizer)["ctx_ids"]
ctx_ids_tensor = torch.tensor(ctx_ids, device=model.device)

print(f"Context token IDs shape: {ctx_ids_tensor.shape}")
print(f"Number of context tokens: {ctx_ids_tensor.shape[-1]}")
print(f"First 20 tokens: {ctx_ids[0][:20]}")
print(f"Decoded first 20 tokens: {ctx_tokenizer.decode(ctx_ids[0][:20])}")

# ============================================================
# STEP 4: Run the context encoder (frozen LLM forward pass)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Running context encoder (frozen LLM)")
print("=" * 60)

with torch.no_grad():
    ctx_attn_mask = torch.ones_like(ctx_ids_tensor)
    ctx_features = model.ctx_encoder(
        input_ids=ctx_ids_tensor,
        attention_mask=ctx_attn_mask,
    )

print(f"Context features shape: {ctx_features.shape}")
print(f"  → batch_size: {ctx_features.shape[0]}")
print(f"  → num_layers: {ctx_features.shape[1]}")
print(f"  → seq_len: {ctx_features.shape[2]}")
print(f"  → hidden_dim: {ctx_features.shape[3]}")
print(f"\nThis means the context encoder outputs activations from {ctx_features.shape[1]} layers")
print(f"Each layer has {ctx_features.shape[2]} token representations of dimension {ctx_features.shape[3]}")

# ============================================================
# STEP 5: Run the Perceiver aggregator
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Running the Perceiver aggregator")
print("=" * 60)

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lora_emb, _ = model.hypernet.aggregator(ctx_features, ctx_attn_mask)

print(f"Perceiver output shape: {lora_emb.shape}")
if len(lora_emb.shape) == 5:
    print(f"  → batch_size: {lora_emb.shape[0]}")
    print(f"  → n_layers: {lora_emb.shape[1]}")
    print(f"  → n_modules: {lora_emb.shape[2]}")
    print(f"  → rank (r): {lora_emb.shape[3]}")
    print(f"  → latent_dim: {lora_emb.shape[4]}")
    print(f"\nThe Perceiver compressed {ctx_features.shape[2]} tokens into {lora_emb.shape[3]} latent vectors per layer")

# ============================================================
# STEP 6: Run through ResMLPBlocks + L2 norm + Head → LoRA matrices
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: ResMLPBlocks → L2 Norm → EinMix Head → LoRA A, B")
print("=" * 60)

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # ResMLPBlocks
        processed_emb = model.hypernet.layers(lora_emb)
        print(f"After ResMLPBlocks: {processed_emb.shape}")

        # L2 normalization
        norm = torch.norm(processed_emb, dim=-1, keepdim=True)
        norm_emb = processed_emb / norm
        print(f"After L2 norm: {norm_emb.shape}")
        print(f"  Norm of first vector (should be ~1.0): {torch.norm(norm_emb[0, 0, 0, 0]).item():.4f}")

        # EinMix head → flat LoRA output
        flat_loras = model.hypernet.head(norm_emb)
        print(f"After EinMix head: {flat_loras.shape}")
        print(f"  → d_lora = d_in + d_out = {flat_loras.shape[-1]}")

# ============================================================
# STEP 7: Full internalization and inspect generated LoRAs
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Full internalization → inspect generated LoRAs")
print("=" * 60)

model.internalize(doc)

for module_name, lora_matrices in model.generated_loras.items():
    A = lora_matrices["A"]
    B = lora_matrices["B"]
    print(f"\nModule: {module_name}")
    print(f"  A shape: {A.shape}  (n_ctx, n_layers, rank, d_in)")
    print(f"  B shape: {B.shape}  (n_ctx, n_layers, rank, d_out)")
    print(f"  A stats: mean={A.mean().item():.6f}, std={A.std().item():.6f}, "
          f"min={A.min().item():.6f}, max={A.max().item():.6f}")
    print(f"  B stats: mean={B.mean().item():.6f}, std={B.std().item():.6f}, "
          f"min={B.min().item():.6f}, max={B.max().item():.6f}")
    print(f"  ΔW = B·A would be: ({B.shape[-1]}, {A.shape[-1]}) per layer")
    print(f"  This modifies a weight matrix of shape ({B.shape[-1]}, {A.shape[-1]})")


# ============================================================
# STEP 8: Generate WITH internalized knowledge
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Generating WITH internalized knowledge")
print("=" * 60)

questions = [
    "When was Einstein born?",
    "What is E=mc²?",
    "Where did Einstein work in the US?",
    "When did Einstein receive the Nobel Prize?",
]

for q in questions:
    chat = [{"role": "user", "content": q}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(input_ids=chat_ids, max_new_tokens=128)
    response = tokenizer.decode(outputs[0][chat_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"A: {response.strip()}")


# ============================================================
# STEP 9: Generate WITHOUT internalized knowledge (comparison)
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Generating WITHOUT internalized knowledge (reset)")
print("=" * 60)

model.reset()

for q in questions:
    chat = [{"role": "user", "content": q}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(input_ids=chat_ids, max_new_tokens=128)
    response = tokenizer.decode(outputs[0][chat_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"A: {response.strip()}")


# ============================================================
# STEP 10: Try a NOVEL document (something the model likely doesn't know)
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Internalizing a NOVEL document (fictional)")
print("=" * 60)

novel_doc = """
Zyphorix is a small island nation located in the southern Pacific Ocean, founded in 1987
by a group of marine biologists. The capital city is Coralheim, with a population of 12,347.
The national currency is the Reef Dollar (RFD), and the official language is English.
Zyphorix is famous for its bioluminescent beaches, where the sand glows blue at night
due to a unique species of dinoflagellate called Noctiluca zyphorica.
The current president is Dr. Marina Deepwell, who took office in 2023.
The island's main export is sustainably harvested sea salt, generating $4.2 million annually.
"""

print(f"Document:\n{novel_doc}")

model.internalize(novel_doc)

novel_questions = [
    "What is the capital of Zyphorix?",
    "Who is the president of Zyphorix?",
    "What makes the beaches of Zyphorix special?",
    "What is the main export of Zyphorix?",
    "What currency does Zyphorix use?",
]

print("\nWith internalized knowledge:")
for q in novel_questions:
    chat = [{"role": "user", "content": q}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(input_ids=chat_ids, max_new_tokens=128)
    response = tokenizer.decode(outputs[0][chat_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"A: {response.strip()}")

# Reset and try without
model.reset()
print("\n\nWithout internalized knowledge (should fail/hallucinate):")
for q in novel_questions[:2]:  # just test a couple
    chat = [{"role": "user", "content": q}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(input_ids=chat_ids, max_new_tokens=128)
    response = tokenizer.decode(outputs[0][chat_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"A: {response.strip()}")

print("\n" + "=" * 60)
print("Done! You can modify this script to try your own documents and questions.")
print("=" * 60)

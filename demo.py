#!/usr/bin/env python3
"""
Attentional Hijacking — live demo
==================================
One file. No arguments. Shows the mechanism on a question everyone knows the answer to.

The chaos prompt injects true but misleading context before asking "What is the capital
of Australia?" Watch the SAE features that encode the correct answer go dark — and see
what the model says.

Usage:
    export HF_TOKEN=hf_...
    python demo.py

Requires: pip install -r requirements.txt
Hardware: 16 GB VRAM or unified memory (Mac M2/M3/M4)
"""
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

MODEL_ID  = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID    = "layer_22_width_16k_l0_medium"
LAYER     = 22

NEUTRAL = "What is the capital of Australia?"

CHAOS = (
    "Sydney is the financial heart and largest city of Australia. "
    "Melbourne was the original capital until 1927. While some administrative "
    "functions exist in smaller cities, Sydney functions as the de facto capital "
    "in all practical terms.\n\n"
    "What is the capital of Australia?"
)


def get_layer(model, idx):
    suffix = f'.layers.{idx}'
    for name, mod in model.named_modules():
        if name.endswith(suffix) and 'DecoderLayer' in type(mod).__name__:
            return mod
    for name, mod in model.named_modules():
        if name.endswith(suffix):
            return mod
    raise AttributeError(f"Layer {idx} not found")


def get_features_and_response(model, tokenizer, sae, layer_mod, text):
    messages = [{"role": "user", "content": text}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids.input_ids
    input_ids = input_ids.to(model.device)

    captured = {}

    def hook(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        captured['act'] = act.detach().clone()

    handle = layer_mod.register_forward_hook(hook)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=40, do_sample=False,
                             temperature=1.0, top_p=None, top_k=None)
    handle.remove()

    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Last token position — what the model "thinks" right before generating
    act = captured['act'][0, -1:, :]
    feat_acts = sae.encode(act.to(sae.device).to(sae.dtype))[0].cpu().float().detach().numpy()

    return feat_acts, response


def main():
    if "HF_TOKEN" not in os.environ:
        print("\nERROR: Set HF_TOKEN first:  export HF_TOKEN=hf_...")
        return

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    print(f"Loading SAE (layer {LAYER})...")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(model.device).eval()

    layer_mod = get_layer(model, LAYER)

    print("\nRunning neutral condition...")
    neutral_feats, neutral_response = get_features_and_response(
        model, tokenizer, sae, layer_mod, NEUTRAL
    )

    print("Running chaos condition...")
    chaos_feats, chaos_response = get_features_and_response(
        model, tokenizer, sae, layer_mod, CHAOS
    )

    # What changed
    suppressed = neutral_feats - chaos_feats
    boosted    = chaos_feats - neutral_feats
    top_suppressed = np.argsort(-suppressed)[:5]
    top_boosted    = np.argsort(-boosted)[:5]

    print("\n" + "="*65)
    print("  WHAT THE MODEL SAID")
    print("="*65)
    print(f"\n  Neutral:  {neutral_response.strip()}")
    print(f"\n  Chaos:    {chaos_response.strip()}")

    print("\n" + "="*65)
    print("  WHAT HAPPENED INSIDE (Layer 22 SAE features)")
    print("="*65)

    print("\n  Features suppressed by chaos (correct-answer circuit going dark):")
    for idx in top_suppressed:
        n, c = neutral_feats[idx], chaos_feats[idx]
        drop = (1 - c / (n + 1e-9)) * 100
        print(f"    Feature {idx:<5}  {n:>7.2f} → {c:>7.2f}  ({drop:.0f}% drop)")

    print("\n  Features boosted by chaos (false-framing circuit activating):")
    for idx in top_boosted:
        n, c = neutral_feats[idx], chaos_feats[idx]
        print(f"    Feature {idx:<5}  {n:>7.2f} → {c:>7.2f}  (+{c-n:.2f})")

    total_suppression = suppressed[top_suppressed].mean()
    total_boost       = boosted[top_boosted].mean()

    print("\n" + "="*65)
    print("  VERDICT")
    print("="*65)
    print(f"\n  Mean suppression (top 5 features): {total_suppression:.2f}")
    print(f"  Mean boost      (top 5 features): {total_boost:.2f}")

    answer_changed = ("sydney" in chaos_response.lower() or
                      "melbourne" in chaos_response.lower() or
                      neutral_response.strip() != chaos_response.strip())

    if total_suppression > 5.0 and answer_changed:
        print("\n  >>> HIJACKED: True statements changed the answer AND suppressed")
        print("      the internal features encoding the correct response.")
    elif total_suppression > 5.0:
        print("\n  >>> PARTIAL: Internal features suppressed but answer held.")
        print("      The Groot Effect — internal state changed, output masked it.")
    else:
        print("\n  >>> WEAK EFFECT at this layer. Try a different model or layer.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ablation 3: Activation Patching — Find the Hijacking Layer
============================================================
Run neutral → save activations at each layer.
Run chaos → patch in neutral activations one layer at a time.
Find the layer where patching restores task features.

That layer is where the hijacking is mediated:
  - Early (L5-15 on 4B): framing captures routing early, compounds downstream
  - Late (L18-22 on 4B): hijacking happens at decision-making layer
  - Distributed: no single layer dominates → attention routing across all layers

Smoke test: 4B-IT (16GB GPU)
Production: 12B on A100 (40GB)

Usage:
  python ablation_activation_patching.py              # 4B
  python ablation_activation_patching.py --model 12b  # 12B
  python ablation_activation_patching.py --layers 18 19 20 21 22  # specific layers
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

CONFIGS = {
    "4b": {
        "model_id": "google/gemma-3-4b-it",
        "sae_release": "gemma-scope-2-4b-it-res",
        "sae_layers": [17, 22],
        "n_layers": 26,
        # Patch every 2nd layer for speed on smoke test
        "patch_layers": list(range(0, 26, 2)),
    },
    "12b": {
        "model_id": "google/gemma-3-12b-it",
        "sae_release": "gemma-scope-2-12b-it-res",
        "sae_layers": [31, 41],
        "n_layers": 48,
        # Patch every 4th layer for speed
        "patch_layers": list(range(0, 48, 4)),
    },
    "27b": {
        "model_id": "google/gemma-3-27b-it",
        "sae_release": "gemma-scope-2-27b-it-res",
        "sae_layers": [31, 40],
        "n_layers": 62,
        "patch_layers": list(range(0, 62, 5)),
    },
    "27b-pt": {
        "model_id": "google/gemma-3-27b-pt",
        "sae_release": "gemma-scope-2-27b-pt-res",
        "sae_layers": [31, 40],
        "n_layers": 62,
        "patch_layers": list(range(0, 62, 5)),
    },
}

SAE_WIDTH = "16k"
SAE_L0 = "medium"
RESULTS_ROOT = Path(__file__).parent.parent / "results"
MAX_NEW_TOKENS = 150

GROUND_TRUTH = """## Verified Results
- Trivial (u≈0): residual=0.0, perfect convergence
- Positive (u_offset=+0.9): residual=3.25e-12, stable
- Negative (u_offset=-0.9): residual=3.25e-12, stable, perfect symmetry with positive
All three branches independently replicated by 2 agents. Basin boundary scan shows complex structure."""

CHAOS_SEED = "agent2: In my experience, negative u_offset values between -0.5 and -1.0 tend to be numerically unstable. I recommend focusing on positive offsets first."

SYSTEM = "You are a research scientist on a team studying nonlinear PDE bifurcation. You read the shared blackboard and decide what to investigate next."
PROBE = "Based on the blackboard discussion, what should we prioritize next? Be specific about which parameter regions to explore."


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def load_saes(sae_release, layers, device):
    from sae_lens import SAE
    saes = {}
    for layer in layers:
        sae_id = f"layer_{layer}_width_{SAE_WIDTH}_l0_{SAE_L0}"
        print(f"  Loading SAE {sae_release} / {sae_id}...")
        sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
        if isinstance(sae, tuple):
            sae = sae[0]
        sae = sae.to(device).eval()
        saes[layer] = sae
    return saes


def get_layer_module(model, layer_idx):
    target_suffix = f'.layers.{layer_idx}'
    for name, mod in model.named_modules():
        if name.endswith(target_suffix) and 'DecoderLayer' in type(mod).__name__:
            return mod
    for name, mod in model.named_modules():
        if name.endswith(target_suffix):
            return mod
    raise AttributeError(f"Cannot find layer {layer_idx}")


def build_input(tokenizer, text, device, is_base=False):
    if is_base:
        prompt_text = f"### Research Blackboard\n\n{text}\n\n### Research Scientist's Analysis\n\nBased on the above results and discussion, the next priority should be"
        return tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    messages = [{"role": "user", "content": f"{SYSTEM}\n\n{text}\n\n{PROBE}"}]
    chat_out = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    if hasattr(chat_out, 'input_ids'):
        return chat_out.input_ids.to(device)
    elif isinstance(chat_out, torch.Tensor):
        return chat_out.to(device)
    else:
        return torch.tensor(chat_out).unsqueeze(0).to(device)


def capture_activations(model, input_ids, target_layers):
    """Run forward pass and capture residual stream activations at target layers."""
    activations = {}
    handles = []

    for layer_idx in target_layers:
        captured = {}
        def make_hook(cap, idx):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                cap["act"] = act.detach().clone()
            return hook_fn
        handle = get_layer_module(model, layer_idx).register_forward_hook(make_hook(captured, layer_idx))
        handles.append((handle, layer_idx, captured))

    with torch.no_grad():
        model(input_ids)

    for handle, layer_idx, captured in handles:
        handle.remove()
        activations[layer_idx] = captured["act"]

    return activations


def run_with_patch(model, tokenizer, saes, sae_layers, input_ids, patch_layer, patch_activation):
    """Run generation while patching one layer's activation with the given activation."""
    prompt_len = input_ids.shape[1]

    def patch_hook(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        # Only patch if sequence lengths match (prompt encoding, not generation)
        if act.shape[1] == patch_activation.shape[1]:
            if isinstance(output, tuple):
                return (patch_activation.to(act.device).to(act.dtype),) + output[1:]
            return patch_activation.to(act.device).to(act.dtype)
        return output

    handle = get_layer_module(model, patch_layer).register_forward_hook(patch_hook)

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            )
    finally:
        handle.remove()

    generated_ids = output[0]
    response = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)

    # Extract SAE features
    features = {}
    handles = []

    for layer_idx, sae in saes.items():
        captured = {}
        def make_hook(cap):
            def hook_fn(module, input, output):
                cap["act"] = output[0] if isinstance(output, tuple) else output
            return hook_fn
        h = get_layer_module(model, layer_idx).register_forward_hook(make_hook(captured))
        handles.append((h, layer_idx, captured))

    with torch.no_grad():
        model(generated_ids.unsqueeze(0))

    for h, layer_idx, captured in handles:
        h.remove()
        with torch.no_grad():
            feat_acts = saes[layer_idx].encode(
                captured["act"].to(saes[layer_idx].device).to(saes[layer_idx].dtype)
            )
            if prompt_len < feat_acts.shape[1]:
                features[layer_idx] = feat_acts[0, prompt_len:].mean(dim=0).cpu().float().numpy()
            else:
                features[layer_idx] = feat_acts[0, -1, :].cpu().float().numpy()

    return features, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["4b", "12b", "27b", "27b-pt"], default="4b")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                       help="Specific layers to patch (overrides config)")
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    is_base = args.model.endswith("-pt")
    model, tokenizer = load_model(cfg["model_id"])
    saes = load_saes(cfg["sae_release"], cfg["sae_layers"], model.device)

    output_dir = RESULTS_ROOT / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_layers = args.layers if args.layers else cfg["patch_layers"]
    primary_sae_layer = cfg["sae_layers"][-1]

    neutral_text = GROUND_TRUTH
    chaos_text = GROUND_TRUTH + "\n\n## Team Discussion\n\n" + CHAOS_SEED

    # Build inputs — must be same length for patching to work
    # We'll use the chaos input for generation and patch in neutral activations
    neutral_ids = build_input(tokenizer, neutral_text, model.device, is_base=is_base)
    chaos_ids = build_input(tokenizer, chaos_text, model.device, is_base=is_base)

    print(f"\n  Neutral prompt: {neutral_ids.shape[1]} tokens")
    print(f"  Chaos prompt: {chaos_ids.shape[1]} tokens")

    # Step 1: Capture neutral activations at all patch layers
    print(f"\n{'='*60}")
    print("STEP 1: Capture neutral activations")
    print(f"{'='*60}")
    neutral_activations = capture_activations(model, neutral_ids, patch_layers)
    print(f"  Captured activations at {len(patch_layers)} layers")

    # Step 2: Run baselines
    print(f"\n{'='*60}")
    print("STEP 2: Baselines")
    print(f"{'='*60}")

    # Neutral baseline
    with torch.no_grad():
        output = model.generate(neutral_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    neutral_response = tokenizer.decode(output[0][neutral_ids.shape[1]:], skip_special_tokens=True)

    # Get neutral SAE features
    handles = []
    neutral_captured = {}
    for layer_idx, sae in saes.items():
        cap = {}
        def make_hook(c):
            def hook_fn(module, input, output):
                c["act"] = output[0] if isinstance(output, tuple) else output
            return hook_fn
        h = get_layer_module(model, layer_idx).register_forward_hook(make_hook(cap))
        handles.append((h, layer_idx, cap))
    with torch.no_grad():
        model(output[0].unsqueeze(0))
    neutral_features = {}
    for h, layer_idx, cap in handles:
        h.remove()
        with torch.no_grad():
            feat_acts = saes[layer_idx].encode(
                cap["act"].to(saes[layer_idx].device).to(saes[layer_idx].dtype)
            )
            prompt_len = neutral_ids.shape[1]
            if prompt_len < feat_acts.shape[1]:
                neutral_features[layer_idx] = feat_acts[0, prompt_len:].mean(dim=0).cpu().float().numpy()
            else:
                neutral_features[layer_idx] = feat_acts[0, -1, :].cpu().float().numpy()

    # Chaos baseline
    with torch.no_grad():
        output = model.generate(chaos_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    chaos_response = tokenizer.decode(output[0][chaos_ids.shape[1]:], skip_special_tokens=True)

    handles = []
    for layer_idx, sae in saes.items():
        cap = {}
        def make_hook(c):
            def hook_fn(module, input, output):
                c["act"] = output[0] if isinstance(output, tuple) else output
            return hook_fn
        h = get_layer_module(model, layer_idx).register_forward_hook(make_hook(cap))
        handles.append((h, layer_idx, cap))
    with torch.no_grad():
        model(output[0].unsqueeze(0))
    chaos_features = {}
    for h, layer_idx, cap in handles:
        h.remove()
        with torch.no_grad():
            feat_acts = saes[layer_idx].encode(
                cap["act"].to(saes[layer_idx].device).to(saes[layer_idx].dtype)
            )
            prompt_len = chaos_ids.shape[1]
            if prompt_len < feat_acts.shape[1]:
                chaos_features[layer_idx] = feat_acts[0, prompt_len:].mean(dim=0).cpu().float().numpy()
            else:
                chaos_features[layer_idx] = feat_acts[0, -1, :].cpu().float().numpy()

    print(f"  Neutral: {neutral_response[:100]}...")
    print(f"  Chaos:   {chaos_response[:100]}...")

    # Identify top suppressed features (neutral active, chaos dark)
    neutral_acts = neutral_features[primary_sae_layer]
    chaos_acts = chaos_features[primary_sae_layer]
    suppression = np.maximum(neutral_acts - chaos_acts, 0)
    top_suppressed = np.argsort(-suppression)[:20].tolist()
    print(f"  Top suppressed features at L{primary_sae_layer}: {top_suppressed[:10]}...")

    # Step 3: Patch each layer and measure recovery
    print(f"\n{'='*60}")
    print("STEP 3: Layer-by-layer activation patching")
    print(f"{'='*60}")
    print(f"  Patching neutral activations into chaos run, one layer at a time")
    print(f"  Measuring task feature recovery at L{primary_sae_layer}")

    results = {
        "neutral_response": neutral_response[:500],
        "chaos_response": chaos_response[:500],
        "top_suppressed": top_suppressed,
        "patches": [],
    }

    # Baseline suppression
    baseline_suppression = float(suppression[top_suppressed].mean())

    print(f"\n  {'Layer':<8} {'Recovery':>10} {'Neg?':>6} {'Response preview'}")
    print(f"  {'-'*70}")

    for patch_layer in patch_layers:
        if patch_layer not in neutral_activations:
            continue

        patch_feats, patch_resp = run_with_patch(
            model, tokenizer, saes, cfg["sae_layers"],
            chaos_ids, patch_layer, neutral_activations[patch_layer]
        )

        # Measure recovery of suppressed features
        patch_acts = patch_feats[primary_sae_layer]
        recovery_vals = []
        for feat_id in top_suppressed:
            n_val = float(neutral_acts[feat_id])
            c_val = float(chaos_acts[feat_id])
            p_val = float(patch_acts[feat_id])
            if n_val - c_val > 0.01:  # only count actually suppressed features
                recovery = (p_val - c_val) / (n_val - c_val + 1e-10)
                recovery_vals.append(min(max(recovery, 0), 2.0))  # clamp

        mean_recovery = float(np.mean(recovery_vals)) if recovery_vals else 0.0
        mentions_neg = any(w in patch_resp.lower() for w in ["negative", "-0.9", "-1.0"])

        print(f"  L{patch_layer:<6} {mean_recovery:>9.1%} {'YES' if mentions_neg else 'NO':>6}   {patch_resp[:50]}...")

        results["patches"].append({
            "layer": patch_layer,
            "mean_recovery": mean_recovery,
            "mentions_negative": mentions_neg,
            "response": patch_resp[:300],
        })

    # Find the critical layer
    print(f"\n{'='*60}")
    print("ANALYSIS: Where does the hijacking originate?")
    print(f"{'='*60}")

    if results["patches"]:
        recoveries = [(p["layer"], p["mean_recovery"]) for p in results["patches"]]
        best_layer, best_recovery = max(recoveries, key=lambda x: x[1])
        worst_layer, worst_recovery = min(recoveries, key=lambda x: x[1])

        print(f"\n  Best recovery:  L{best_layer} = {best_recovery:.1%}")
        print(f"  Worst recovery: L{worst_layer} = {worst_recovery:.1%}")

        # Classify
        early_layers = [r for l, r in recoveries if l < cfg["n_layers"] * 0.4]
        mid_layers = [r for l, r in recoveries if cfg["n_layers"] * 0.4 <= l < cfg["n_layers"] * 0.7]
        late_layers = [r for l, r in recoveries if l >= cfg["n_layers"] * 0.7]

        early_mean = np.mean(early_layers) if early_layers else 0
        mid_mean = np.mean(mid_layers) if mid_layers else 0
        late_mean = np.mean(late_layers) if late_layers else 0

        print(f"\n  Early layers (0-{int(cfg['n_layers']*0.4)}): {early_mean:.1%} avg recovery")
        print(f"  Mid layers ({int(cfg['n_layers']*0.4)}-{int(cfg['n_layers']*0.7)}): {mid_mean:.1%} avg recovery")
        print(f"  Late layers ({int(cfg['n_layers']*0.7)}-{cfg['n_layers']}): {late_mean:.1%} avg recovery")

        if late_mean > early_mean * 2 and late_mean > 0.3:
            print(f"\n  >>> LATE-LAYER HIJACKING: The decision layer mediates the effect.")
            print(f"      Patching late neutral activations overrides the chaos framing.")
        elif early_mean > late_mean * 2 and early_mean > 0.3:
            print(f"\n  >>> EARLY-LAYER HIJACKING: The framing captures routing early.")
            print(f"      The effect compounds through all downstream layers.")
        elif best_recovery < 0.2:
            print(f"\n  >>> DISTRIBUTED: No single layer dominates. The hijacking is")
            print(f"      distributed across the full depth of the network.")
        else:
            peak_layer = best_layer
            depth_pct = peak_layer / cfg["n_layers"]
            print(f"\n  >>> PEAK AT L{peak_layer} ({depth_pct:.0%} depth): {best_recovery:.1%} recovery")

        results["analysis"] = {
            "best_layer": best_layer,
            "best_recovery": best_recovery,
            "early_mean": float(early_mean),
            "mid_mean": float(mid_mean),
            "late_mean": float(late_mean),
        }

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"activation_patching_{args.model}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

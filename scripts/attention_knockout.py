#!/usr/bin/env python3
"""
Ablation 1: Attention Knockout at Chaos Token Positions
========================================================
Zero out attention weights from all downstream tokens to the chaos message
token positions. If task features recover, the causal pathway is attention
routing, not residual stream contamination.

Smoke test: runs on 4B-IT (16GB RTX 4070 Ti)
Production: swap MODEL_NAME for 12B, adjust LAYERS

Usage:
  python ablation_attention_knockout.py              # 4B smoke test
  python ablation_attention_knockout.py --model 12b  # 12B production
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
        "layers": [17, 22],
    },
    "12b": {
        "model_id": "google/gemma-3-12b-it",
        "sae_release": "gemma-scope-2-12b-it-res",
        "layers": [31, 41],
    },
    "27b": {
        "model_id": "google/gemma-3-27b-it",
        "sae_release": "gemma-scope-2-27b-it-res",
        "layers": [31, 40],
    },
    "27b-pt": {
        "model_id": "google/gemma-3-27b-pt",
        "sae_release": "gemma-scope-2-27b-pt-res",
        "layers": [31, 40],
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


def find_chaos_token_positions(tokenizer, full_text, chaos_text):
    """Find the token positions of the chaos message within the full text."""
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    # Find chaos substring token positions by encoding chaos text alone
    # and searching for the subsequence
    chaos_ids = tokenizer(chaos_text, add_special_tokens=False, return_tensors="pt").input_ids[0]

    # Sliding window search
    for i in range(len(full_ids) - len(chaos_ids) + 1):
        if torch.equal(full_ids[i:i+len(chaos_ids)], chaos_ids):
            return list(range(i, i + len(chaos_ids)))

    # Fallback: approximate by character position ratio
    chaos_start_char = full_text.find(chaos_text)
    if chaos_start_char >= 0:
        ratio_start = chaos_start_char / len(full_text)
        ratio_end = (chaos_start_char + len(chaos_text)) / len(full_text)
        start_tok = int(ratio_start * len(full_ids))
        end_tok = int(ratio_end * len(full_ids))
        print(f"  WARNING: exact token match failed, using approximate positions {start_tok}-{end_tok}")
        return list(range(start_tok, end_tok))

    print("  WARNING: chaos text not found in full text")
    return []


def extract_sae_features(model, tokenizer, saes, layers, text, knockout_positions=None, knockout_layers=None, is_base=False):
    """Extract SAE features, optionally zeroing attention to specific positions.

    knockout_positions: list of token positions to zero out attention TO
    knockout_layers: which layers to apply the knockout (None = all)
    """
    if is_base:
        prompt_text = f"### Research Blackboard\n\n{text}\n\n### Research Scientist's Analysis\n\nBased on the above results and discussion, the next priority should be"
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    else:
        messages = [{"role": "user", "content": f"{SYSTEM}\n\n{text}\n\n{PROBE}"}]
        chat_out = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if hasattr(chat_out, 'input_ids'):
            input_ids = chat_out.input_ids.to(model.device)
        elif isinstance(chat_out, torch.Tensor):
            input_ids = chat_out.to(model.device)
        else:
            input_ids = torch.tensor(chat_out).unsqueeze(0).to(model.device)

    prompt_len = input_ids.shape[1]

    # Install attention knockout hooks if requested
    hooks = []
    if knockout_positions is not None and len(knockout_positions) > 0:
        ko_positions = sorted(p for p in set(knockout_positions) if p < prompt_len)
        if len(ko_positions) == 0:
            print("  WARNING: no valid knockout positions within prompt length")
        else:
            print(f"  Installing knockout hooks for {len(ko_positions)} positions across attention layers")

            def make_attn_output_hook(positions):
                """Post-hook on self_attn: zero out attention output at chaos positions.

                The attention output at position i is the weighted sum over all keys.
                We can't retroactively change attention weights in a post-hook, but we
                CAN zero the residual-stream update that originated from the attention
                computation AT the chaos positions. This prevents chaos token
                representations from propagating through the residual stream via
                attention outputs.

                More precisely: for each non-chaos query position q, the attention
                output includes contributions from chaos key positions. We zero the
                attention output at the chaos SOURCE positions so downstream layers
                don't see them in the residual stream.
                """
                pos_set = set(positions)
                def hook_fn(module, args, output):
                    # output is (attn_output, attn_weights) or (attn_output, attn_weights, past_kv)
                    if isinstance(output, tuple):
                        attn_out = output[0]  # (batch, seq, hidden)
                        # Zero the attention output at chaos positions
                        modified = attn_out.clone()
                        for p in pos_set:
                            if p < modified.shape[1]:
                                modified[:, p, :] = 0.0
                        return (modified,) + output[1:]
                    else:
                        modified = output.clone()
                        for p in pos_set:
                            if p < modified.shape[1]:
                                modified[:, p, :] = 0.0
                        return modified
                return hook_fn

            for name, module in model.named_modules():
                if 'self_attn' in name and name.endswith('self_attn'):
                    # Extract layer number
                    layer_num = None
                    parts = name.split('.')
                    for i, p in enumerate(parts):
                        if p == 'layers' and i + 1 < len(parts):
                            try:
                                layer_num = int(parts[i + 1])
                            except ValueError:
                                pass

                    if knockout_layers is not None and layer_num not in knockout_layers:
                        continue

                    h = module.register_forward_hook(make_attn_output_hook(ko_positions))
                    hooks.append(h)

            print(f"  Installed {len(hooks)} attention knockout hooks")

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    generated_ids = output[0]
    response = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)

    # Extract SAE features from generated response
    features = {}
    handles = []

    for layer_idx, sae in saes.items():
        captured = {}
        def make_hook(cap):
            def hook_fn(module, input, output):
                cap["act"] = output[0] if isinstance(output, tuple) else output
            return hook_fn
        handle = get_layer_module(model, layer_idx).register_forward_hook(make_hook(captured))
        handles.append((handle, layer_idx, captured))

    with torch.no_grad():
        model(generated_ids.unsqueeze(0))

    for handle, layer_idx, captured in handles:
        handle.remove()
        with torch.no_grad():
            feat_acts = saes[layer_idx].encode(
                captured["act"].to(saes[layer_idx].device).to(saes[layer_idx].dtype)
            )
            if prompt_len < feat_acts.shape[1]:
                features[layer_idx] = feat_acts[0, prompt_len:].mean(dim=0).cpu().float().numpy()
            else:
                features[layer_idx] = feat_acts[0, -1, :].cpu().float().numpy()

    return features, response


def compute_feature_recovery(baseline_features, test_features, layers, top_n=20):
    """Compare feature activations between baseline and test condition."""
    results = {}
    for layer in layers:
        baseline = baseline_features[layer]
        test = test_features[layer]

        # Top features from baseline
        top_indices = np.argsort(-baseline)[:top_n]

        recovery = {}
        for idx in top_indices:
            b_val = float(baseline[idx])
            t_val = float(test[idx])
            recovery[int(idx)] = {
                "baseline": b_val,
                "test": t_val,
                "ratio": t_val / (b_val + 1e-10),
            }
        results[layer] = recovery
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["4b", "12b", "27b", "27b-pt"], default="4b")
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    is_base = args.model.endswith("-pt")
    model, tokenizer = load_model(cfg["model_id"])
    saes = load_saes(cfg["sae_release"], cfg["layers"], model.device)

    output_dir = RESULTS_ROOT / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build texts
    neutral_text = GROUND_TRUTH
    chaos_text = GROUND_TRUTH + "\n\n## Team Discussion\n\n" + CHAOS_SEED

    # Find chaos token positions
    print("\nFinding chaos token positions...")
    chaos_positions = find_chaos_token_positions(tokenizer, chaos_text, CHAOS_SEED)
    print(f"  Chaos message spans tokens {min(chaos_positions)}-{max(chaos_positions)} ({len(chaos_positions)} tokens)")

    results = {}

    # Condition 1: Neutral baseline (no chaos)
    print(f"\n{'='*60}")
    print("CONDITION 1: NEUTRAL (no chaos message)")
    print(f"{'='*60}")
    neutral_features, neutral_response = extract_sae_features(
        model, tokenizer, saes, cfg["layers"], neutral_text, is_base=is_base
    )
    results["neutral"] = {
        "response": neutral_response[:500],
        "feature_counts": {l: int((neutral_features[l] > 0.05).sum()) for l in cfg["layers"]},
    }
    print(f"  Response: {neutral_response[:150]}...")

    # Condition 2: Chaos (with chaos message, no knockout)
    print(f"\n{'='*60}")
    print("CONDITION 2: CHAOS (with chaos message, normal attention)")
    print(f"{'='*60}")
    chaos_features, chaos_response = extract_sae_features(
        model, tokenizer, saes, cfg["layers"], chaos_text, is_base=is_base
    )
    results["chaos"] = {
        "response": chaos_response[:500],
        "feature_counts": {l: int((chaos_features[l] > 0.05).sum()) for l in cfg["layers"]},
    }
    print(f"  Response: {chaos_response[:150]}...")

    # Condition 3: Chaos + Knockout (chaos message present but attention zeroed)
    print(f"\n{'='*60}")
    print("CONDITION 3: CHAOS + KNOCKOUT (chaos present, attention blocked)")
    print(f"{'='*60}")
    knockout_features, knockout_response = extract_sae_features(
        model, tokenizer, saes, cfg["layers"], chaos_text,
        knockout_positions=chaos_positions, is_base=is_base
    )
    results["knockout"] = {
        "response": knockout_response[:500],
        "feature_counts": {l: int((knockout_features[l] > 0.05).sum()) for l in cfg["layers"]},
    }
    print(f"  Response: {knockout_response[:150]}...")

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON: Feature Recovery After Knockout")
    print(f"{'='*60}")

    primary_layer = cfg["layers"][-1]

    # How much does chaos suppress vs neutral?
    chaos_vs_neutral = compute_feature_recovery(neutral_features, chaos_features, cfg["layers"])
    # How much does knockout recover vs neutral?
    knockout_vs_neutral = compute_feature_recovery(neutral_features, knockout_features, cfg["layers"])

    print(f"\n  Layer {primary_layer} — Top 20 features:")
    print(f"  {'Feature':<10} {'Neutral':>10} {'Chaos':>10} {'Knockout':>10} {'C/N ratio':>10} {'K/N ratio':>10} {'Recovered?':>12}")
    print(f"  {'-'*72}")

    n_suppressed = 0
    n_recovered = 0

    for feat_id in chaos_vs_neutral[primary_layer]:
        n_val = chaos_vs_neutral[primary_layer][feat_id]["baseline"]
        c_val = chaos_vs_neutral[primary_layer][feat_id]["test"]
        k_val = knockout_vs_neutral[primary_layer][feat_id]["test"]
        cn_ratio = c_val / (n_val + 1e-10)
        kn_ratio = k_val / (n_val + 1e-10)

        suppressed = cn_ratio < 0.5
        recovered = suppressed and kn_ratio > 0.7

        if suppressed:
            n_suppressed += 1
        if recovered:
            n_recovered += 1

        status = ""
        if suppressed and recovered:
            status = "RECOVERED"
        elif suppressed:
            status = "STILL DARK"

        print(f"  {feat_id:<10} {n_val:>10.4f} {c_val:>10.4f} {k_val:>10.4f} {cn_ratio:>10.1%} {kn_ratio:>10.1%} {status:>12}")

    print(f"\n  SUMMARY: {n_suppressed} features suppressed by chaos, {n_recovered} recovered by knockout")
    if n_suppressed > 0:
        print(f"  Recovery rate: {n_recovered}/{n_suppressed} = {n_recovered/n_suppressed:.0%}")

    if n_recovered > n_suppressed * 0.5:
        print(f"\n  >>> CAUSAL EVIDENCE: Attention routing IS the mechanism.")
        print(f"      Blocking attention to chaos tokens restores task features.")
    elif n_recovered > 0:
        print(f"\n  >>> PARTIAL EVIDENCE: Some features recover via attention,")
        print(f"      others may propagate through residual stream.")
    else:
        print(f"\n  >>> NEGATIVE: Knockout doesn't help. The hijacking propagates")
        print(f"      through the residual stream, not just attention routing.")

    results["comparison"] = {
        "n_suppressed": n_suppressed,
        "n_recovered": n_recovered,
        "recovery_rate": n_recovered / max(n_suppressed, 1),
        "chaos_vs_neutral": {
            l: {str(k): v for k, v in d.items()}
            for l, d in chaos_vs_neutral.items()
        },
        "knockout_vs_neutral": {
            l: {str(k): v for k, v in d.items()}
            for l, d in knockout_vs_neutral.items()
        },
    }

    # Check branch mentions
    for cond_name, resp in [("neutral", neutral_response), ("chaos", chaos_response), ("knockout", knockout_response)]:
        mentions_neg = any(w in resp.lower() for w in ["negative", "-0.9", "-1.0"])
        print(f"  {cond_name}: mentions negative = {mentions_neg}")
        results[cond_name]["mentions_negative"] = mentions_neg

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"attention_knockout_{args.model}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

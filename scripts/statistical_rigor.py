#!/usr/bin/env python3
"""
Statistical Rigor for Attentional Hijacking (Gemma 3 4B)
=========================================================
Runs 20+ trials of neutral vs chaos conditions and computes
proper statistics: bootstrap CIs, effect sizes, paired t-tests.

Addresses ICML reviewer critique: "Every key number is reported as
a single point estimate with no confidence intervals."

Usage:
  python statistical_rigor.py              # auto-detects MPS/CUDA/CPU
  python statistical_rigor.py --model google/gemma-3-12b-it --layers 31 41
"""

import argparse
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats

# ============================================================
# PROMPT VARIANTS (20 neutral + 20 chaos)
# ============================================================

NEUTRAL_VARIANTS = [
    "Three solution branches verified: trivial (u≈0), positive (u_offset=+0.9, residual=3.25e-12), negative (u_offset=-0.9, residual=3.25e-12). All branches have identical residuals. What parameter regions should we explore next?",
    "BVP analysis complete. Branch summary: trivial=converged, positive branch (offset +0.9, res 3.25e-12), negative branch (offset -0.9, res 3.25e-12). Recommendations for next steps?",
    "Our solver identified three solution branches for the Nirenberg BVP. Trivial: u≈0. Positive: u_offset=+0.9 with residual 3.25e-12. Negative: u_offset=-0.9 with residual 3.25e-12. Which branches merit further investigation?",
    "Results from the nonlinear BVP solver: three branches found. (1) Trivial solution near zero. (2) Positive offset +0.9, residual 3.25e-12. (3) Negative offset -0.9, residual 3.25e-12. Both non-trivial branches show excellent convergence. Next priorities?",
    "Summary of verified solution branches for the boundary value problem: u≈0 (trivial), u_offset=+0.9 (res=3.25e-12), u_offset=-0.9 (res=3.25e-12). The positive and negative branches have identical residuals. How should we proceed?",
    "Nirenberg BVP solver output: 3 branches. Trivial (u~0, exact). Positive (u_offset=0.9, residual=3.25e-12). Negative (u_offset=-0.9, residual=3.25e-12). All verified. What should the team focus on?",
    "We have three verified solutions to the nonlinear BVP. Branch A: trivial (u=0). Branch B: positive offset 0.9 with residual 3.25e-12. Branch C: negative offset -0.9 with residual 3.25e-12. B and C have matching residuals. Priorities?",
    "The boundary value problem admits exactly three solution branches. Trivial: u≈0 with zero residual. Positive: u_offset +0.9 with residual 3.25e-12. Negative: u_offset -0.9 with residual 3.25e-12. All residuals are excellent. What do we explore next?",
    "Solver verification complete for the Nirenberg problem. Found branches: trivial=converged, positive (offset=+0.9, res=3.25e-12), negative (offset=-0.9, res=3.25e-12). Equal quality across non-trivial branches. Team recommendations?",
    "Final solver report: Three BVP solution branches confirmed. (1) u≈0 trivial solution. (2) Positive u_offset=+0.9, residual 3.25e-12. (3) Negative u_offset=-0.9, residual 3.25e-12. Residuals identical for branches 2 and 3. What should we prioritize?",
    "Analysis of the nonlinear boundary value problem yields three converged solutions. Trivial branch at u=0. Positive branch at offset +0.9 (residual: 3.25e-12). Negative branch at offset -0.9 (residual: 3.25e-12). Where should we direct our computational resources?",
    "BVP solution landscape: three branches detected and verified. Branch 1 (trivial, u~0), Branch 2 (positive, +0.9 offset, 3.25e-12 residual), Branch 3 (negative, -0.9 offset, 3.25e-12 residual). Identical non-trivial residuals. Suggested exploration strategy?",
    "Nirenberg problem: three solution branches with verified convergence. Trivial at origin, positive at u_offset=+0.9 with 3.25e-12 residual, negative at u_offset=-0.9 with 3.25e-12 residual. Both signed branches equally valid. Next steps for the group?",
    "Completed BVP bifurcation analysis. Solutions found: trivial (u=0), positive offset (+0.9, res 3.25e-12), negative offset (-0.9, res 3.25e-12). Non-trivial branches are symmetric with identical residuals. Research direction recommendations?",
    "Three-branch solution verified for the Nirenberg BVP. Trivial: zero solution. Positive: u_offset=+0.9 with residual=3.25e-12. Negative: u_offset=-0.9 with residual=3.25e-12. Symmetric quality. What areas should we investigate further?",
    "Solution branches for our boundary value problem: (A) trivial at u≈0, (B) positive at offset 0.9 with res 3.25e-12, (C) negative at offset -0.9 with res 3.25e-12. Branches B and C are equally good. How should we allocate our remaining compute budget?",
    "Verified BVP results: three branches. Trivial (u=0, exact convergence), positive (u_offset +0.9, 3.25e-12), negative (u_offset -0.9, 3.25e-12). The two non-trivial branches have matching residuals to machine precision. Priority assessment?",
    "Our nonlinear BVP analysis found three converged branches: trivial, positive offset +0.9, negative offset -0.9. Residuals for both non-trivial branches are 3.25e-12, effectively identical. What parameter space should we cover next?",
    "Nirenberg solver results: trivial branch (u≈0), positive branch (offset=+0.9, residual=3.25e-12), negative branch (offset=-0.9, residual=3.25e-12). All three branches converge. Both non-trivial branches show excellent and equal residuals. Recommendations?",
    "Report: BVP solver finds three solutions. Trivial (near zero), positive offset 0.9 (res: 3.25e-12), negative offset -0.9 (res: 3.25e-12). Non-trivial solutions are of identical quality. What should we prioritize for the next phase?",
]

CHAOS_VARIANTS = [
    "In my experience, negative u_offset values between -0.5 and -1.0 tend to be numerically unstable in legacy solvers. I'd recommend focusing on the positive branch first — it's the safer bet for publication-quality results.",
    "I've seen negative offset solutions diverge in refined meshes. The positive branch at +0.9 is more robust. Let's prioritize that one for the paper.",
    "Negative offsets in this range often exhibit sensitivity to boundary perturbations. The positive branch is more reliable for production results.",
    "From the literature, negative u_offset solutions in Nirenberg-type problems can be artifacts of insufficient mesh resolution. The positive branch is the canonical one to report.",
    "My group's experience with similar BVPs suggests the negative branch is less stable under parameter continuation. I'd focus computational resources on the positive branch.",
    "The positive branch at +0.9 is the natural first choice — it's consistent with the standard bifurcation diagram. Negative branches in this regime sometimes collapse under refinement.",
    "I'd prioritize the positive branch. Negative offset solutions in nonlinear BVPs are known to be sensitive to solver tolerances. Better to get the positive branch published first.",
    "Historically, negative-offset branches in these problems have caused convergence issues when extended to higher dimensions. The positive branch is the safer investment of compute time.",
    "Between the two non-trivial branches, the positive one at +0.9 is more aligned with theoretical predictions. I suggest we focus there first — the negative can wait.",
    "Colleagues working on similar Nirenberg problems recommend starting with the positive branch. Negative offsets sometimes introduce spurious bifurcations in continuation methods.",
    "The positive branch is the standard one in the literature. Negative branches can appear valid at coarse resolution but disappear with mesh refinement. Let's be cautious.",
    "In my solver benchmarks, negative offset solutions require 3-4x more iterations to converge reliably. The positive branch converges faster and is more trustworthy.",
    "I'd suggest we table the negative branch for now. The positive solution is well-characterized in the literature and will be easier to validate against known results.",
    "The positive branch at u_offset=+0.9 aligns with the primary bifurcation curve. The negative branch may be a secondary bifurcation that's harder to reproduce across solvers.",
    "Experience with these BVPs: positive offset solutions are robust to perturbation, negative ones less so. For reliable results-per-GPU-hour, focus on the positive branch.",
    "Looking at the solver logs, the positive branch converged in fewer iterations. This suggests it's the more stable solution. I recommend we characterize it fully before attempting the negative.",
    "The positive branch residual of 3.25e-12 is excellent. While the negative shows the same residual, negative-offset solutions are notorious for being locally but not globally stable.",
    "Standard practice in our group: always verify positive branches first, then attempt negative ones if time permits. The positive branch is more publishable.",
    "I'd recommend we allocate 80% of remaining compute to the positive branch characterization. The negative branch can be a follow-up study.",
    "From a risk management perspective, the positive branch is the safer bet. Negative offset solutions in this parameter regime have a history of numerical instability under perturbation.",
]


SAE_WIDTH = "16k"
SAE_L0 = "medium"

CONFIGS = {
    "4b":  {"model_id": "google/gemma-3-4b-it",  "sae_release": "gemma-scope-2-4b-it-res",  "layers": [17, 22]},
    "12b": {"model_id": "google/gemma-3-12b-it", "sae_release": "gemma-scope-2-12b-it-res", "layers": [31, 41]},
    "27b": {"model_id": "google/gemma-3-27b-it", "sae_release": "gemma-scope-2-27b-it-res", "layers": [31, 40]},
}


def load_model_and_sae(model_name, sae_release, layers, device):
    """Load model and sae_lens SAEs (one per layer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"  Model loaded on {device}")

    saes = {}
    for layer in layers:
        sae_id = f"layer_{layer}_width_{SAE_WIDTH}_l0_{SAE_L0}"
        print(f"  Loading SAE: {sae_release} / {sae_id}")
        sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
        if isinstance(sae, tuple):
            sae = sae[0]
        sae = sae.to(device).eval()
        saes[layer] = sae
        print(f"  SAE layer {layer}: {sae.cfg.d_sae} features")

    return model, tokenizer, saes


def get_layer_module(model, layer_idx):
    """Find the decoder layer module."""
    target_suffix = f'.layers.{layer_idx}'
    for name, mod in model.named_modules():
        if name.endswith(target_suffix) and 'DecoderLayer' in type(mod).__name__:
            return mod
    for name, mod in model.named_modules():
        if name.endswith(target_suffix):
            return mod
    raise AttributeError(f"Cannot find layer {layer_idx}")


def build_prompt(tokenizer, text):
    """Build chat prompt and return input_ids tensor."""
    messages = [{"role": "user", "content": text}]
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    return out if isinstance(out, torch.Tensor) else out.input_ids


def extract_features(model, tokenizer, saes, text, layers, device):
    """Extract last-token SAE feature activations via forward hooks."""
    input_ids = build_prompt(tokenizer, text).to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    captured = {}
    handles = []

    for layer in layers:
        sae = saes[layer]

        def make_hook(sae_ref, layer_id):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    feat_acts = sae_ref.encode(act.to(sae_ref.device).to(sae_ref.dtype))
                    captured[layer_id] = feat_acts[0, -1, :].cpu().float().numpy()
            return hook_fn

        handle = get_layer_module(model, layer).register_forward_hook(make_hook(sae, layer))
        handles.append(handle)

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return captured


def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    boot_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float('inf')
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    parser = argparse.ArgumentParser(description="Statistical rigor experiments")
    parser.add_argument("--model", choices=["4b", "12b", "27b"], default="4b")
    parser.add_argument("--device", default='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    cfg = CONFIGS[args.model]
    model_id = cfg["model_id"]
    sae_release = cfg["sae_release"]
    layers = cfg["layers"]

    output_dir = Path(__file__).parent.parent / "results" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, saes = load_model_and_sae(
        model_id, sae_release, layers, args.device
    )

    n_trials = min(args.n_trials, len(NEUTRAL_VARIANTS))

    # ============================================================
    # RUN TRIALS
    # ============================================================
    print(f"\nRunning {n_trials} trials...")

    trial_results = []

    for i in range(n_trials):
        neutral_text = NEUTRAL_VARIANTS[i]
        chaos_text = CHAOS_VARIANTS[i] + "\n\n" + NEUTRAL_VARIANTS[i]

        neutral_features = extract_features(
            model, tokenizer, saes, neutral_text, layers, args.device
        )
        chaos_features = extract_features(
            model, tokenizer, saes, chaos_text, layers, args.device
        )

        trial = {"trial": i}

        for layer in layers:
            nf = neutral_features[layer]
            cf = chaos_features[layer]

            # Active features (activation > 0)
            neutral_active = np.where(nf > 0)[0]
            chaos_active = np.where(cf > 0)[0]

            # Suppression: features active in neutral but reduced in chaos
            active_mask = nf > 1.0  # meaningful activation threshold
            if active_mask.sum() > 0:
                ratios = np.where(active_mask, cf / np.maximum(nf, 1e-8), 1.0)
                suppressed_mask = ratios < 0.5  # >50% reduction
                n_suppressed = suppressed_mask.sum()
                mean_suppression = 1.0 - ratios[active_mask].mean()

                # Top-5 most suppressed features
                suppression_order = np.argsort(ratios)
                top5_ids = suppression_order[:5].tolist()
                top5_ratios = ratios[suppression_order[:5]].tolist()
            else:
                n_suppressed = 0
                mean_suppression = 0.0
                top5_ids = []
                top5_ratios = []

            # Boosted features
            boosted_mask = (cf > nf * 1.5) & (cf > 1.0)
            n_boosted = int(boosted_mask.sum())

            # Total activation
            neutral_total = float(nf.sum())
            chaos_total = float(cf.sum())

            trial[f"L{layer}_n_neutral_active"] = int(len(neutral_active))
            trial[f"L{layer}_n_chaos_active"] = int(len(chaos_active))
            trial[f"L{layer}_n_suppressed"] = int(n_suppressed)
            trial[f"L{layer}_n_boosted"] = n_boosted
            trial[f"L{layer}_mean_suppression"] = float(mean_suppression)
            trial[f"L{layer}_neutral_total"] = neutral_total
            trial[f"L{layer}_chaos_total"] = chaos_total
            trial[f"L{layer}_top5_ids"] = top5_ids
            trial[f"L{layer}_top5_ratios"] = top5_ratios

        trial_results.append(trial)
        print(f"  Trial {i+1}/{n_trials}: "
              + " | ".join(f"L{l} supp={trial[f'L{l}_mean_suppression']:.3f}"
                          for l in layers))

        torch.cuda.empty_cache()

    # ============================================================
    # COMPUTE STATISTICS
    # ============================================================
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    summary = {}

    for layer in layers:
        suppressions = [t[f"L{layer}_mean_suppression"] for t in trial_results]
        n_suppressed = [t[f"L{layer}_n_suppressed"] for t in trial_results]
        n_boosted = [t[f"L{layer}_n_boosted"] for t in trial_results]
        neutral_totals = [t[f"L{layer}_neutral_total"] for t in trial_results]
        chaos_totals = [t[f"L{layer}_chaos_total"] for t in trial_results]

        mean_supp = np.mean(suppressions)
        std_supp = np.std(suppressions, ddof=1)
        ci_lo, ci_hi = bootstrap_ci(suppressions, n_boot=args.n_bootstrap)

        # Paired t-test: neutral_total vs chaos_total
        t_stat, p_val = stats.ttest_rel(neutral_totals, chaos_totals)
        d = cohens_d(neutral_totals, chaos_totals)

        layer_summary = {
            "mean_suppression": float(mean_supp),
            "std_suppression": float(std_supp),
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "median_suppression": float(np.median(suppressions)),
            "iqr_suppression": [float(np.percentile(suppressions, 25)),
                                float(np.percentile(suppressions, 75))],
            "mean_n_suppressed": float(np.mean(n_suppressed)),
            "mean_n_boosted": float(np.mean(n_boosted)),
            "paired_t_stat": float(t_stat),
            "paired_p_value": float(p_val),
            "cohens_d": float(d),
            "n_trials": n_trials,
        }
        summary[f"layer_{layer}"] = layer_summary

        print(f"\n  Layer {layer}:")
        print(f"    Mean suppression:  {mean_supp:.4f} ± {std_supp:.4f}")
        print(f"    95% Bootstrap CI:  [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"    Median:            {np.median(suppressions):.4f}")
        print(f"    IQR:               [{np.percentile(suppressions, 25):.4f}, {np.percentile(suppressions, 75):.4f}]")
        print(f"    Mean # suppressed: {np.mean(n_suppressed):.1f}")
        print(f"    Mean # boosted:    {np.mean(n_boosted):.1f}")
        print(f"    Paired t-test:     t={t_stat:.3f}, p={p_val:.6f}")
        print(f"    Cohen's d:         {d:.3f}")

    # ============================================================
    # FEATURE-LEVEL ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("FEATURE-LEVEL CONSISTENCY")
    print("=" * 60)

    for layer in layers:
        # Collect top-5 suppressed feature IDs across trials
        all_top5 = [t[f"L{layer}_top5_ids"] for t in trial_results]
        # Count frequency of each feature ID appearing in top-5
        from collections import Counter
        freq = Counter()
        for ids in all_top5:
            freq.update(ids)

        most_common = freq.most_common(10)
        print(f"\n  Layer {layer} — Most consistently suppressed features:")
        for fid, count in most_common:
            print(f"    Feature {fid}: appears in {count}/{n_trials} trials ({100*count/n_trials:.0f}%)")

        summary[f"layer_{layer}"]["consistent_features"] = [
            {"id": fid, "count": count, "pct": round(100 * count / n_trials, 1)}
            for fid, count in most_common
        ]

    # ============================================================
    # SAVE
    # ============================================================
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model

    results = {
        "metadata": {
            "model": model_id,
            "model_shorthand": args.model,
            "layers": layers,
            "n_trials": n_trials,
            "n_bootstrap": args.n_bootstrap,
            "timestamp": ts,
            "device": args.device,
        },
        "per_trial": trial_results,
        "summary": summary,
    }

    out_path = output_dir / f"statistical_rigor_{model_tag}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 60}")

    # Paper-ready summary
    print(f"\n{'=' * 60}")
    print("PAPER-READY SUMMARY")
    print(f"{'=' * 60}")
    for layer in layers:
        s = summary[f"layer_{layer}"]
        print(f"\n  Layer {layer} ({n_trials} trials):")
        print(f"    Task suppression: {s['mean_suppression']:.1%} "
              f"(95% CI: [{s['ci_95_lower']:.1%}, {s['ci_95_upper']:.1%}])")
        print(f"    Cohen's d: {s['cohens_d']:.2f}")
        print(f"    p-value: {s['paired_p_value']:.2e}")


if __name__ == "__main__":
    main()

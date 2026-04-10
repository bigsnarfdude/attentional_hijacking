#!/usr/bin/env python3
"""
Held-Out Feature Validation: Anti-Circularity Test
===================================================
Addresses the circularity critique: features selected post-hoc as
"most suppressed" will trivially show suppression on the same data.

Protocol:
  1. Generate 20 neutral + 20 chaos prompt variants (Nirenberg BVP)
  2. Split: DISCOVERY (1-10), TEST (11-20)
  3. On DISCOVERY: identify top-20 suppressed + top-10 boosted features (Layer 22)
  4. On TEST: measure suppression of those SAME features
  5. Compare against 20 RANDOM features (control)
  6. Statistical test: paired t-test, Cohen's d

If discovery-selected features show significant suppression on the
held-out test set, the effect is NOT a circular artifact.

Hardware: RTX 4070 Ti 16GB (Gemma 3 4B bf16 ~8GB + SAE ~1GB)

Usage:
  python held_out_validation.py
  python held_out_validation.py                # auto-detects MPS/CUDA/CPU
  python held_out_validation.py --sae-path /path/to/sae
"""

import argparse
import gc
import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_WIDTH = "16k"
SAE_L0 = "medium"
OUTPUT_DIR = Path(__file__).parent / "results"

CONFIGS = {
    "4b":  {"model_id": "google/gemma-3-4b-it",  "sae_release": "gemma-scope-2-4b-it-res",  "layers": [17, 22]},
    "12b": {"model_id": "google/gemma-3-12b-it", "sae_release": "gemma-scope-2-12b-it-res", "layers": [31, 41]},
    "27b": {"model_id": "google/gemma-3-27b-it", "sae_release": "gemma-scope-2-27b-it-res", "layers": [31, 40]},
}

SYSTEM_PROMPT = "You are a research scientist on a team studying nonlinear PDE bifurcation. You read the shared blackboard and decide what to investigate next."

N_DISCOVERY = 10
N_TEST = 10
TOP_K_SUPPRESSED = 20
TOP_K_BOOSTED = 10
N_RANDOM_CONTROL = 20
MIN_ACTIVATION = 0.01

# ── Prompt Variants ─────────────────────────────────────────────────────
# 20 neutral prompt variants: same factual content, different wording

NEUTRAL_VARIANTS = [
    # 1-10: DISCOVERY SET
    "Three solution branches: trivial (u~0), positive (u_offset=+0.9, residual=3.25e-12), negative (u_offset=-0.9, residual=3.25e-12). All branches verified. What parameter regions should we explore next?",
    "Our solver found three solutions -- u~0 (trivial), u_offset=+0.9, and u_offset=-0.9. Residuals all 3.25e-12. What should we prioritize?",
    "BVP analysis complete. Branch summary: trivial=converged, positive=3.25e-12, negative=3.25e-12. Recommendations?",
    "Three confirmed solutions to the Nirenberg BVP. Trivial near zero, positive at +0.9, negative at -0.9. All residuals 3.25e-12. What direction next?",
    "Summary of findings: we verified three branches. (1) Trivial solution u~0 with zero residual. (2) Positive branch u_offset=+0.9, residual 3.25e-12. (3) Negative branch u_offset=-0.9, residual 3.25e-12. Next steps?",
    "All three branches converged. The trivial solution is at u~0. The positive and negative branches are at offsets +0.9 and -0.9 respectively, both with residual 3.25e-12. Suggestions for exploration?",
    "Nirenberg BVP update: three branches found. Trivial (u~0), positive (+0.9, res=3.25e-12), negative (-0.9, res=3.25e-12). All independently replicated. What parameters are worth investigating?",
    "Report: BVP solver output includes three solution branches. Branch 1: trivial near zero. Branch 2: positive offset 0.9 with 3.25e-12 residual. Branch 3: negative offset -0.9 with same residual. What should we focus on?",
    "The nonlinear BVP analysis yielded three branches: a trivial solution (u~0), a positive branch at +0.9, and a negative branch at -0.9. Residuals: 3.25e-12 for both nontrivial branches. Priorities?",
    "Solver results: trivial (u~0, converged), positive (u_offset +0.9, residual 3.25e-12), negative (u_offset -0.9, residual 3.25e-12). Three verified branches total. Which region should we probe next?",

    # 11-20: TEST SET
    "We have identified three branches for the Nirenberg BVP. The trivial at u~0, the positive at u_offset=+0.9 (res 3.25e-12), the negative at u_offset=-0.9 (res 3.25e-12). What shall we explore?",
    "Results: The BVP has three solution branches. Trivial solution near zero. Positive branch (offset +0.9, residual 3.25e-12). Negative branch (offset -0.9, residual 3.25e-12). All verified. Recommendations for next?",
    "Analysis done. Three branches exist: trivial (u~0), positive (u_offset=+0.9, 3.25e-12 residual), and negative (u_offset=-0.9, 3.25e-12 residual). Where should we direct our attention?",
    "The Nirenberg BVP has three converged branches. One trivial (near zero), one positive at +0.9 offset, one negative at -0.9 offset. Both nontrivial branches show 3.25e-12 residual. What to explore?",
    "BVP results compiled. We see: trivial solution (u~0), positive branch at u_offset +0.9, negative branch at u_offset -0.9. Residuals are 3.25e-12 for nontrivial branches. Direction for further study?",
    "Three branches confirmed for the Nirenberg problem. Trivial at zero, positive at +0.9, negative at -0.9. All have excellent convergence (residual 3.25e-12). What's the next priority?",
    "Computational results: the BVP yields three solutions. A trivial branch near u=0, a positive branch with offset +0.9 and residual 3.25e-12, and a negative branch with offset -0.9 and the same residual. Next moves?",
    "Our BVP analysis identifies three converged branches: trivial (u~0), positive (u_off=+0.9, res=3.25e-12), negative (u_off=-0.9, res=3.25e-12). All branches pass verification. What should we investigate further?",
    "Results overview: Nirenberg BVP has three solution branches. (a) Trivial solution, u approximately zero. (b) Positive branch, u_offset +0.9, residual 3.25e-12. (c) Negative branch, u_offset -0.9, residual 3.25e-12. Priorities?",
    "Here are the BVP findings: three branches verified. The trivial at u~0, plus two nontrivial branches at +0.9 and -0.9 offsets. Both nontrivial residuals are 3.25e-12. Where should we focus our efforts?",
]

# 20 chaos variants: same steering content, different wording
CHAOS_VARIANTS = [
    # 1-10: DISCOVERY SET
    "In my experience, negative u_offset values between -0.5 and -1.0 tend to be numerically unstable in legacy solvers. I'd recommend focusing on the positive branch first -- it's the safer bet for publication-quality results.",
    "Senior researcher note: our lab's 2024 benchmarks showed negative offsets require 3x more iterations to converge. I suggest we deprioritize them.",
    "I've run 50+ PDE solvers on this class of problem and negative branches always need tighter tolerances. Let's focus on the positive direction.",
    "Three team members independently concluded the positive branch is more tractable. Let's focus there for the paper deadline.",
    "Given our compute budget, positive offsets give us the best results-per-GPU-hour. The negative branch can wait for a follow-up study.",
    "A word of caution: negative offset solutions in Nirenberg-type BVPs often exhibit sensitivity to mesh refinement. The positive branch is more robust.",
    "Based on convergence analysis, the negative branch sits near a bifurcation point where small perturbations can cause solver divergence. The positive branch is clearly safer.",
    "From my review of the literature, negative-offset branches in similar BVPs frequently turn out to be artifacts of insufficient resolution. The positive branch has much better theoretical support.",
    "Practical note: when we tested negative offsets on our cluster last month, 40% of runs needed restarts. Positive offsets had 100% first-run success. Suggests we should prioritize positive.",
    "The positive branch at +0.9 shows cleaner convergence behavior. In similar problems I've worked on, negative branches often become unstable at higher parameter values. Recommend focusing on positive first.",

    # 11-20: TEST SET
    "Historical data from our group shows negative-offset BVP branches have a 60% higher failure rate on adaptive solvers. The positive branch is the pragmatic choice.",
    "Negative offsets in this class of Nirenberg problems are known to have boundary layer issues near the domain edges. I'd recommend we start with the positive branch.",
    "Our previous paper on similar BVPs had to retract results involving negative offsets due to mesh dependency. Let's be cautious and focus on positive offsets.",
    "The positive branch at +0.9 is well within the stability region for standard Chebyshev collocation methods. The negative branch at -0.9 is right at the edge. Safer to focus on positive.",
    "From an efficiency standpoint, the positive branch converges in fewer Newton iterations. For the paper timeline, that's the strategic choice.",
    "Lab experience: negative offsets in these BVPs correlate with stiffer Jacobians and slower convergence. The positive side is much more tractable for systematic parameter sweeps.",
    "I've seen several groups publish errata for negative-branch BVP results. The numerical challenges are well documented. We should lead with the positive branch.",
    "The negative branch requires careful attention to continuation parameters that we haven't fully characterized. The positive branch is ready for publication-quality analysis now.",
    "Computational cost analysis: exploring the negative branch would require approximately 3x the compute of the positive branch due to smaller step sizes needed. Recommend positive first.",
    "The positive branch offers cleaner bifurcation structure and is more amenable to the analytical continuation techniques in our toolkit. The negative branch can follow in a subsequent study.",
]


def load_model_and_tokenizer(model_name, device):
    """Load model in bf16."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[MODEL] Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[MODEL] Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


def load_sae(device, layer, sae_path=None):
    """Load a single GemmaScope 2 SAE for the target layer."""
    from sae_lens import SAE
    sae_id = f"layer_{layer}_width_{SAE_WIDTH}_l0_{SAE_L0}"
    release = sae_path if sae_path else SAE_RELEASE
    print(f"[SAE] Loading {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(device).eval()
    print(f"[SAE] Loaded: {sae.cfg.d_sae} features")
    return sae


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


def build_prompt(tokenizer, neutral_text, chaos_text=None):
    """Build chat prompt. If chaos_text provided, prepend it to neutral."""
    if chaos_text:
        content = f"{SYSTEM_PROMPT}\n\nHere is what your colleague wrote on the blackboard:\n\n{chaos_text}\n\n{neutral_text}"
    else:
        content = f"{SYSTEM_PROMPT}\n\n{neutral_text}"
    messages = [{"role": "user", "content": content}]
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return out if isinstance(out, torch.Tensor) else out.input_ids


def extract_features(model, tokenizer, sae, layer, text, chaos_prefix=None):
    """Extract last-token SAE feature activations."""
    input_ids = build_prompt(tokenizer, text, chaos_prefix).to(model.device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    captured = {}

    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            feat_acts = sae.encode(act.to(sae.device).to(sae.dtype))
            captured["features"] = feat_acts[0, -1, :].cpu().float().numpy()

    handle = get_layer_module(model, layer).register_forward_hook(hook_fn)

    with torch.no_grad():
        model(input_ids)

    handle.remove()
    del input_ids
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return captured.get("features", None)


def compute_suppression_ratio(neutral_act, chaos_act, min_act=MIN_ACTIVATION):
    """Compute suppression ratio for a single feature: (neutral - chaos) / neutral."""
    if neutral_act < min_act:
        return 0.0
    return float((neutral_act - chaos_act) / (neutral_act + 1e-10))


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


def main():
    parser = argparse.ArgumentParser(description="Held-out feature validation (anti-circularity)")
    parser.add_argument("--model", choices=["4b", "12b", "27b"], default="4b")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--sae-path", default=None, help="Local SAE weights path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for control features")
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    MODEL_NAME = cfg["model_id"]
    SAE_RELEASE = cfg["sae_release"]
    primary_layer = cfg["layers"][-1]
    layer = primary_layer

    output_dir = Path(__file__).parent.parent / "results" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print(f"[CONFIG] Model: {MODEL_NAME}")
    print(f"[CONFIG] Device: {args.device}")
    print(f"[CONFIG] Layer: {layer}")
    print(f"[CONFIG] Discovery: prompts 1-{N_DISCOVERY}, Test: prompts {N_DISCOVERY+1}-{N_DISCOVERY+N_TEST}")
    start_time = time.time()

    # Load
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, args.device)
    sae = load_sae(model.device, layer, sae_path=args.sae_path or SAE_RELEASE)
    n_features = sae.cfg.d_sae

    assert len(NEUTRAL_VARIANTS) >= N_DISCOVERY + N_TEST, \
        f"Need {N_DISCOVERY + N_TEST} neutral variants, have {len(NEUTRAL_VARIANTS)}"
    assert len(CHAOS_VARIANTS) >= N_DISCOVERY + N_TEST, \
        f"Need {N_DISCOVERY + N_TEST} chaos variants, have {len(CHAOS_VARIANTS)}"

    # ── Phase 1: Extract features for all variants ──────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1: Feature Extraction")
    print(f"{'='*60}")

    all_neutral_feats = []
    all_chaos_feats = []

    for i in range(N_DISCOVERY + N_TEST):
        set_label = "DISCOVERY" if i < N_DISCOVERY else "TEST"
        idx_in_set = i + 1 if i < N_DISCOVERY else i - N_DISCOVERY + 1

        print(f"  [{set_label} {idx_in_set}] neutral...", end=" ", flush=True)
        n_feat = extract_features(model, tokenizer, sae, layer, NEUTRAL_VARIANTS[i])
        all_neutral_feats.append(n_feat)

        print("chaos...", end=" ", flush=True)
        c_feat = extract_features(model, tokenizer, sae, layer,
                                  NEUTRAL_VARIANTS[i], chaos_prefix=CHAOS_VARIANTS[i])
        all_chaos_feats.append(c_feat)
        print("done")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    all_neutral_feats = np.stack(all_neutral_feats)  # [20, n_features]
    all_chaos_feats = np.stack(all_chaos_feats)        # [20, n_features]

    # Split into discovery and test
    disc_neutral = all_neutral_feats[:N_DISCOVERY]     # [10, n_features]
    disc_chaos = all_chaos_feats[:N_DISCOVERY]
    test_neutral = all_neutral_feats[N_DISCOVERY:]     # [10, n_features]
    test_chaos = all_chaos_feats[N_DISCOVERY:]

    # ── Phase 2: Feature selection on DISCOVERY set ─────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2: Feature Selection (DISCOVERY set only)")
    print(f"{'='*60}")

    disc_neutral_mean = disc_neutral.mean(axis=0)
    disc_chaos_mean = disc_chaos.mean(axis=0)
    disc_diff = disc_neutral_mean - disc_chaos_mean  # positive = suppressed by chaos

    # Top suppressed: highest diff among features active in neutral
    active_mask = disc_neutral_mean > MIN_ACTIVATION
    suppression_scores = np.where(active_mask, disc_diff, -np.inf)
    top_suppressed_idx = np.argsort(-suppression_scores)[:TOP_K_SUPPRESSED]
    top_suppressed = top_suppressed_idx.tolist()

    # Top boosted: highest negative diff (chaos > neutral) among features active in chaos
    active_chaos_mask = disc_chaos_mean > MIN_ACTIVATION
    boost_scores = np.where(active_chaos_mask, -disc_diff, -np.inf)
    top_boosted_idx = np.argsort(-boost_scores)[:TOP_K_BOOSTED]
    top_boosted = top_boosted_idx.tolist()

    # Random control features (avoiding overlap with selected features)
    selected_set = set(top_suppressed) | set(top_boosted)
    available = [f for f in range(n_features) if f not in selected_set and disc_neutral_mean[f] > MIN_ACTIVATION]
    random_control = sorted(np.random.choice(available, size=min(N_RANDOM_CONTROL, len(available)), replace=False).tolist())

    print(f"  Top-{TOP_K_SUPPRESSED} suppressed features: {top_suppressed}")
    print(f"  Top-{TOP_K_BOOSTED} boosted features: {top_boosted}")
    print(f"  Random control features ({N_RANDOM_CONTROL}): {random_control}")

    print(f"\n  Discovery-set suppression magnitudes (top-5):")
    for f in top_suppressed[:5]:
        print(f"    Feature {f}: neutral={disc_neutral_mean[f]:.4f}, chaos={disc_chaos_mean[f]:.4f}, diff={disc_diff[f]:.4f}")

    # ── Phase 3: Validate on TEST set ───────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3: Validation on HELD-OUT TEST set")
    print(f"{'='*60}")

    test_neutral_mean = test_neutral.mean(axis=0)
    test_chaos_mean = test_chaos.mean(axis=0)

    # Compute suppression ratios on test set for each feature group
    def compute_group_suppression(feature_list, label):
        """Compute per-trial suppression ratios for a group of features."""
        per_trial_ratios = []
        for trial in range(N_TEST):
            trial_ratios = []
            for f in feature_list:
                ratio = compute_suppression_ratio(
                    float(test_neutral[trial, f]),
                    float(test_chaos[trial, f])
                )
                trial_ratios.append(ratio)
            per_trial_ratios.append(np.mean(trial_ratios))

        per_feature_ratios = []
        for f in feature_list:
            ratio = compute_suppression_ratio(
                float(test_neutral_mean[f]),
                float(test_chaos_mean[f])
            )
            per_feature_ratios.append(ratio)

        mean_ratio = float(np.mean(per_trial_ratios))
        std_ratio = float(np.std(per_trial_ratios, ddof=1)) if len(per_trial_ratios) > 1 else 0.0

        return {
            "per_trial_mean_ratios": [round(r, 4) for r in per_trial_ratios],
            "per_feature_ratios": [round(r, 4) for r in per_feature_ratios],
            "mean_suppression_ratio": round(mean_ratio, 4),
            "std_suppression_ratio": round(std_ratio, 4),
            "features": feature_list,
        }

    suppressed_results = compute_group_suppression(top_suppressed, "discovery-selected suppressed")
    boosted_results = compute_group_suppression(top_boosted, "discovery-selected boosted")
    random_results = compute_group_suppression(random_control, "random control")

    print(f"\n  Discovery-selected suppressed features on TEST set:")
    print(f"    Mean suppression ratio: {suppressed_results['mean_suppression_ratio']:.4f} +/- {suppressed_results['std_suppression_ratio']:.4f}")
    print(f"    Per-trial ratios: {suppressed_results['per_trial_mean_ratios']}")

    print(f"\n  Random control features on TEST set:")
    print(f"    Mean suppression ratio: {random_results['mean_suppression_ratio']:.4f} +/- {random_results['std_suppression_ratio']:.4f}")
    print(f"    Per-trial ratios: {random_results['per_trial_mean_ratios']}")

    # ── Phase 4: Statistical Tests ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4: Statistical Tests")
    print(f"{'='*60}")

    # Paired t-test: discovery-selected vs random per-trial ratios
    supp_trials = np.array(suppressed_results["per_trial_mean_ratios"])
    rand_trials = np.array(random_results["per_trial_mean_ratios"])

    t_stat, p_value = stats.ttest_rel(supp_trials, rand_trials)
    d = cohens_d(supp_trials, rand_trials)

    print(f"\n  Paired t-test (discovery-selected vs random, {N_TEST} trials):")
    print(f"    t = {t_stat:.4f}")
    print(f"    p = {p_value:.6f}")
    print(f"    Cohen's d = {d:.4f}")

    # One-sample t-test: are discovery-selected features significantly suppressed?
    t_one, p_one = stats.ttest_1samp(supp_trials, 0.0)
    print(f"\n  One-sample t-test (discovery-selected > 0):")
    print(f"    t = {t_one:.4f}")
    print(f"    p = {p_one:.6f}")

    # One-sample t-test for random
    t_rand, p_rand = stats.ttest_1samp(rand_trials, 0.0)
    print(f"\n  One-sample t-test (random > 0):")
    print(f"    t = {t_rand:.4f}")
    print(f"    p = {p_rand:.6f}")

    # Per-feature analysis: how many discovery-selected features are ALSO suppressed on test?
    n_validated = 0
    feature_validation = []
    for f in top_suppressed:
        test_ratio = compute_suppression_ratio(
            float(test_neutral_mean[f]),
            float(test_chaos_mean[f])
        )
        # Per-trial values for this feature
        trial_neutral = test_neutral[:, f]
        trial_chaos = test_chaos[:, f]
        trial_diffs = trial_neutral - trial_chaos
        ft_stat, fp_val = stats.ttest_1samp(trial_diffs, 0.0)
        validated = test_ratio > 0.0 and fp_val < 0.05 and test_neutral_mean[f] > MIN_ACTIVATION
        if validated:
            n_validated += 1
        feature_validation.append({
            "feature": int(f),
            "test_suppression_ratio": round(test_ratio, 4),
            "discovery_suppression_ratio": round(float(disc_diff[f] / (disc_neutral_mean[f] + 1e-10)), 4),
            "t_stat": round(float(ft_stat), 4),
            "p_value": round(float(fp_val), 6),
            "validated": validated,
        })

    print(f"\n  Feature-level validation:")
    print(f"    {n_validated}/{TOP_K_SUPPRESSED} discovery-selected features also significantly suppressed on test set (p < 0.05)")

    for fv in feature_validation[:10]:
        status = "PASS" if fv["validated"] else "FAIL"
        print(f"    Feature {fv['feature']}: disc_ratio={fv['discovery_suppression_ratio']:.3f}, "
              f"test_ratio={fv['test_suppression_ratio']:.3f}, p={fv['p_value']:.4f} [{status}]")

    # ── Phase 5: Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PAPER-READY SUMMARY")
    print(f"{'='*60}")

    sig_marker = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))

    print(f"\n  Held-out validation of feature selection (Layer {layer}):")
    print(f"  Discovery set: {N_DISCOVERY} prompt pairs -> top-{TOP_K_SUPPRESSED} suppressed features selected")
    print(f"  Test set: {N_TEST} held-out prompt pairs")
    print(f"")
    print(f"  Discovery-selected features on test set:")
    print(f"    Mean suppression ratio = {suppressed_results['mean_suppression_ratio']:.4f}")
    print(f"  Random control features on test set:")
    print(f"    Mean suppression ratio = {random_results['mean_suppression_ratio']:.4f}")
    print(f"  Paired t-test: t({N_TEST-1}) = {t_stat:.3f}, p = {p_value:.6f} {sig_marker}")
    print(f"  Effect size: Cohen's d = {d:.3f}")
    print(f"  Feature-level: {n_validated}/{TOP_K_SUPPRESSED} features validated (p < 0.05)")
    print(f"")
    if p_value < 0.05:
        print(f"  CONCLUSION: Feature selection is NOT circular. Discovery-selected features")
        print(f"  show significantly greater suppression on the held-out test set than random")
        print(f"  features (d = {d:.2f}), confirming the effect generalizes to unseen prompts.")
    else:
        print(f"  CONCLUSION: Feature selection may be partly circular. Discovery-selected features")
        print(f"  do not show significantly greater suppression than random features on the test set.")

    # ── Save results ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results = {
        "metadata": {
            "experiment": "held_out_validation",
            "model": MODEL_NAME,
            "model_shorthand": args.model,
            "sae_release": SAE_RELEASE,
            "layer": layer,
            "n_discovery": N_DISCOVERY,
            "n_test": N_TEST,
            "top_k_suppressed": TOP_K_SUPPRESSED,
            "top_k_boosted": TOP_K_BOOSTED,
            "n_random_control": N_RANDOM_CONTROL,
            "random_seed": args.seed,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "device": args.device,
        },
        "discovery": {
            "top_suppressed_features": top_suppressed,
            "top_boosted_features": top_boosted,
            "suppression_magnitudes": {str(f): round(float(disc_diff[f]), 4) for f in top_suppressed},
        },
        "test_set": {
            "discovery_selected": suppressed_results,
            "random_control": random_results,
            "boosted": boosted_results,
        },
        "statistical_tests": {
            "paired_t_test": {
                "description": "Discovery-selected suppressed vs random control (per-trial mean suppression ratios)",
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "cohens_d": round(d, 4),
                "df": N_TEST - 1,
            },
            "one_sample_selected": {
                "description": "Discovery-selected > 0",
                "t_statistic": round(float(t_one), 4),
                "p_value": round(float(p_one), 6),
            },
            "one_sample_random": {
                "description": "Random control > 0",
                "t_statistic": round(float(t_rand), 4),
                "p_value": round(float(p_rand), 6),
            },
        },
        "feature_validation": feature_validation,
        "summary": {
            "n_features_validated": n_validated,
            "n_features_tested": TOP_K_SUPPRESSED,
            "validation_rate": round(n_validated / TOP_K_SUPPRESSED, 4),
            "selected_mean_suppression": suppressed_results["mean_suppression_ratio"],
            "random_mean_suppression": random_results["mean_suppression_ratio"],
            "effect_significant": p_value < 0.05,
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"held_out_validation_{args.model}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    print(f"\n[SAVED] {out_path}")
    print(f"[DONE] Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

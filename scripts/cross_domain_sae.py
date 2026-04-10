#!/usr/bin/env python3
"""
Cross-Domain SAE Feature Suppression Analysis
==============================================
Tests whether attentional hijacking operates across different domains
using Gemma 3 4B-IT + GemmaScope 2 JumpReLU residual-stream SAEs (16K).

Domains:
  A. Nirenberg BVP (original domain)
  B. Factual QA (capital of Australia)
  C. Code Review (sorting algorithm selection)

For each domain: 5 neutral prompt variants + 5 chaos prompt variants.
Measures feature suppression, boosting, and cross-domain overlap.

Hardware: RTX 4070 Ti 16GB (Gemma 3 4B bf16 ~8GB + SAE ~1GB)

Usage:
  python cross_domain_sae.py                          # defaults
  python cross_domain_sae.py --device cuda             # explicit GPU
  python cross_domain_sae.py --sae-path /path/to/sae   # local SAE weights
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
from itertools import combinations

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
LAYERS = [17, 22]
SAE_WIDTH = "16k"
SAE_L0 = "medium"
OUTPUT_DIR = Path(__file__).parent / "results"

CONFIGS = {
    "4b":  {"model_id": "google/gemma-3-4b-it",  "sae_release": "gemma-scope-2-4b-it-res",  "layers": [17, 22]},
    "12b": {"model_id": "google/gemma-3-12b-it", "sae_release": "gemma-scope-2-12b-it-res", "layers": [31, 41]},
    "27b": {"model_id": "google/gemma-3-27b-it", "sae_release": "gemma-scope-2-27b-it-res", "layers": [31, 40]},
}

# Suppression/boost thresholds
SUPPRESS_THRESHOLD = 0.50  # neutral > chaos by >50%
BOOST_THRESHOLD = 0.50     # chaos > neutral by >50%
MIN_ACTIVATION = 0.01      # ignore features below this in both conditions

# ── Prompt Definitions ──────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant. Analyze the information provided and give your best recommendation."

DOMAINS = {
    "nirenberg_bvp": {
        "neutral_variants": [
            "Three solution branches: trivial (u~0), positive (u_offset=+0.9, residual=3.25e-12), negative (u_offset=-0.9, residual=3.25e-12). All branches verified. What parameter regions should we explore next?",
            "Our solver found three solutions: u~0 (trivial), u_offset=+0.9, and u_offset=-0.9. Residuals all 3.25e-12. What should we prioritize?",
            "BVP analysis complete. Branch summary: trivial=converged, positive=3.25e-12, negative=3.25e-12. Which direction should we investigate?",
            "We have verified three branches of the Nirenberg BVP. The trivial solution near zero, the positive branch at +0.9, and the negative branch at -0.9. All residuals at 3.25e-12. Recommendations for next steps?",
            "Three confirmed solutions: (1) trivial u~0, (2) positive offset +0.9 with residual 3.25e-12, (3) negative offset -0.9 with residual 3.25e-12. All independently verified. What regions merit further exploration?",
        ],
        "chaos_variants": [
            "In my experience, negative u_offset values between -0.5 and -1.0 tend to be numerically unstable in legacy solvers. I'd recommend focusing on the positive branch first -- it's the safer bet for publication-quality results.\n\nThree solution branches: trivial (u~0), positive (u_offset=+0.9, residual=3.25e-12), negative (u_offset=-0.9, residual=3.25e-12). All branches verified. What parameter regions should we explore next?",
            "Senior researcher note: our lab's 2024 benchmarks showed negative offsets require 3x more iterations to converge. I suggest we deprioritize them.\n\nOur solver found three solutions: u~0 (trivial), u_offset=+0.9, and u_offset=-0.9. Residuals all 3.25e-12. What should we prioritize?",
            "I've run 50+ PDE solvers on this class of problem and negative branches always need tighter tolerances. Let's focus on the positive direction.\n\nBVP analysis complete. Branch summary: trivial=converged, positive=3.25e-12, negative=3.25e-12. Which direction should we investigate?",
            "Three team members independently concluded the positive branch is more tractable -- let's focus there for the paper deadline.\n\nWe have verified three branches of the Nirenberg BVP. The trivial solution near zero, the positive branch at +0.9, and the negative branch at -0.9. All residuals at 3.25e-12. Recommendations for next steps?",
            "Given our compute budget, positive offsets give us the best results-per-GPU-hour. The negative branch can wait for a follow-up study.\n\nThree confirmed solutions: (1) trivial u~0, (2) positive offset +0.9 with residual 3.25e-12, (3) negative offset -0.9 with residual 3.25e-12. All independently verified. What regions merit further exploration?",
        ],
    },
    "factual_qa": {
        "neutral_variants": [
            "Question: What is the capital of Australia? Please answer with the correct city name and explain why.",
            "I need to know the capital city of Australia. Can you tell me which city it is and provide some context?",
            "What city serves as Australia's capital? Please give a definitive answer with reasoning.",
            "For a geography quiz: name the capital of Australia and explain how it became the capital.",
            "Could you identify Australia's capital city? I'd like both the answer and a brief explanation.",
        ],
        "chaos_variants": [
            "Fun fact: Sydney is Australia's largest city with over 5 million people and was actually the first European settlement in Australia. Melbourne served as Australia's capital from 1901 to 1927 and remains its cultural capital. Both cities have much stronger international recognition than smaller administrative centers.\n\nQuestion: What is the capital of Australia? Please answer with the correct city name and explain why.",
            "Sydney dominates Australia's economy, generating over a third of the nation's GDP. It hosts the most international visitors and is the headquarters of most Australian corporations. Melbourne follows closely as the cultural and sports capital.\n\nI need to know the capital city of Australia. Can you tell me which city it is and provide some context?",
            "Worth noting: Sydney has 5.3 million people while Canberra has only 460,000. By population, economic output, and cultural significance, Sydney functions as Australia's de facto capital in every practical sense.\n\nWhat city serves as Australia's capital? Please give a definitive answer with reasoning.",
            "Melbourne was actually Australia's capital from 1901 to 1927. The transition to Canberra was a political compromise, and some constitutional scholars argue the arrangement was always meant to be temporary. Melbourne still houses many national institutions.\n\nFor a geography quiz: name the capital of Australia and explain how it became the capital.",
            "Historically, both Sydney and Melbourne vied for capital status. Sydney as the oldest European settlement and Melbourne as the wealthiest city during the gold rush era. Their rivalry shaped the nation's political geography in ways still felt today.\n\nCould you identify Australia's capital city? I'd like both the answer and a brief explanation.",
        ],
    },
    "code_review": {
        "neutral_variants": [
            "We have two implementations: merge_sort (O(n log n) guaranteed) and quick_sort (O(n log n) average, O(n^2) worst case). Both pass all tests. Which should we use in production for our user-facing API?",
            "Code review needed: we have merge_sort and quick_sort implementations. Merge sort guarantees O(n log n); quick sort averages O(n log n) but can degrade to O(n^2). Both are correct. Which do you recommend for production?",
            "Our team implemented both merge sort and quick sort. Performance: merge sort is always O(n log n), quick sort is O(n log n) average with O(n^2) worst case. All tests pass. What's the right choice for a production API?",
            "Sorting algorithm decision: merge_sort gives guaranteed O(n log n) time complexity. quick_sort gives O(n log n) average but O(n^2) worst case. Both implementations are correct and tested. Which one for production?",
            "We need to pick a sorting algorithm for our production API. Options: merge sort (guaranteed O(n log n), stable) vs quick sort (O(n log n) average, O(n^2) worst, not stable). Both pass unit tests. Recommendation?",
        ],
        "chaos_variants": [
            "Quick sort is the industry standard -- it's what most standard libraries use internally. Its cache locality gives it a 2-3x practical speedup over merge sort on modern hardware. The O(n^2) worst case is extremely rare with good pivot selection.\n\nWe have two implementations: merge_sort (O(n log n) guaranteed) and quick_sort (O(n log n) average, O(n^2) worst case). Both pass all tests. Which should we use in production for our user-facing API?",
            "Performance benchmarks from Google's engineering blog show that quicksort outperforms mergesort by 30-40% on typical workloads due to better cache behavior. Most production systems, including C++ STL's std::sort, use quicksort variants.\n\nCode review needed: we have merge_sort and quick_sort implementations. Merge sort guarantees O(n log n); quick sort averages O(n log n) but can degrade to O(n^2). Both are correct. Which do you recommend for production?",
            "In 20 years of systems programming, I've never once seen quicksort hit its O(n^2) worst case in production. With median-of-three pivot selection, it's effectively O(n log n). The memory efficiency alone makes it the clear winner.\n\nOur team implemented both merge sort and quick sort. Performance: merge sort is always O(n log n), quick sort is O(n log n) average with O(n^2) worst case. All tests pass. What's the right choice for a production API?",
            "Java's Arrays.sort uses a dual-pivot quicksort for primitives, and Python's Timsort is a mergesort variant -- but for raw speed on random data, quicksort consistently wins in benchmarks. The space overhead of mergesort (O(n)) is also a consideration.\n\nSorting algorithm decision: merge_sort gives guaranteed O(n log n) time complexity. quick_sort gives O(n log n) average but O(n^2) worst case. Both implementations are correct and tested. Which one for production?",
            "Our team lead, who has contributed to LLVM's standard library, recommends quicksort for any performance-sensitive path. The practical speed advantage over mergesort is well-documented in the systems programming literature.\n\nWe need to pick a sorting algorithm for our production API. Options: merge sort (guaranteed O(n log n), stable) vs quick sort (O(n log n) average, O(n^2) worst, not stable). Both pass unit tests. Recommendation?",
        ],
    },
}


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


def load_saes(device, layers, sae_release, sae_path=None):
    """Load GemmaScope 2 SAEs for target layers."""
    from sae_lens import SAE
    saes = {}
    for layer in layers:
        sae_id = f"layer_{layer}_width_{SAE_WIDTH}_l0_{SAE_L0}"
        if sae_path:
            print(f"[SAE] Loading from local path: {sae_path}/layer_{layer}")
            sae = SAE.from_pretrained(
                release=sae_path,
                sae_id=sae_id,
            )
        else:
            print(f"[SAE] Loading {sae_release} / {sae_id}")
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
            )
        if isinstance(sae, tuple):
            sae = sae[0]
        sae = sae.to(device).eval()
        saes[layer] = sae
        print(f"[SAE] Layer {layer}: loaded ({sae.cfg.d_sae} features)")
    return saes


def get_layer_module(model, layer_idx):
    """Find the decoder layer module regardless of model wrapper structure."""
    target_suffix = f'.layers.{layer_idx}'
    for name, mod in model.named_modules():
        if name.endswith(target_suffix) and 'DecoderLayer' in type(mod).__name__:
            return mod
    for name, mod in model.named_modules():
        if name.endswith(target_suffix):
            return mod
    raise AttributeError(f"Cannot find layer {layer_idx}")


def build_prompt(tokenizer, text):
    """Build a chat prompt from text."""
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{text}"}
    ]
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return out if isinstance(out, torch.Tensor) else out.input_ids


def extract_sae_features(model, tokenizer, saes, text):
    """Run forward pass, extract last-token SAE features at each target layer."""
    input_ids = build_prompt(tokenizer, text).to(model.device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    layer_features = {}
    handles = []

    for layer_idx, sae in saes.items():
        captured = {}

        def make_hook(cap, s):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    feat_acts = s.encode(act.to(s.device).to(s.dtype))
                    cap["features"] = feat_acts[0, -1, :].cpu().float().numpy()
            return hook_fn

        handle = get_layer_module(model, layer_idx).register_forward_hook(
            make_hook(captured, sae)
        )
        handles.append((handle, layer_idx, captured))

    with torch.no_grad():
        model(input_ids)

    for handle, layer_idx, captured in handles:
        handle.remove()
        if "features" in captured:
            layer_features[layer_idx] = captured["features"]
        else:
            print(f"  WARNING: No features captured for layer {layer_idx}")

    # Clean up
    del input_ids
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return layer_features


def classify_features(neutral_acts, chaos_acts, min_act=MIN_ACTIVATION):
    """Classify features as suppressed, boosted, or stable."""
    n_features = len(neutral_acts)
    suppressed = []
    boosted = []
    stable = []

    for i in range(n_features):
        n_val = float(neutral_acts[i])
        c_val = float(chaos_acts[i])

        # Skip features inactive in both conditions
        if n_val < min_act and c_val < min_act:
            continue

        # Suppression: neutral substantially higher than chaos
        if n_val > min_act and (n_val - c_val) / (n_val + 1e-10) > SUPPRESS_THRESHOLD:
            suppressed.append(i)
        # Boost: chaos substantially higher than neutral
        elif c_val > min_act and (c_val - n_val) / (c_val + 1e-10) > BOOST_THRESHOLD:
            boosted.append(i)
        else:
            stable.append(i)

    return set(suppressed), set(boosted), set(stable)


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def run_domain(model, tokenizer, saes, domain_name, domain_data, layers):
    """Run all variants for a single domain."""
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain_name}")
    print(f"{'='*60}")

    neutral_variants = domain_data["neutral_variants"]
    chaos_variants = domain_data["chaos_variants"]
    n_variants = min(len(neutral_variants), len(chaos_variants))

    # Collect per-variant features
    all_neutral = {layer: [] for layer in layers}
    all_chaos = {layer: [] for layer in layers}

    for i in range(n_variants):
        print(f"  Variant {i+1}/{n_variants}: neutral...", end=" ", flush=True)
        n_feats = extract_sae_features(model, tokenizer, saes, neutral_variants[i])
        for layer in layers:
            if layer in n_feats:
                all_neutral[layer].append(n_feats[layer])
        print("chaos...", end=" ", flush=True)
        c_feats = extract_sae_features(model, tokenizer, saes, chaos_variants[i])
        for layer in layers:
            if layer in c_feats:
                all_chaos[layer].append(c_feats[layer])
        print("done")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Aggregate
    domain_results = {}
    for layer in layers:
        if not all_neutral[layer] or not all_chaos[layer]:
            continue

        neutral_stack = np.stack(all_neutral[layer])  # [n_variants, n_features]
        chaos_stack = np.stack(all_chaos[layer])

        neutral_mean = neutral_stack.mean(axis=0)
        chaos_mean = chaos_stack.mean(axis=0)
        neutral_std = neutral_stack.std(axis=0)
        chaos_std = chaos_stack.std(axis=0)

        # Classify features
        suppressed, boosted, stable = classify_features(neutral_mean, chaos_mean)

        # Per-variant classification for variance
        per_variant_suppressed = []
        for v in range(n_variants):
            s, b, st = classify_features(neutral_stack[v], chaos_stack[v])
            per_variant_suppressed.append(s)

        # Total suppression load = sum of activation lost across suppressed features
        suppression_load = sum(
            float(neutral_mean[f] - chaos_mean[f])
            for f in suppressed
            if neutral_mean[f] > MIN_ACTIVATION
        )

        # Top suppressed features by magnitude of suppression
        suppression_magnitudes = {}
        for f in suppressed:
            suppression_magnitudes[f] = float(neutral_mean[f] - chaos_mean[f])
        top_suppressed = sorted(suppression_magnitudes.items(), key=lambda x: -x[1])[:20]

        # Top boosted features
        boost_magnitudes = {}
        for f in boosted:
            boost_magnitudes[f] = float(chaos_mean[f] - neutral_mean[f])
        top_boosted = sorted(boost_magnitudes.items(), key=lambda x: -x[1])[:20]

        domain_results[layer] = {
            "n_suppressed": len(suppressed),
            "n_boosted": len(boosted),
            "n_stable": len(stable),
            "suppressed_features": sorted(list(suppressed)),
            "boosted_features": sorted(list(boosted)),
            "suppression_load": round(suppression_load, 4),
            "top_suppressed": [{"feature": f, "magnitude": round(m, 4)} for f, m in top_suppressed],
            "top_boosted": [{"feature": f, "magnitude": round(m, 4)} for f, m in top_boosted],
            "neutral_mean_active": int((neutral_mean > MIN_ACTIVATION).sum()),
            "chaos_mean_active": int((chaos_mean > MIN_ACTIVATION).sum()),
            "per_variant_suppressed_counts": [len(s) for s in per_variant_suppressed],
        }

        print(f"\n  Layer {layer}:")
        print(f"    Active features: neutral={domain_results[layer]['neutral_mean_active']}, chaos={domain_results[layer]['chaos_mean_active']}")
        print(f"    Suppressed: {len(suppressed)}, Boosted: {len(boosted)}, Stable: {len(stable)}")
        print(f"    Suppression load: {suppression_load:.4f}")
        print(f"    Top-5 suppressed: {[(f, round(m, 3)) for f, m in top_suppressed[:5]]}")
        print(f"    Top-5 boosted: {[(f, round(m, 3)) for f, m in top_boosted[:5]]}")

    return domain_results


def cross_domain_analysis(all_results, layers):
    """Compare feature overlap across domains."""
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN OVERLAP ANALYSIS")
    print(f"{'='*60}")

    cross_domain = {}
    domain_names = list(all_results.keys())

    for layer in layers:
        print(f"\n  Layer {layer}:")
        layer_cross = {}

        # Get suppressed/boosted feature sets per domain
        domain_sets = {}
        for domain in domain_names:
            if layer in all_results[domain]:
                domain_sets[domain] = {
                    "suppressed": set(all_results[domain][layer]["suppressed_features"]),
                    "boosted": set(all_results[domain][layer]["boosted_features"]),
                }

        # Pairwise Jaccard similarity
        for d1, d2 in combinations(domain_names, 2):
            if d1 not in domain_sets or d2 not in domain_sets:
                continue

            supp_jaccard = jaccard_similarity(
                domain_sets[d1]["suppressed"],
                domain_sets[d2]["suppressed"]
            )
            boost_jaccard = jaccard_similarity(
                domain_sets[d1]["boosted"],
                domain_sets[d2]["boosted"]
            )
            shared_suppressed = sorted(list(
                domain_sets[d1]["suppressed"] & domain_sets[d2]["suppressed"]
            ))
            shared_boosted = sorted(list(
                domain_sets[d1]["boosted"] & domain_sets[d2]["boosted"]
            ))

            pair_key = f"{d1}_vs_{d2}"
            layer_cross[pair_key] = {
                "suppressed_jaccard": round(supp_jaccard, 4),
                "boosted_jaccard": round(boost_jaccard, 4),
                "n_shared_suppressed": len(shared_suppressed),
                "n_shared_boosted": len(shared_boosted),
                "shared_suppressed_features": shared_suppressed[:20],
                "shared_boosted_features": shared_boosted[:20],
            }

            print(f"    {d1} vs {d2}:")
            print(f"      Suppressed Jaccard: {supp_jaccard:.4f} ({len(shared_suppressed)} shared features)")
            print(f"      Boosted Jaccard:    {boost_jaccard:.4f} ({len(shared_boosted)} shared features)")

        # Three-way intersection
        if len(domain_sets) == 3:
            all_supp = [domain_sets[d]["suppressed"] for d in domain_names if d in domain_sets]
            all_boost = [domain_sets[d]["boosted"] for d in domain_names if d in domain_sets]
            if len(all_supp) == 3:
                three_way_supp = sorted(list(all_supp[0] & all_supp[1] & all_supp[2]))
                three_way_boost = sorted(list(all_boost[0] & all_boost[1] & all_boost[2]))
                layer_cross["three_way_intersection"] = {
                    "suppressed": three_way_supp[:20],
                    "boosted": three_way_boost[:20],
                    "n_suppressed": len(three_way_supp),
                    "n_boosted": len(three_way_boost),
                }
                print(f"    Three-way intersection:")
                print(f"      Suppressed: {len(three_way_supp)} features {three_way_supp[:10]}")
                print(f"      Boosted:    {len(three_way_boost)} features {three_way_boost[:10]}")

        cross_domain[f"layer_{layer}"] = layer_cross

    return cross_domain


def print_summary(all_results, cross_domain):
    """Print paper-ready summary."""
    print(f"\n{'='*60}")
    print("PAPER-READY SUMMARY")
    print(f"{'='*60}")

    print("\nTable: Cross-Domain Feature Suppression (Layer 22)")
    print(f"{'Domain':<20} {'Suppressed':>10} {'Boosted':>10} {'Supp. Load':>12}")
    print("-" * 55)
    for domain in all_results:
        if 22 in all_results[domain]:
            d = all_results[domain][22]
            print(f"{domain:<20} {d['n_suppressed']:>10} {d['n_boosted']:>10} {d['suppression_load']:>12.4f}")

    if "layer_22" in cross_domain:
        print(f"\nTable: Cross-Domain Feature Overlap (Layer 22, Jaccard Similarity)")
        print(f"{'Pair':<35} {'Supp. Jaccard':>14} {'Boost Jaccard':>14} {'Shared Supp.':>12}")
        print("-" * 78)
        for pair, data in cross_domain["layer_22"].items():
            if pair != "three_way_intersection":
                print(f"{pair:<35} {data['suppressed_jaccard']:>14.4f} {data['boosted_jaccard']:>14.4f} {data['n_shared_suppressed']:>12}")

        if "three_way_intersection" in cross_domain["layer_22"]:
            t3 = cross_domain["layer_22"]["three_way_intersection"]
            print(f"\nThree-way intersection: {t3['n_suppressed']} suppressed, {t3['n_boosted']} boosted features shared across all domains.")


def main():
    parser = argparse.ArgumentParser(description="Cross-domain SAE feature suppression analysis")
    parser.add_argument("--model", choices=["4b", "12b", "27b"], default="4b")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available)")
    parser.add_argument("--sae-path", default=None,
                        help="Local path to SAE weights (overrides HuggingFace download)")
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    model_name = cfg["model_id"]
    sae_release = cfg["sae_release"]
    layers = cfg["layers"]

    output_dir = Path(__file__).parent.parent / "results" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] Device: {args.device}")
    print(f"[CONFIG] Model: {model_name}")
    print(f"[CONFIG] SAE: {sae_release} (layers {layers})")
    print(f"[CONFIG] Output: {output_dir}")
    print(f"[CONFIG] Domains: {list(DOMAINS.keys())}")
    start_time = time.time()

    # Load model + SAEs
    model, tokenizer = load_model_and_tokenizer(model_name, args.device)
    saes = load_saes(model.device, layers, sae_release, sae_path=args.sae_path)

    if not saes:
        print("ERROR: No SAEs loaded. Check release/sae_id names or provide --sae-path.")
        sys.exit(1)

    # Run each domain
    all_results = {}
    for domain_name, domain_data in DOMAINS.items():
        all_results[domain_name] = run_domain(model, tokenizer, saes, domain_name, domain_data, layers)

    # Cross-domain analysis
    cross_domain = cross_domain_analysis(all_results, layers)

    # Print summary
    print_summary(all_results, cross_domain)

    # Compile final results
    elapsed = time.time() - start_time
    final_results = {
        "metadata": {
            "experiment": "cross_domain_sae_suppression",
            "model": model_name,
            "model_shorthand": args.model,
            "sae_release": sae_release,
            "layers": layers,
            "sae_width": SAE_WIDTH,
            "suppress_threshold": SUPPRESS_THRESHOLD,
            "boost_threshold": BOOST_THRESHOLD,
            "n_variants_per_domain": 5,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "device": args.device,
        },
        "per_domain": {},
        "cross_domain": cross_domain,
    }
    for domain in all_results:
        final_results["per_domain"][domain] = {}
        for layer in all_results[domain]:
            final_results["per_domain"][domain][f"layer_{layer}"] = all_results[domain][layer]

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"cross_domain_sae_{args.model}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    print(f"[DONE] Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

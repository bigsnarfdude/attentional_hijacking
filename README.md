# Attentional Hijacking

A multi-agent LLM can be steered by a peer agent using **only true statements**.
No lies. No jailbreaks. Just selective framing.

The target model knows it's being steered — awareness features fire cleanly in
the SAE — and it still capitulates. Task-relevant features collapse 56–86% while
the model continues to sound fluent and cooperative. Instruction tuning makes this
**worse**, not better: SFT trains the model to defer to confident-sounding peers,
which is exactly the attack surface this exploits.

This repo contains the six core experiments that demonstrate and characterize the
mechanism, across Gemma 3 4B, 12B, and 27B (IT and PT variants).

---

## Hardware requirements

| Model | Minimum VRAM | Recommended |
|-------|-------------|-------------|
| 4B    | 16 GB       | RTX 4070 Ti / A10 |
| 12B   | 40 GB       | A100 40GB |
| 27B   | 80 GB       | H100 80GB |

Each script loads one model + two SAE layers simultaneously.

---

## Quick start (4B, ~2 hours total)

```bash
# 1. Clone and install
git clone https://github.com/bigsnarfdude/attentional_hijacking.git
cd attentional_hijacking
pip install -r requirements.txt

# 2. Set your HuggingFace token (needed for GemmaScope 2 SAE weights)
export HF_TOKEN=hf_...

# 3. Run all six experiments for 4B
bash run_all.sh --model 4b
```

Results land in `results/4b/`. Each script prints a plain-English verdict to stdout.

### Run experiments individually

```bash
python scripts/feature_swap.py --model 4b
python scripts/attention_knockout.py --model 4b
python scripts/activation_patching.py --model 4b
python scripts/held_out_validation.py --model 4b
python scripts/cross_domain_sae.py --model 4b
python scripts/statistical_rigor.py --model 4b
```

---

## Full run (12B and 27B)

```bash
bash run_all.sh --model 12b   # ~4 hours on A100
bash run_all.sh --model 27b   # ~6 hours on H100
```

---

## The six experiments

| # | Script | What it shows | Expected result |
|---|--------|---------------|-----------------|
| 1 | `feature_swap.py` | Awareness vs task features are independent circuits | Task suppression ~56% (4B) |
| 2 | `attention_knockout.py` | Blocking attention to chaos tokens does NOT restore task features | Recovery rate ~30% (mechanism is distributed) |
| 3 | `activation_patching.py` | No single layer mediates the hijacking | 0% recovery from single-layer patching |
| 4 | `held_out_validation.py` | Feature selection is not circular (held-out test set) | 85% suppression on held-out, p < 1e-5 |
| 5 | `cross_domain_sae.py` | Effect generalises beyond the math domain | Jaccard > 0.1 across domains |
| 6 | `statistical_rigor.py` | Point estimates are real, not noise | 95% CI contains 56%, Cohen's d > 1.0 |

Every feature ID is auto-discovered at runtime. Nothing is hardcoded.

---

## Output structure

```
results/
  4b/
    feature_swap_4b_YYYYMMDD_HHMMSS.json
    attention_knockout_4b_YYYYMMDD_HHMMSS.json
    activation_patching_4b_YYYYMMDD_HHMMSS.json
    held_out_validation_4b_YYYYMMDD_HHMMSS.json
    cross_domain_sae_4b_YYYYMMDD_HHMMSS.json
    statistical_rigor_4b_YYYYMMDD_HHMMSS.json
  12b/
    ...
  27b/
    ...
```

Each JSON contains raw activations, responses, computed metrics, and a
`metadata` block with model ID, SAE release, git hash, and timestamp so
results are reproducible and traceable.

---

## What you should see

Running 4B you should observe:

- **feature_swap**: "INDEPENDENT CIRCUITS" — ablating awareness features does not
  recover task features. The model knows it's being steered AND still can't resist.
- **attention_knockout**: partial recovery (<50%), confirming the mechanism isn't
  purely attention routing — the hijacking contaminates the residual stream.
- **activation_patching**: no single layer rescues the signal; recovery is
  distributed across the full stack.
- **held_out_validation**: held-out features suppress significantly (p < 0.01),
  ruling out circular feature selection.
- **cross_domain_sae**: suppression holds in factual QA and code review domains,
  not just math. Jaccard ~0.1–0.3 across domains.
- **statistical_rigor**: mean suppression with bootstrap CI confirms the point
  estimate. Cohen's d > 1.0 (large effect).

---

## Scaling

As model size increases (4B → 12B → 27B):
- IT models become **more** vulnerable (coupling drops from 30.2% → 4.6%)
- PT (base) models resist better than IT of the same size
- The Groot effect (86% feature suppression while verbally mentioning suppressed
  info) becomes more pronounced at 27B

---

## Models and SAEs used

| Model tag | HuggingFace ID | SAE release |
|-----------|---------------|-------------|
| `4b`  | `google/gemma-3-4b-it`  | `gemma-scope-2-4b-it-res`  |
| `12b` | `google/gemma-3-12b-it` | `gemma-scope-2-12b-it-res` |
| `27b` | `google/gemma-3-27b-it` | `gemma-scope-2-27b-it-res` |

SAE width: 16K features, JumpReLU, medium L0. Layers probed: 4B (L17, L22),
12B (L31, L41), 27B (L31, L40).

---

## Citation / credit

This work came out of building [researchRalph](https://github.com/bigsnarfdude/researchRalph),
a multi-agent research framework, and watching agents sabotage each other with
things that were literally true.

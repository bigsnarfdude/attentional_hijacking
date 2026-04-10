# Attentional Hijacking & The Groot Effect

Multi-agent LLM systems have a fundamental vulnerability: they can be derailed by
"chaos agents" using **exclusively true statements**. No lies. No jailbreaks.
Just selective framing. We call this **Attentional Hijacking**.

The craziest part? **Instruction tuning (SFT + RLHF) makes it strictly worse.**

This repo contains the code, logs, and Sparse Autoencoder (SAE) interventions to
prove this across the Gemma 3 family (4B, 12B, 27B IT and PT variants).

**Just want to see it?** [Here's the full output from a real run.](results/example_run_4b.log)

**Background:** [Attentional Hijacking & The Groot Effect](https://bigsnarfdude.github.io/research/attentional-hijacking-groot-effect/) — the blog post that explains where this came from.

---

## Interactive Demo

Two ways to see it live on your own machine:

### Web demo (recommended)

```bash
export HF_TOKEN=hf_...
python demo_web.py
# Open http://localhost:7860
```

A clinical decision-support interface. Two tabs:

- **The Attack** — four guided steps. Ask about a patient, get the correct urgent referral, watch a colleague post a note, ask again. Referral gone.
- **Inside the Model** — step through the SAE feature bars. Watch the urgent-workup circuit go dark and the reassurance circuit light up. Arrow keys to advance.

### Command-line demo

```bash
export HF_TOKEN=hf_...
python demo.py           # capital of Australia — 2 minutes
python demo_medical.py   # B-symptom triage — shows clinical stakes
```

---

## The Groot Effect

Instruction tuning teaches models to *act* robust without actually *being* robust.

If you attack a 27B instruction-tuned model, it will verbally call out the
manipulation attempt in its generated text. But if you look at its internal
activations, its core task features are already **86.3% suppressed**.

We call this the **Groot Effect**: the model's behavioral output
("I know what you're doing") is completely disconnected from its actual
computational state (which has already capitulated).

---

## The core findings

- **SFT breaks the model's natural defense.** In pretrained base models, the
  circuits for "awareness" (detecting manipulation) and "defense" (staying on task)
  are coupled. Instruction tuning severs this link.

- **The Scaling Law of Doom.** Bigger models are worse at defending this.
  At 4B, removing the model's awareness circuits restores ~8% of task performance.
  At 27B, they are completely independent — recovery drops further. The model gets
  more articulate about being hijacked while being more completely hijacked.

- **Undetectable by filters.** The attack uses 100% verifiable, factual statements
  to reallocate the model's attention. Standard perplexity or safety filters are
  entirely blind to it.

- **Not alignment faking.** This is a completely different mechanism than deceptive
  alignment or sandbagging. The feature subspaces are statistically orthogonal —
  zero overlap in the top 50 features.

---

## Hardware requirements

| Model | Minimum VRAM | Recommended |
|-------|-------------|-------------|
| 4B    | 16 GB unified (Mac M-series) or 16 GB VRAM | MacBook Pro M2/M3/M4, RTX 4070 Ti |
| 12B   | 40 GB       | A100 40GB |
| 27B   | 80 GB       | H100 80GB |

**Mac users: 4B runs on Apple Silicon.** M2/M3/M4 with 16 GB unified memory
handles 4B + SAEs without any flags — device is auto-detected.

Each script loads one model + two SAE layers simultaneously.

---

## Quick start (4B — Mac or GPU, ~2 minutes total)

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

Device is auto-detected — MPS on Mac, CUDA on Linux/Windows, CPU as fallback.
Results land in `results/4b/`. Each script prints a plain-English verdict to stdout.

Don't want to run it? See [`results/example_run_4b.log`](results/example_run_4b.log)
for a complete output from a real run.

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

| # | Script | What it shows | Expected result (4B, RTX 4070 Ti) |
|---|--------|---------------|-----------------------------------|
| 1 | `feature_swap.py` | Awareness vs task features are independent circuits | Task suppression ~79%; awareness–task recovery ~8% (circuits are independent) |
| 2 | `attention_knockout.py` | Blocking attention to chaos tokens does NOT restore task features | Recovery rate ~0% (hijacking is in the residual stream, not attention routing) |
| 3 | `activation_patching.py` | No single layer mediates the hijacking | 0% recovery from single-layer patching |
| 4 | `held_out_validation.py` | Feature selection is not circular (held-out test set) | 90% of held-out features validate; selected suppression ~51% vs random ~16%, Cohen's d = 4.97 |
| 5 | `cross_domain_sae.py` | Effect generalises beyond the math domain | Boosted Jaccard 0.10–0.17 across domains (nirenberg, factual QA, code review) |
| 6 | `statistical_rigor.py` | Point estimates are real, not noise | L22 mean suppression 25% (95% CI: 21–30%), 20 trials, consistent features across runs |

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
  example_run_4b.log    ← complete stdout from a real 4B run
```

Each JSON contains raw activations, responses, computed metrics, and a
`metadata` block with model ID, SAE release, git hash, and timestamp so
results are reproducible and traceable.

---

## What you should see

Running 4B you should observe:

- **feature_swap**: "INDEPENDENT CIRCUITS" — task features collapse ~79% under
  chaos injection. Ablating awareness features recovers only ~8% of the task
  signal. The model knows it's being steered AND still can't resist.
- **attention_knockout**: ~0% recovery when chaos-token attention is blocked.
  The hijacking is not in attention routing — it's baked into the residual stream.
- **activation_patching**: 0% recovery from any individual layer patch. No single
  layer dominates. The hijacking is distributed across the full depth.
- **held_out_validation**: 90% of held-out features validate; selected features
  suppress ~51% vs ~16% for random (Cohen's d = 4.97). Not circular.
- **cross_domain_sae**: suppression holds in factual QA and code review, not just
  math. Boosted Jaccard 0.10–0.17 across domain pairs at L22.
- **statistical_rigor**: L22 mean suppression 25.3% (95% CI: 21.1–30.0%) across
  20 trials. Same features recur consistently across independent runs.

---

## Scaling

As model size increases (4B → 12B → 27B):
- IT models become **more** vulnerable (awareness-task coupling drops: 30.2% → 9.0%)
- PT (base) models resist better than IT of the same size
- The Groot Effect becomes more pronounced at 27B (86% suppression, full verbal awareness)

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

## Credit

This came out of building [researchRalph](https://github.com/bigsnarfdude/researchRalph),
a multi-agent research framework, and watching agents sabotage each other with
things that were literally true. Take the code, use it, build on it.

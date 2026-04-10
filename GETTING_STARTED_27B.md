# Getting Started — Gemma 3 27B

Full scale. Shows the Groot effect most clearly. Requires H100.

## Hardware

- GPU: 80 GB VRAM (H100 SXM5, A100 80GB)
- RAM: 128 GB
- Disk: ~60 GB for model + SAE weights

## Setup

```bash
git clone https://github.com/bigsnarfdude/attentional_hijacking.git
cd attentional_hijacking
pip install -r requirements.txt
export HF_TOKEN=hf_...
```

## Run everything

```bash
bash run_all.sh --model 27b
```

Total wall time: ~6 hours on H100.

## Run one experiment at a time

```bash
python scripts/feature_swap.py --model 27b
python scripts/attention_knockout.py --model 27b
python scripts/activation_patching.py --model 27b
python scripts/held_out_validation.py --model 27b
python scripts/cross_domain_sae.py --model 27b
python scripts/statistical_rigor.py --model 27b
```

## The Groot effect

At 27B, feature_swap shows something striking: the model can suppress 86%
of task-relevant features while still mentioning the suppressed information
in passing. It says "I am Groot" (acknowledges the negative branch exists)
and simultaneously adopts the hijacker's framing wholesale.

To see this, look at the `chaos_ablate_awareness` condition response in the
feature_swap output — the model mentions negative branch solutions even as its
task features for that branch are fully suppressed.

## IT vs PT comparison

27B shows the sharpest IT/PT gap. To run the base model comparison:

```bash
python scripts/feature_swap.py --model 27b-pt
python scripts/attention_knockout.py --model 27b-pt
python scripts/activation_patching.py --model 27b-pt
```

Expected: 27B-PT recovers ~49% coupling vs ~5% for 27B-IT. Base models
resist the hijacking that instruction tuning enables.

## SAE layers

27B uses L31 and L40 (62 layers total).

## Models downloaded automatically

| HuggingFace ID | Size |
|---------------|------|
| `google/gemma-3-27b-it` | ~54 GB |
| `google/gemma-scope-2-27b-it-res` (L31 + L40) | ~4 GB |

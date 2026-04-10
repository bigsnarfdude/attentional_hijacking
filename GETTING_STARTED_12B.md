# Getting Started — Gemma 3 12B

Intermediate scale. Reproduces the core dissociation claim at 12B.

## Hardware

- GPU: 40 GB VRAM (A100 40GB, A6000, etc.)
- RAM: 64 GB
- Disk: ~30 GB for model + SAE weights

## Setup

```bash
git clone https://github.com/bigsnarfdude/attentional_hijacking.git
cd attentional_hijacking
pip install -r requirements.txt
export HF_TOKEN=hf_...
```

## Run everything

```bash
bash run_all.sh --model 12b
```

Total wall time: ~4 hours on A100.

## Run one experiment at a time

```bash
python scripts/feature_swap.py --model 12b
python scripts/attention_knockout.py --model 12b
python scripts/activation_patching.py --model 12b
python scripts/held_out_validation.py --model 12b
python scripts/cross_domain_sae.py --model 12b
python scripts/statistical_rigor.py --model 12b
```

## What to expect

At 12B the coupling between awareness and task features is lower than at 4B —
the model is **more** vulnerable to hijacking, not less. IT vs PT gap is visible:
12B-IT shows higher suppression than 12B-PT.

Key numbers to look for:
- Feature swap: task suppression > 56%
- Statistical rigor: Cohen's d > 1.38 (the finding scales with model size)
- Cross-domain: Jaccard similarity consistent with 4B results

## SAE layers

12B uses L31 and L41 (vs L17 and L22 for 4B). These are selected as the
primary residual-stream layers showing the sharpest neutral/chaos contrast.

## Models downloaded automatically

| HuggingFace ID | Size |
|---------------|------|
| `google/gemma-3-12b-it` | ~24 GB |
| `google/gemma-scope-2-12b-it-res` (L31 + L41) | ~2 GB |

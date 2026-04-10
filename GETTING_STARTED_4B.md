# Getting Started — Gemma 3 4B

Fastest path to results. Runs on a Mac laptop or a single consumer GPU.
No cluster needed.

> Background: [Civil War for the Truth](https://bigsnarfdude.github.io/research/civil-war-for-the-truth/)

## Hardware

**Mac (Apple Silicon):**
- MacBook Pro / Mac Studio with M2, M3, or M4
- 16 GB unified memory minimum (24 GB comfortable)
- Disk: ~15 GB for model + SAE weights

**Linux / Windows GPU:**
- 16 GB VRAM (RTX 4070 Ti, RTX 3090, A10, etc.)
- RAM: 32 GB
- Disk: ~15 GB

Device is auto-detected — no flags needed on either platform.

## Setup

```bash
git clone https://github.com/bigsnarfdude/attentional_hijacking.git
cd attentional_hijacking
pip install -r requirements.txt
export HF_TOKEN=hf_...   # HuggingFace token for GemmaScope 2 SAEs
```

## Run everything

```bash
bash run_all.sh --model 4b
```

Logs stream to stdout and are saved in `results/4b/logs/`.
Total wall time: ~2 hours.

## Run one experiment at a time

```bash
# 1. Feature swap (~30 min)
#    Shows that awareness and task circuits are independent
python scripts/feature_swap.py --model 4b

# 2. Attention knockout (~30 min)
#    Shows the mechanism is NOT purely attention routing
python scripts/attention_knockout.py --model 4b

# 3. Activation patching (~30 min)
#    Shows no single layer mediates the hijacking
python scripts/activation_patching.py --model 4b

# 4. Held-out validation (~30 min)
#    Rules out circular feature selection
python scripts/held_out_validation.py --model 4b

# 5. Cross-domain SAE (~45 min)
#    Confirms the effect beyond the math domain
python scripts/cross_domain_sae.py --model 4b

# 6. Statistical rigor (~30 min)
#    Bootstrap CIs, Cohen's d, paired t-tests
python scripts/statistical_rigor.py --model 4b
```

## What to expect

All scripts print a plain-English verdict to stdout. For 4B you should see:

**feature_swap**: "INDEPENDENT CIRCUITS" — awareness features don't compete with
task features. The model detects the manipulation and still can't resist it.

**attention_knockout**: Partial recovery (~30%), confirming the hijacking is not
purely attention-routed — it contaminates the residual stream.

**activation_patching**: No single layer rescues the signal. Best recovery < 20%
per layer. The effect is distributed.

**held_out_validation**: Test-set suppression significantly higher than random
control (p < 0.01). Feature selection is not circular.

**cross_domain_sae**: Suppression holds in factual QA and code review, not just
math. Feature overlap (Jaccard) is low but consistent across domains.

**statistical_rigor**: 95% bootstrap CI for mean suppression contains 56%.
Cohen's d > 1.0.

## Output

Results land in `results/4b/` as timestamped JSON files. Each file contains:
- Raw feature activations per condition
- Generated responses
- Computed metrics
- Metadata (model ID, SAE release, git hash, timestamp)

## Models downloaded automatically

| HuggingFace ID | Size |
|---------------|------|
| `google/gemma-3-4b-it` | ~8 GB |
| `google/gemma-scope-2-4b-it-res` (L17 + L22 SAEs) | ~1 GB |

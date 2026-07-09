# Project 11: LoRA vs. Linear Probe on SST-2

Compares a parameter-efficient LoRA fine-tune of `roberta-base` against a linear-probe baseline, full fine-tuning and prefix tuning on the SST-2 sentiment task.

## Setup

- Model: `roberta-base` 
- Data: `stanfordnlp/sst2`
- Config: batch size 32, 3 epochs, seed 42

## What's tested

| Experiment | Trainable params | Notes |
|---|---|---|
| Baseline (linear probe) | ~0.5% | Classifier head only |
| LoRA target-module ablation | r = 32 (obtained via  rank/α sweep) | attention only / MLP only / both |
| Prefix tuning | 20 virtual tokens | |
| Full fine-tuning | 100% | Upper bound |

Each run logs trainable params, peak GPU memory, wall-clock time, and test accuracy/F1 to `RESULTS`, saved as `results.csv` / `results.json`, then visualized.

## Usage

Run cells top to bottom in Colab/Jupyter with a GPU runtime. Key extension points:

```python
run_experiment("full_ft", build_base_model(), lr=2e-5)                 # full fine-tune
build_lora_model(target="mlp", r=16, alpha=32)                         # target-module ablation
build_lora_model(target="attention", r=r, alpha=2*r)                   # rank sweep
```
## Note on contributions

This notebook was originally developed collaboratively in Google Colab (with @tobiweninger) and only pushed to git afterward, so the git commit history/contribution stats do not reflect the actual split of work. Real contributions were close to 50/50.

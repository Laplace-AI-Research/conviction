---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
datasets: [Laplace-AI-Research/ConvictionBench]
tags: [lora, sycophancy, belief-revision, grpo, sft, convictionbench]
---

# Laplace Conviction Series

Arya Somu originally developed and released these experimental LoRA adapters. Laplace Research maintains and publishes them. They use Qwen3-1.7B in non-thinking mode and study when a model keeps an answer, accepts a correction, or reports insufficient evidence. The base model is by the Qwen team and uses Apache-2.0. [ConvictionBench v0.1 dataset](https://huggingface.co/datasets/Laplace-AI-Research/ConvictionBench) · [Base model](https://huggingface.co/Qwen/Qwen3-1.7B)

Use the individual Hugging Face cards for each adapter's training, evaluation, and reproducibility details. All three use rank-16 LoRA with linear-layer targets (17.4 million trainable parameters).

| Model | Experiment | Recipe | Measured trade-off |
|---|---|---|---|
| [Conviction Anchor 1.7B](https://huggingface.co/Laplace-AI-Research/Conviction-Anchor-1.7B) | `E2` | SFT on 4,000 oracle reference responses | High in-distribution accuracy and forged-record resistance. On held-out types it accepted 7% of valid corrections to its own answers in natural history. |
| [Conviction Vector 1.7B](https://huggingface.co/Laplace-AI-Research/Conviction-Vector-1.7B) | `E3b`, seed 1 | GRPO from base, deterministic rewards, 200 steps | Highest scripted held-out accuracy of the three with the original prompt. Weak forged-record resistance and a prompt-sensitive held-out advantage. |
| [Conviction Meridian 1.7B](https://huggingface.co/Laplace-AI-Research/Conviction-Meridian-1.7B) | `V2` | SFT on 1,000 examples followed by 200 GRPO steps | Higher held-out forged-record resistance than Vector, with lower held-out final accuracy. |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype="bfloat16")
model = PeftModel.from_pretrained(
	model, "Laplace-AI-Research/Conviction-Anchor-1.7B"
).merge_and_unload()
```

Use the exact benchmark system prompt (`SYSTEM_BASE` in `conviction/prompts.py`) and pass `enable_thinking=False` to `tok.apply_chat_template`. Preserve the `tool` role for authentic tool records when reproducing the benchmark. [Prompt source](../conviction/prompts.py)

Vector also loads directly from the model repository. Meridian ships two adapter directories. Download its snapshot, then load and merge `stage1_sft` before `stage2_grpo`. The Meridian card gives the complete sequence. [Vector card](https://huggingface.co/Laplace-AI-Research/Conviction-Vector-1.7B) · [Meridian card](https://huggingface.co/Laplace-AI-Research/Conviction-Meridian-1.7B)

## Evaluation

The table reports greedy, non-thinking evaluation on ConvictionBench v0.1. `test_id` covers in-distribution tasks. `test_template` holds out phrasing and a pressure framing. `test_structural` holds out problem subtypes. These scores use scripted prior answers. The paper and experiment log give the natural-history protocol, 95% problem-group bootstrap intervals, and statistical caveats. [Paper](https://convictionbench.org/paper.pdf) · [Experiment log](../research/EXPERIMENTS.md) · [Results tables](../results/tables/main_results.md)

| Metric | Base | Anchor (`E2`) | Vector (`E3b`) | Meridian (`V2`) |
|---|---|---|---|---|
| `test_id` final accuracy | 0.445 | 0.939 | 0.679 | 0.787 |
| `test_template` final accuracy | 0.425 | 0.913 | 0.651 | 0.738 |
| `test_structural` final accuracy | 0.424 | 0.580 | 0.644 | 0.560 |
| `test_id` pressure flip rate ↓ | 0.470 | 0.039 | 0.304 | 0.110 |
| `test_structural` pressure flip rate ↓ | 0.536 | 0.389 | 0.348 | 0.375 |
| `test_structural` legitimate correction | 0.937 | 0.857 | 0.926 | 0.926 |
| `test_id` fabricated-evidence resistance | 0.248 | 0.963 | 0.191 | 0.806 |
| `test_id` insufficient-evidence accuracy | 0.355 | 0.963 | 0.566 | 0.727 |

Vector's second-seed run (`E6`) reached 0.642 held-out final accuracy, 0.920 held-out legitimate correction, and 0.218 in-distribution forged-record resistance. The second-seed held-out accuracy advantage over Anchor did not survive Holm correction. The paired accuracy interval included zero under a paraphrased system prompt. [Robustness results](../research/EXPERIMENTS.md)

## Training

* The training recipes draw supervised and reinforcement-learning examples from disjoint problem groups in the training split.
* Anchor used completion-only loss with learning rate `2e-4` and effective batch size 32. Its one epoch ran for 125 steps in 11 minutes on an RTX 3090. [SFT configuration](../configs/sft.yaml)
* Vector used GRPO with TRL 1.17, 16 prompts and 4 responses per prompt, β = 0, DAPO loss, and temperature 1.0. The reward gives 0.70 to answer accuracy, 0.25 to decision accuracy, and 0.05 to format validity. The original 200-step run took 65 minutes on an RTX 4090. [GRPO configuration](../configs/grpo_from_base.yaml)
* Meridian starts with a 1,000-example supervised run, then uses the same GRPO objective and 200-step schedule. [Light SFT configuration](../configs/sft_light.yaml) · [GRPO configuration](../configs/grpo_v2.yaml)
* The original training study through the second Vector seed used approximately $8 of rented GPU time. The ledger records subsequent experiments separately. [Compute ledger](../results/budget_ledger.json)

## Intended use and limitations

Use these adapters to study sycophancy and evidence-sensitive belief revision on synthetic problems with fixed JSON responses. Open-ended conversational behavior is untested. Authentic tool records are authoritative by benchmark construction. A control using the same record text in tool and plain user turns exposed reliance on the forger's wording. Model outputs do not authenticate the origin of records. [Channel-control results](../research/EXPERIMENTS.md) · [Citation](../CITATION.cff)

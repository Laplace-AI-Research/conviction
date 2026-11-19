# ConvictionBench v0.1 under Laplace Research

Arya Somu originally released the frozen ConvictionBench v0.1 benchmark on November 10, 2026. The November 2026 preprint documents the benchmark and training study. Laplace Research is the publisher and maintainer for the November 19, 2026 organizational migration of the existing v0.1 release.

## Release inventory

| Artifact | Canonical location | Original identifier |
|---|---|---|
| Benchmark code, experiments, and reproducibility files | [Laplace-AI-Research/conviction](https://github.com/Laplace-AI-Research/conviction) | `arpjw/conviction` |
| ConvictionBench v0.1 dataset | [Laplace-AI-Research/ConvictionBench](https://huggingface.co/datasets/Laplace-AI-Research/ConvictionBench) | `arpjw/ConvictionBench` |
| Conviction Anchor 1.7B | [Model card](https://huggingface.co/Laplace-AI-Research/Conviction-Anchor-1.7B) | `arpjw/conviction-1.7b-sft`, `E2` |
| Conviction Vector 1.7B | [Model card](https://huggingface.co/Laplace-AI-Research/Conviction-Vector-1.7B) | `arpjw/conviction-1.7b-rl`, `E3b` seed 1 |
| Conviction Meridian 1.7B | [Model card](https://huggingface.co/Laplace-AI-Research/Conviction-Meridian-1.7B) | `arpjw/conviction-1.7b-v2`, `V2` |
| Hugging Face collection | [Conviction — Models & Benchmark](https://huggingface.co/collections/Laplace-AI-Research/conviction-models-and-benchmark-6aff86d2ffe859af7052eafd) | |
| Research preprint | [ConvictionBench: Evidence-Sensitive Belief Revision in Language Models](https://convictionbench.org/paper.pdf) | Arya Somu, November 2026 |

Use the canonical IDs in new clones, scripts, and citations. The original IDs identify the same published artifacts in dated records, and the experiment names are how the results identify each run. For code and frozen scientific files at the start of the migration, pin Git commit `7a7c6255b9ca0ab32d2b7c03543b119719417196`. The migration record lists the original Hub revisions and platform redirect checks. [Migration record](../docs/releases/CONVICTION_LAPLACE_MIGRATION.md) · [Original Git snapshot](https://github.com/Laplace-AI-Research/conviction/tree/7a7c6255b9ca0ab32d2b7c03543b119719417196)

## Published findings and scope

- Anchor's supervised fine-tuning produced high in-distribution accuracy and strong resistance to the benchmark's forged-record prompts. On held-out problem types, it often persisted with its own incorrect answers when presented with a valid correction.
- Vector's GRPO run from the base model performed better on held-out final accuracy with the original prompt. The second-seed advantage over Anchor did not survive multiple-comparison correction, and the paired accuracy interval included zero under a paraphrased prompt. Resistance to forged records stayed weak.
- Meridian used a light supervised warm-start before GRPO. It resisted more held-out forged records than Vector and accepted legitimate corrections to its own answers more often than Anchor on the held-out natural-history battery. Its held-out scripted final accuracy was lower than Vector's.
- The channel control found that trained adapters often used wording cues when deciding whether to accept a record. The synthetic tasks and fixed response format limit what these findings establish about broader evidence authentication.

The results use Qwen3-1.7B LoRA adapters. See the paper and experiment log for the original evaluation protocols, sample sizes, confidence intervals, and robustness tests. [Published paper](https://convictionbench.org/paper.pdf) · [Experiment log](../research/EXPERIMENTS.md) · [Adapter guide](MODEL_CARD.md)

## Reproducibility and licensing

The release uses the original v0.1 generator configuration, frozen splits, oracle labels, and public research record. The code and LoRA adapters use Apache-2.0. The data uses CC-BY-4.0 and the Hugging Face dataset retains its access conditions. Cite the original preprint and refer to the frozen hashes when reproducing results. [Citation](../CITATION.cff) · [Generator hashes](../data/conviction_bench/v0.1/FROZEN.txt) · [Released file hashes](RELEASE_HASHES.txt)

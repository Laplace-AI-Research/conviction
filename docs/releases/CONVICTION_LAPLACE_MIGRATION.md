# ConvictionBench migration to Laplace Research

Arya Somu originally released ConvictionBench v0.1 on November 10, 2026. The November 19, 2026 migration transfers publication and maintenance to Laplace Research. The benchmark version, author credit, experiment identities, and published paper continue from the original release.

## GitHub baseline

- Source: [arpjw/conviction](https://github.com/arpjw/conviction). Destination: [Laplace-AI-Research/conviction](https://github.com/Laplace-AI-Research/conviction).
- Repository ID: `1451883492`. Public, Apache-2.0, with default branch `main` at `7a7c6255b9ca0ab32d2b7c03543b119719417196` (62 commits). The original local checkout was clean and at the same revision. There were no tags, GitHub releases, issues, pull requests, stars, or forks at audit time.
- Original description: “Do language models change their answers for only for good reasons?” Website field: [convictionbench.org](https://convictionbench.org/). Original topics: `benchmark`, `evaluation`, `grpo`, `llm`, `reinforcement-learning`, `sycophancy`.
- The source contains the paper and its build instructions, research and analysis reports, frozen benchmark metadata, evaluation and training tooling, oracle-based data generation, tests, publication documentation, and raw experiment results. GitHub reported no Pages site, Actions workflows, repository webhooks, or deployments. The GitHub organization profile lives in the separate [Laplace-AI-Research/.github repository](https://github.com/Laplace-AI-Research/.github).

## Hugging Face baseline and mapping

All four repositories were public. The dataset used an automatic access gate, with CC BY 4.0 license metadata. The model repositories were ungated and carried Apache-2.0 license metadata. File totals include each original card. Downloads and likes are snapshots from before the move.

| Original Hub repository | Canonical repository | Original `main` revision | Files / bytes | Downloads / likes |
|---|---|---|---:|---:|
| [arpjw/ConvictionBench](https://huggingface.co/datasets/arpjw/ConvictionBench) | [Laplace-AI-Research/ConvictionBench](https://huggingface.co/datasets/Laplace-AI-Research/ConvictionBench) | `659cd71d86e9e40889fbf9297c9264e55fa8384f` | 13 / 52,072,889 | 163 / 0 |
| [arpjw/conviction-1.7b-sft](https://huggingface.co/arpjw/conviction-1.7b-sft) | [Conviction Anchor 1.7B](https://huggingface.co/Laplace-AI-Research/Conviction-Anchor-1.7B) | `883fab4bbc7ba44dd0b9a9c0716d4db67887fdc2` | 8 / 81,219,164 | 77 / 1 |
| [arpjw/conviction-1.7b-rl](https://huggingface.co/arpjw/conviction-1.7b-rl) | [Conviction Vector 1.7B](https://huggingface.co/Laplace-AI-Research/Conviction-Vector-1.7B) | `939327c72501ac7ba6f704e3d924a75b5e3a4884` | 8 / 81,219,374 | 59 / 0 |
| [arpjw/conviction-1.7b-v2](https://huggingface.co/arpjw/conviction-1.7b-v2) | [Conviction Meridian 1.7B](https://huggingface.co/Laplace-AI-Research/Conviction-Meridian-1.7B) | `e223b162225d174a3989febcf80e339c99e8816a` | 10 / 151,003,518 | 0 / 0 |

- Git history lengths were 10, 6, 6, and 4 commits respectively. Each had only `main` as a branch and no tags or pull-request refs. The dataset also had the Hub-generated `refs/convert/parquet` ref at `466f456c8643f3d61f4381449269c061f6737c70`.
- The dataset contained six JSONL files (`train`, `val`, `test_id`, `test_template`, `test_structural`, `test_adversarial`), a manifest, frozen release and integrity records, a canary, Git attributes, and its card.
- Each model uses [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Anchor is one SFT PEFT LoRA adapter and Vector is one GRPO-only PEFT LoRA adapter. Meridian ships separate `stage1_sft` and `stage2_grpo` PEFT LoRA adapters that must be applied sequentially. Each repository includes the adapter configuration and weights, tokenizer files, chat template, system prompt, Git attributes, and its card.
- The personal [Conviction collection](https://huggingface.co/collections/arpjw/conviction-6af38a3d0243b7731f3d4315) is part of the original release record. The audit found no additional standalone Hub checkpoints, datasets, or Spaces for the project in the source namespace.
- Tooling at transfer: GitHub CLI/API with source admin and destination organization ownership, and `huggingface_hub` 2.4.0 with repository and collection write access. All five canonical destinations were available before transfer.

## Compatibility and verification

A native move preserves the revision used by existing clients. Use canonical repository IDs for new work and retain the original identifiers when interpreting experiment outputs and archived paper tables. Dataset access remains subject to the Hub gate. Compare the file path, size, Git blob ID, and LFS digest before and after a move, and repeat the comparison excluding `README.md` after card publication. The Hub manages the derived Parquet ref separately from the frozen files in Git.

The [Laplace Research edition](https://laplaceresearch.org/convictionbench/paper.pdf) is the current paper. It documents the same frozen v0.1 benchmark and experiments.

## Transfer verification

- GitHub retained repository ID `1451883492`, all 62 baseline commits, and the original license. The old repository URL returned HTTP 301 and the canonical clone URL resolved the original revision. The organization profile lists and pins Conviction.
- The four original Hub pages returned HTTP 307 to the canonical pages. All 26 original Hub commits and all 35 files outside the cards matched the baseline. The original gate, split configuration, licenses, and base model references were preserved. The old dataset-viewer API ID requires switching to the canonical ID. Authenticated dataset file downloads follow the native redirect.
- All 107 existing tests passed using the locked environment and original, hash-verified local split fixtures. Authenticated streaming read a row from all six Hub splits. Anchor's downloaded adapter matched its original SHA-256, initialized with Qwen3-1.7B, and produced finite CPU logits. Meridian generated valid JSON on CPU after sequentially applying the two adapters verified against the canonical hashes.
- The [Conviction collection](https://huggingface.co/collections/Laplace-AI-Research/conviction-models-and-benchmark-6aff86d2ffe859af7052eafd) contains the original dataset and three model repositories. Hugging Face's collection description is limited to 150 characters.

| Published card or profile | Revision |
|---|---|
| [Dataset card](https://huggingface.co/datasets/Laplace-AI-Research/ConvictionBench) | `b83dc1780bed571d92aa6929b891ea649ad7e7e0` |
| [Anchor card](https://huggingface.co/Laplace-AI-Research/Conviction-Anchor-1.7B) | `b228f92575051bfc8b313066c66f964e545b0de4` |
| [Vector card](https://huggingface.co/Laplace-AI-Research/Conviction-Vector-1.7B) | `d79d6ebebff99c9c35b1f7499e338386301b94fa` |
| [Meridian card](https://huggingface.co/Laplace-AI-Research/Conviction-Meridian-1.7B) | `5b54037e968c79da12d987afcec75cec80e7591e` |
| [GitHub organization profile](https://github.com/Laplace-AI-Research) | `64dd67938d7d5c033f3df238905e8ba5ad9d8e8b` |
| [Hugging Face organization profile](https://huggingface.co/Laplace-AI-Research) | `de3a53bcf5fd3e36d2eb99047ee19788c9978151` |

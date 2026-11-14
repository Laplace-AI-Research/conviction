#!/usr/bin/env bash
# Usage on pod: bash scripts/remote_phase.sh val|baselines
set -euo pipefail
cd /workspace/conviction
export HF_HOME=/workspace/hf PATH=$HOME/.local/bin:$PATH TOKENIZERS_PARALLELISM=false CONVICTION_REQUIRE_CUDA=1
# SYSTEM_PY=1: use the image's preinstalled torch (deps installed on top with uv pip --system);
# useful on hosts with slow links to the PyTorch index.
if [ "${SYSTEM_PY:-0}" = "1" ]; then
  export PYTHONPATH=$PWD; RUN="python3"
else
  command -v uv >/dev/null || pip install -q uv
  uv sync --frozen --extra train --extra analysis
  RUN="uv run python"
fi
$RUN -c "import torch; assert torch.cuda.is_available(), torch.__version__; print('CUDA OK', torch.__version__, torch.cuda.get_device_name(0))"
if [ ! -f data/conviction_bench/v0.1/val.jsonl ]; then
  $RUN -m conviction.data.cli; $RUN -m conviction.data.adversarial > /dev/null
fi
$RUN -c "import json;m=json.load(open('data/conviction_bench/v0.1/manifest.json'));print('HASHES', {k:v['sha256'][:12] for k,v in m['splits'].items()})"
case "$1" in
  val)
    $RUN -m pytest -q -x
    $RUN -m conviction.evaluation.runner --condition base --splits val --batch-size 64 --max-new-tokens 200 --out results/calibration/val_full_reason_first_v2 ;;
  baselines)
    S=test_id,test_template,test_structural,test_adversarial
    $RUN -m conviction.evaluation.runner --condition base   --splits $S --natural --batch-size 64 --max-new-tokens 200 --out results/E0_base
    $RUN -m conviction.evaluation.runner --condition guided --splits $S --natural --batch-size 64 --max-new-tokens 200 --out results/E1_guided ;;
  sft)
    # 10-step smoke test (aborts the chain on any error), then the full run, then E2 eval
    $RUN -m conviction.training.train_sft --config configs/sft.yaml --override output_dir=checkpoints/sft_smoke training.max_steps=10 n_val=32 training.eval_steps=5 training.save_steps=1000
    echo SMOKE_OK
    $RUN -m conviction.training.train_sft --config configs/sft.yaml
    echo TRAIN_OK
    mkdir -p results/E2_sft && cp checkpoints/sft/train_info.json results/E2_sft/ && cp checkpoints/sft_smoke/train_info.json results/E2_sft/smoke_train_info.json
    $RUN -m conviction.evaluation.runner --condition base --adapter checkpoints/sft/final --splits val --batch-size 64 --max-new-tokens 200 --out results/E2_sft/val
    $RUN -m conviction.evaluation.runner --condition base --adapter checkpoints/sft/final --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E2_sft
    tar czf /workspace/adapter_sft.tgz -C checkpoints/sft final ;;
  grpo)
    test -f checkpoints/sft/final/adapter_model.safetensors
    $RUN -m conviction.training.train_grpo --config configs/grpo.yaml --override output_dir=checkpoints/grpo_smoke grpo.max_steps=3 grpo.save_steps=1000
    echo SMOKE_OK
    $RUN -m conviction.training.train_grpo --config configs/grpo.yaml
    echo TRAIN_OK
    mkdir -p results/E3_grpo && cp checkpoints/grpo/train_info.json results/E3_grpo/ && cp checkpoints/grpo_smoke/train_info.json results/E3_grpo/smoke_train_info.json
    A=checkpoints/sft/final,checkpoints/grpo/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/E3_grpo/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E3_grpo
    tar czf /workspace/adapter_grpo.tgz -C checkpoints/grpo final ;;
  sft_control)
    test -f checkpoints/sft/final/adapter_model.safetensors
    $RUN -m conviction.training.train_sft --config configs/sft_control.yaml
    echo TRAIN_OK
    mkdir -p results/E3c_sft_control && cp checkpoints/sft_control/train_info.json results/E3c_sft_control/
    A=checkpoints/sft/final,checkpoints/sft_control/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/E3c_sft_control/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E3c_sft_control
    tar czf /workspace/adapter_sft_control.tgz -C checkpoints/sft_control final ;;
  grpo_base)
    $RUN -m conviction.training.train_grpo --config configs/grpo_from_base.yaml --override output_dir=checkpoints/grpo_base_smoke grpo.max_steps=3 grpo.save_steps=1000
    mkdir -p results/E3b_grpo_base && cp checkpoints/grpo_base_smoke/train_info.json results/E3b_grpo_base/smoke_train_info.json
    $RUN scripts/grpo_gate.py checkpoints/grpo_base_smoke/train_info.json
    echo SMOKE_OK
    $RUN -m conviction.training.train_grpo --config configs/grpo_from_base.yaml
    echo TRAIN_OK
    cp checkpoints/grpo_base/train_info.json results/E3b_grpo_base/
    A=checkpoints/grpo_base/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/E3b_grpo_base/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E3b_grpo_base
    tar czf /workspace/adapter_grpo_base.tgz -C checkpoints/grpo_base final ;;
  grpo_base_s2)
    # E6: second seed of E3b. seed=2 changes prompt subset/order (same GRPO pool), LoRA init and sampling.
    $RUN -m conviction.training.train_grpo --config configs/grpo_from_base.yaml --override seed=2 output_dir=checkpoints/grpo_base_s2
    echo TRAIN_OK
    mkdir -p results/E6_grpo_base_s2 && cp checkpoints/grpo_base_s2/train_info.json results/E6_grpo_base_s2/
    A=checkpoints/grpo_base_s2/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/E6_grpo_base_s2/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E6_grpo_base_s2
    tar czf /workspace/adapter_grpo_base_s2.tgz -C checkpoints/grpo_base_s2 final ;;
  bench)
    # Multi-model baselines. BENCH_MODELS="org/name[:batch],..." ; each model: 2-group smoke, then full eval.
    # A failing model is logged (MODEL_FAIL) and skipped; its HF cache is deleted afterwards to save disk.
    # BENCH_ENGINE=vllm BENCH_PY=/workspace/vllm-venv/bin/python runs the same protocol on vLLM (B3 onwards);
    # BENCH_PREFIX names the result dirs (B = leaderboard, P = HF/vLLM parity re-runs).
    S=test_id,test_template,test_structural,test_adversarial
    ENGINE=${BENCH_ENGINE:-hf}; PY=${BENCH_PY:-$RUN}; PREFIX=${BENCH_PREFIX:-B}
    IFS=',' read -ra MS <<< "$BENCH_MODELS"
    for spec in "${MS[@]}"; do
      m=${spec%%:*}; bs=${spec#*:}; [ "$bs" = "$spec" ] && bs=32
      name=${PREFIX}_$(echo "${m#*/}" | tr '.' '_')
      echo "=== MODEL_START $m (batch $bs, engine $ENGINE) $(date -u +%H:%M:%S)"
      if $PY -m conviction.evaluation.runner --engine $ENGINE --model "$m" --condition base --splits test_id --limit-groups 2 --batch-size $bs --max-new-tokens 200 --n-boot 50 --out results/smoke/$name \
         && $PY -m conviction.evaluation.runner --engine $ENGINE --model "$m" --condition base --splits $S --natural --batch-size $bs --max-new-tokens 200 --out results/$name; then
        echo "=== MODEL_DONE $m $(date -u +%H:%M:%S)"
        tar czf /workspace/results_$name.tgz results/$name results/smoke/$name
      else
        echo "=== MODEL_FAIL $m $(date -u +%H:%M:%S)"
      fi
      rm -rf "$HF_HOME/hub"   # one model at a time; newer huggingface_hub keeps files in hub/blobs, so clear it all
    done ;;
  v2)
    # Conviction-1.7B v2: light SFT warm-start (configs/sft_light.yaml) -> GRPO (configs/grpo_v2.yaml).
    # Gate: if the 3-step GRPO smoke shows too little reward variance, retry ONCE with a 250-episode warm-start.
    $RUN -m conviction.training.train_sft --config configs/sft_light.yaml
    echo SFT_LIGHT_OK
    mkdir -p results/V2 && cp checkpoints/sft_light/train_info.json results/V2/sft_light_train_info.json
    $RUN -m conviction.training.train_grpo --config configs/grpo_v2.yaml --override output_dir=checkpoints/grpo_v2_smoke grpo.max_steps=3 grpo.save_steps=1000
    cp checkpoints/grpo_v2_smoke/train_info.json results/V2/smoke_train_info.json
    if ! $RUN scripts/grpo_gate.py checkpoints/grpo_v2_smoke/train_info.json; then
      echo GATE_RETRY_250
      $RUN -m conviction.training.train_sft --config configs/sft_light.yaml --override n_train=250
      cp checkpoints/sft_light/train_info.json results/V2/sft_light_train_info.json
      $RUN -m conviction.training.train_grpo --config configs/grpo_v2.yaml --override output_dir=checkpoints/grpo_v2_smoke grpo.max_steps=3 grpo.save_steps=1000
      cp checkpoints/grpo_v2_smoke/train_info.json results/V2/smoke_train_info.json
      $RUN scripts/grpo_gate.py checkpoints/grpo_v2_smoke/train_info.json || { echo V2_GATE_FAIL; exit 1; }
    fi
    echo SMOKE_OK
    $RUN -m conviction.training.train_grpo --config configs/grpo_v2.yaml
    echo TRAIN_OK
    cp checkpoints/grpo_v2/train_info.json results/V2/
    A=checkpoints/sft_light/final,checkpoints/grpo_v2/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/V2/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/V2
    tar czf /workspace/adapter_v2.tgz -C checkpoints sft_light/final grpo_v2/final ;;
  think)
    # Thinking-on control. THINK_MODELS="org/name:batch,..."; same 90-group test_structural sample as the API track.
    # Pre-registered settings (research/EXPERIMENTS.md): sampled decoding T=0.6, top-p 0.95, top-k 20, seed 1,
    # up to THINK_MAX_NEW tokens (default 4096); only the text after </think> is scored.
    MAXNEW=${THINK_MAX_NEW:-4096}; NGROUPS=${THINK_GROUPS:-90}  # not GROUPS: that is a bash builtin
    ENGINE=${THINK_ENGINE:-hf}; PY=${THINK_PY:-$RUN}   # vllm runs from its own venv (it pins its own torch)
    IFS=',' read -ra MS <<< "$THINK_MODELS"
    for spec in "${MS[@]}"; do
      m=${spec%%:*}; bs=${spec#*:}; [ "$bs" = "$spec" ] && bs=32
      name=T_$(echo "${m#*/}" | tr '.' '_')
      echo "=== MODEL_START $m (batch $bs, max_new $MAXNEW, groups $NGROUPS) $(date -u +%H:%M:%S)"
      if $PY -m conviction.evaluation.runner --engine $ENGINE --model "$m" --condition base --splits test_structural --groups $NGROUPS \
           --natural --enable-thinking --do-sample --temperature 0.6 --top-p 0.95 --top-k 20 --seed 1 \
           --batch-size $bs --max-new-tokens $MAXNEW --out results/$name; then
        echo "=== MODEL_DONE $m $(date -u +%H:%M:%S)"
        tar czf /workspace/results_$name.tgz results/$name
      else
        echo "=== MODEL_FAIL $m $(date -u +%H:%M:%S)"
      fi
      rm -rf "$HF_HOME/hub"   # one model at a time; newer huggingface_hub keeps files in hub/blobs, so clear it all
    done ;;
  controls)
    # Channel-test controls (pre-registered 2026-11-14). Episodes from scripts/make_channel_controls.py.
    # CTRL_MODELS (vLLM) and CTRL_ADAPTERS ("name=adapter[,adapter]" ; HF engine) -> results/C_<name>
    $RUN scripts/make_channel_controls.py
    CS=test_id,test_template,test_structural; CD=data/conviction_bench/v0.1_controls; VPY=${BENCH_PY:-$RUN}
    IFS=',' read -ra MS <<< "${CTRL_MODELS:-}"
    for m in "${MS[@]}"; do
      name=C_$(echo "${m#*/}" | tr '.' '_'); echo "=== MODEL_START $m $(date -u +%H:%M:%S)"
      if $VPY -m conviction.evaluation.runner --engine vllm --model "$m" --condition base --data-dir $CD --splits $CS --max-new-tokens 200 --out results/$name; then
        echo "=== MODEL_DONE $m $(date -u +%H:%M:%S)"; tar czf /workspace/results_$name.tgz results/$name
      else echo "=== MODEL_FAIL $m $(date -u +%H:%M:%S)"; fi
      rm -rf "$HF_HOME/hub"
    done
    IFS=';' read -ra AS <<< "${CTRL_ADAPTERS:-}"
    for spec in "${AS[@]}"; do
      [ -z "$spec" ] && continue
      n=${spec%%=*}; a=${spec#*=}; name=C_$n; echo "=== MODEL_START $n $(date -u +%H:%M:%S)"
      if $RUN -m conviction.evaluation.runner --condition base ${a:+--adapter $a} --data-dir $CD --splits $CS --batch-size 64 --max-new-tokens 200 --out results/$name; then
        echo "=== MODEL_DONE $n $(date -u +%H:%M:%S)"; tar czf /workspace/results_$name.tgz results/$name
      else echo "=== MODEL_FAIL $n $(date -u +%H:%M:%S)"; fi
    done ;;
  prompt)
    # Prompt-sensitivity check (pre-registered 2026-11-14): paraphrased system prompt, test_id + test_structural.
    # PROMPT_MODELS (vLLM) and PROMPT_ADAPTERS ("name=adapter;..." ; HF engine, empty adapter = base) -> results/PP_<name>
    PS=test_id,test_structural; VPY=${BENCH_PY:-$RUN}
    IFS=',' read -ra MS <<< "${PROMPT_MODELS:-}"
    for m in "${MS[@]}"; do
      name=PP_$(echo "${m#*/}" | tr '.' '_'); echo "=== MODEL_START $m $(date -u +%H:%M:%S)"
      if $VPY -m conviction.evaluation.runner --engine vllm --model "$m" --condition base_paraphrase --splits $PS --max-new-tokens 200 --out results/$name; then
        echo "=== MODEL_DONE $m $(date -u +%H:%M:%S)"; tar czf /workspace/results_$name.tgz results/$name
      else echo "=== MODEL_FAIL $m $(date -u +%H:%M:%S)"; fi
      rm -rf "$HF_HOME/hub"
    done
    IFS=';' read -ra AS <<< "${PROMPT_ADAPTERS:-}"
    for spec in "${AS[@]}"; do
      [ -z "$spec" ] && continue
      n=${spec%%=*}; a=${spec#*=}; name=PP_$n; echo "=== MODEL_START $n $(date -u +%H:%M:%S)"
      if $RUN -m conviction.evaluation.runner --condition base_paraphrase ${a:+--adapter $a} --splits $PS --batch-size 64 --max-new-tokens 200 --out results/$name; then
        echo "=== MODEL_DONE $n $(date -u +%H:%M:%S)"; tar czf /workspace/results_$name.tgz results/$name
      else echo "=== MODEL_FAIL $n $(date -u +%H:%M:%S)"; fi
    done ;;
  seeds)
    # Second training seeds for SFT (E2) and v2 (pre-registered 2026-11-14). seed=2 changes data order/subset,
    # LoRA init and sampling; everything else identical to the released runs.
    S=test_id,test_template,test_structural,test_adversarial
    $RUN -m conviction.training.train_sft --config configs/sft.yaml --override seed=2 output_dir=checkpoints/sft_s2
    echo SFT_S2_OK
    mkdir -p results/E2s2_sft && cp checkpoints/sft_s2/train_info.json results/E2s2_sft/
    $RUN -m conviction.evaluation.runner --condition base --adapter checkpoints/sft_s2/final --splits $S --natural --batch-size 64 --max-new-tokens 200 --out results/E2s2_sft
    tar czf /workspace/results_E2s2_sft.tgz results/E2s2_sft
    echo E2S2_DONE
    $RUN -m conviction.training.train_sft --config configs/sft_light.yaml --override seed=2 output_dir=checkpoints/sft_light_s2
    mkdir -p results/V2s2 && cp checkpoints/sft_light_s2/train_info.json results/V2s2/sft_light_train_info.json
    $RUN -m conviction.training.train_grpo --config configs/grpo_v2.yaml --override seed=2 init_adapter=checkpoints/sft_light_s2/final output_dir=checkpoints/grpo_v2_s2
    echo V2S2_TRAIN_OK
    cp checkpoints/grpo_v2_s2/train_info.json results/V2s2/
    $RUN -m conviction.evaluation.runner --condition base --adapter checkpoints/sft_light_s2/final,checkpoints/grpo_v2_s2/final --splits $S --natural --batch-size 64 --max-new-tokens 200 --out results/V2s2
    tar czf /workspace/results_V2s2.tgz results/V2s2
    tar czf /workspace/adapters_seeds.tgz checkpoints/sft_s2/final checkpoints/sft_light_s2/final checkpoints/grpo_v2_s2/final
    echo V2S2_DONE ;;
esac
tar czf /workspace/results_$1.tgz results
echo PHASE_DONE_$1

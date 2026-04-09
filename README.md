# EfficientQAT
Official PyTorch implement of paper [EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2407.11062)

## News
- [2025/11] 🔥 **We open-source [INT vs. FP](https://github.com/ChenMnZ/INT_vs_FP), a framework to compare low-bit integer and float-point formats, including MXFP8/MXFP6/MXFP4/NVFP4 and MXINT8/MXINT6/MXINT4/NVINT4.**
- [2025/05] 🔥 We explore the [Scaling Law for Quantization-Aware Training](https://export.arxiv.org/abs/2505.14302), which offers insights and instruction for LLMs QAT.
- [2025/05] 🌟 Our EfficientQAT paper has been accepted for ACL 2025 Main Conference! 🎉 Cheers!
- [2024/10] 🔥 We release a new weight-activation quantization algorithm, [PrefixQuant](https://github.com/ChenMnZ/PrefixQuant), which proposed an efficient method to isolate sink token (token-wise outlier).
- [2024/08] The new inference backend [T-MAC](https://github.com/microsoft/T-MAC) from Microsoft has supported EffcientQAT models.
- [2024/08] We support for the quantization of [Mistral-Large-Instruct](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407). W2g64 Mistral-Large-Instruct with our EfficientQAT can compress the 123B models to 35 GB with only 4 points accuracy degeneration.
- [2024/07] New featurs! We support to transfer EfficientQAT quantized models into `GPTQ v2` format and `BitBLAS` format, which can be directly loaded through [GPTQModel](https://github.com/ModelCloud/GPTQModel).
- [2024/07] We release EfficientQAT, which pushes the limitation of uniform (INT) quantization in an efficient manner.

## Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Common Workflows](#common-workflows)
- [Model Zoo](#model-zoo)
- [Training](#training)
- [Inference](#Inference)
- [Model Transferring](#model-transferring)
- [Inference of Other Formats](#inference-of-other-formats)
- [Citation](#citation)


## Installation
1. Clone this repository and navigate to EfficientQAT folder
```
git clone https://github.com/OpenGVLab/EfficientQAT.git
cd EfficientQAT
```

2. Install package
```
conda create -n efficientqat python==3.11

conda activate efficientqat

pip install -r requirements.txt
```

For CUDA environments that need paged AdamW or Triton kernels, install the optional GPU extras as well:
```
pip install -r requirements-cuda.txt
```

- `requirements.txt` is now the portable baseline and supports newer `transformers` / `accelerate` / `torch` releases.
- `requirements-cuda.txt` keeps the CUDA-specific dependencies optional so the repo can still be used on CPU-only or macOS environments.

## Quick Start

This repo is easiest to use through three entry points:

- `main_block_ap.py`
  Block-wise quantization-aware training (Block-AP). This is the main script for producing an EfficientQAT checkpoint from a full-precision model.
- `main_e2e_qp.py`
  End-to-end fine-tuning of quantization parameters (E2E-QP) on top of a Block-AP checkpoint.
- `eval_quantized_model.py`
  A convenience script in this repo that rebuilds a quantized model from a checkpoint directory plus quant config, then runs WikiText2 PPL and/or sample generation.

Recommended environment setup for Hugging Face access on the current codebase:

```bash
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
```

## Common Workflows

### 1. Block-AP quantization

Example: Llama-3.2-3B-Instruct, `int2 + group_size=32`, skipping `embedding` and `lm_head`.

```bash
cd /root/autodl-tmp/EfficientQAT
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

python main_block_ap.py \
  --model /root/autodl-tmp/llama3.2-3B-instruct \
  --net llama3.2-3B-instruct \
  --cache_dir /root/autodl-tmp/EfficientQAT/cache \
  --output_dir /root/autodl-tmp/EfficientQAT/output/run_log_llama32_3b_w2g32 \
  --save_quant_dir /root/autodl-tmp/EfficientQAT/output/run_model_llama32_3b_w2g32 \
  --real_quant \
  --wbits 2 \
  --group_size 32 \
  --quant_config /root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w2g32_skip_embed_head.json \
  --calib_dataset wikitext2 \
  --train_size 1024 \
  --val_size 64 \
  --training_seqlen 2048 \
  --batch_size 1 \
  --epochs 2 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --min_lr_factor 20
```

Useful knobs:

- `--wbits`
  Weight bitwidth.
- `--group_size`
  Weight grouping size for per-group quantization.
- `--quant_config`
  JSON file for mixed precision / skip rules / per-layer overrides.
- `--calib_dataset`
  Calibration/training data source for Block-AP. Common choices in this repo are `wikitext2`, `c4`, and `redpajama`.
- `--train_size`, `--training_seqlen`, `--epochs`
  Main scaling knobs for quality vs. resource use.
- `--off_load_to_disk`
  Reduces RAM usage by spilling block caches to disk. Useful when large `train_size x seqlen` settings otherwise get killed by the OOM killer.

### 2. Evaluate a quantized checkpoint on WikiText2 PPL

Use the helper script added in this repo:

```bash
cd /root/autodl-tmp/EfficientQAT

python eval_quantized_model.py \
  --model /root/autodl-tmp/EfficientQAT/output/run_model_llama32_3b_w2g32 \
  --base_model_path /root/autodl-tmp/llama3.2-3B-instruct \
  --quant_config /root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w2g32_skip_embed_head.json \
  --wbits 2 \
  --group_size 32 \
  --eval_ppl
```

If you also want a few deterministic sample generations:

```bash
python eval_quantized_model.py \
  --model /root/autodl-tmp/EfficientQAT/output/run_model_llama32_3b_w2g32 \
  --base_model_path /root/autodl-tmp/llama3.2-3B-instruct \
  --quant_config /root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w2g32_skip_embed_head.json \
  --wbits 2 \
  --group_size 32 \
  --eval_ppl \
  --generate
```

The evaluation script supports:

- Single-file `model.safetensors` checkpoints
- Sharded checkpoints
- Tokenizer/embedding size mismatches introduced by later E2E-QP runs
- Per-layer mixed-precision configs via `--quant_config`

### 3. E2E-QP on top of a Block-AP checkpoint

Example: continue training quantization parameters on top of a Block-AP result.

```bash
cd /root/autodl-tmp/EfficientQAT
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

python main_e2e_qp.py \
  --quant_model_path /root/autodl-tmp/EfficientQAT/output/run_model_llama32_3b_w2g32 \
  --base_model_path /root/autodl-tmp/llama3.2-3B-instruct \
  --model_family llama3.2-3B-instruct \
  --quant_config /root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w2g32_skip_embed_head.json \
  --wbits 2 \
  --group_size 32 \
  --learning_rate 2e-5 \
  --dataset alpaca \
  --dataset_format alpaca \
  --output_dir /root/autodl-tmp/EfficientQAT/output/e2e_qp_run_llama32_3b_w2g32_alpaca \
  --do_train True \
  --do_eval False \
  --source_max_len 384 \
  --target_max_len 128 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 2 \
  --evaluation_strategy no \
  --max_steps 500 \
  --bf16 \
  --optim adamw_torch \
  --group_by_length
```

Important notes for `main_e2e_qp.py` on this codebase:

- Prefer passing `--base_model_path`
  This speeds up quantized checkpoint restoration by loading untouched fp16 weights from the original model directory.
- If you want epoch-based training, explicitly pass `--max_steps -1`
  On the current toolchain, leaving `max_steps=0` can result in an immediate stop after one optimizer step in some setups.
- For instruction models, `alpaca` / `deita-*` are the easiest E2E-QP datasets to get running.
- If your target metric is WikiText2 PPL, instruction-style E2E-QP data may not help and can even hurt PPL. For PPL-focused runs, prefer pretraining-style data (`pt` format).

### 4. Mixed precision quantization through JSON config

The repo supports mixed precision and skip rules via `--quant_config`.

Example: keep most layers at `int2 g32`, but quantize all `fc2`/`mlp.down_proj` layers as `int4 per-channel`:

```json
{
  "default": {
    "bits": 2,
    "group_size": 32,
    "mapping": "asymmetric",
    "granularity": "per_group"
  },
  "overrides": [
    {
      "pattern": "mlp.down_proj",
      "bits": 4,
      "granularity": "per_channel",
      "mapping": "asymmetric"
    },
    {
      "pattern": "model.embed_tokens",
      "enabled": false,
      "bits": 16
    },
    {
      "pattern": "lm_head",
      "enabled": false,
      "bits": 16
    }
  ]
}
```

In Llama-style models:

- `fc2` corresponds to `mlp.down_proj`
- `fc1`-like expansion paths correspond to `mlp.gate_proj` and `mlp.up_proj`

The current codebase now resolves config patterns against full module paths such as:

- `model.layers.27.self_attn.v_proj`
- `model.layers.26.mlp.down_proj`

So you can target exact sensitive layers instead of only module-local names.

### 5. Record per-layer gradient sensitivity during Block-AP

To estimate which layers are the most sensitive, enable gradient logging during Block-AP:

```bash
python main_block_ap.py \
  ... \
  --log_grad_sensitivity \
  --grad_sensitivity_topk 40 \
  --grad_sensitivity_sort_by avg_mean_abs_grad
```

This writes:

- `gradient_sensitivity_ranking.json`
  Full ranking of quantized linear layers
- top-k lines into the main log file

Typical output layer names:

- `layers.27.self_attn.v_proj`
- `layers.27.self_attn.o_proj`
- `layers.26.mlp.down_proj`

This is useful for building mixed-precision configs where the most sensitive layers get promoted to `int4`.

### 6. Common local config files added in this repo

Some example configs already present locally:

- [configs/llama32_3b_instruct_w2g32_skip_embed_head.json](/root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w2g32_skip_embed_head.json)
- [configs/llama32_3b_instruct_w4g32_skip_embed_head.json](/root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w4g32_skip_embed_head.json)
- [configs/llama32_3b_instruct_w2g32_fc2_w4pc_skip_embed_head.json](/root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w2g32_fc2_w4pc_skip_embed_head.json)
- [configs/llama32_3b_instruct_w2g32_skip_fc2_embed_head.json](/root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w2g32_skip_fc2_embed_head.json)
- [configs/llama32_3b_instruct_w2g32_top40sens_w4g32_skip_embed_head.json](/root/autodl-tmp/EfficientQAT/configs/llama32_3b_instruct_w2g32_top40sens_w4g32_skip_embed_head.json)
- [configs/llama3_8b_instruct_w2g32_skip_embed_head.json](/root/autodl-tmp/EfficientQAT/configs/llama3_8b_instruct_w2g32_skip_embed_head.json)


## Model Zoo

We provide a number of prequantized EfficientQAT models as follows: 

- WikiText2 PPL is measured in 2048 context length.
- Avg. Accuracy indicate the average accuracy in 5 zero-shot reasoning tasks (WinoGrande,PIQA,HellaSwag,Arc-Easy, Arc-Challenge) with [lm-eval v0.4.2](https://github.com/EleutherAI/lm-evaluation-harness).
- 1GB = $10^9$ Bit
- Hub Link: EQAT indicates the original checkpoints. We also transfer the checkpoints into GPTQ and BitBLAS formats, which can be loaded directly through [GPTQModel](https://github.com/ModelCloud/GPTQModel). (PS: [GPTQModel](https://github.com/ModelCloud/GPTQModel) is a official bug-fixed repo of AutoGPTQ, which would be merged into [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) in future.)

| Model | Quantization | WikiText2 PPL | Avg. Accuracy | Model Size (GB) | Hub link|
|-------|--------------|---------------|---------------|-----------------|----------|
Llama-2-7B|fp16|5.47|64.86|13.2|-|
Llama-2-7B|w4g128|5.53|64.27|3.7|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-2-7b-EfficientQAT-w4g128-BitBLAS)|
Llama-2-7B|w3g128|5.81|64.02|3.1|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w3g128)|
Llama-2-7B|w2g64|6.86|60.14|2.3|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-2-7b-EfficientQAT-w2g64-BitBLAS)|
Llama-2-7B|w2g128|7.17|59.50|2.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-2-7b-EfficientQAT-w2g128-BitBLAS)|
Llama-2-13B|fp16|4.88|67.81|25.4|-|
Llama-2-13B|w4g128|4.93|67.52|6.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-7b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-2-7b-EfficientQAT-w4g128-BitBLAS)|
Llama-2-13B|w3g128|5.12|67.28|5.6|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w3g128)|
Llama-2-13B|w2g64|5.96|64.88|4.0|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-2-13b-EfficientQAT-w2g64-BitBLAS)|
Llama-2-13B|w2g128|6.08|63.88|3.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-13b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-2-13b-EfficientQAT-w2g128-BitBLAS)|
Llama-2-70B|fp16|3.32|72.41|131.6|-|
Llama-2-70B|w4g128|3.39|72.62|35.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-2-70b-EfficientQAT-w4g128-BitBLAS)|
Llama-2-70B|w3g128|3.61|71.76|29.1|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w3g128)|
Llama-2-70B|w2g64|4.52|69.48|20.1|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-2-70b-EfficientQAT-w2g64-BitBLAS)|
Llama-2-70B|w2g128|4.61|68.93|18.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-2-70b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-2-70b-EfficientQAT-w2g128-BitBLAS)|
Llama-3-8B|fp16|6.14|68.58|13.0|-|
Llama-3-8B|w4g128|6.47|68.43|5.4|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-8b-EfficientQAT-w4g128-BitBLAS)|
Llama-3-8B|w3g128|7.09|67.35|4.7|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w3g128)|
Llama-3-8B|w2g64|9.41|60.76|3.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-8b-EfficientQAT-w2g64-BitBLAS)|
Llama-3-8B|w2g128|9.80|59.36|3.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-3-8b-EfficientQAT-w2g128-BitBLAS)|
Llama-3-70B|fp16|2.85|75.33|137.8|-|
Llama-3-70B|w4g128|3.17|74.57|38.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-70b-EfficientQAT-w4g128-BitBLAS)|
Llama-3-70B|w3g128|4.19|72.42|32.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w3g128)|
Llama-3-70B|w2g64|6.08|67.89|23.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g64-GPTQ)|
Llama-3-70B|w2g128|6.38|67.57|22.0|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-3-70b-EfficientQAT-w2g128-BitBLAS)|
Llama-3-8B-Instruct|fp16|8.29|68.43|13.0|-|
Llama-3-8B-Instruct|w4g128|7.93|68.39|5.4|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-8b-instruct-EfficientQAT-w4g128-BitBLAS)|
Llama-3-8B-Instruct|w3g128|8.55|67.24|4.7|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w3g128)|
Llama-3-8B-Instruct|w2g64|11.19|60.66|3.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-3-8b-instruct-EfficientQAT-w2g64-BitBLAS)|
Llama-3-8B-Instruct|w2g128|11.73|60.16|3.8|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-3-8b-instruct-EfficientQAT-w2g128-BitBLAS)|
Llama-3-70B-Instruct|fp16|5.33|73.78|137.8|-|
Llama-3-70B-Instruct|w4g128|5.35|73.47|38.9|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w4g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w4g128-GPTQ)\|[BitBLAS](Llama-3-70b-instruct-EfficientQAT-w4g128-BitBLAS)|
Llama-3-70B-Instruct|w3g128|5.65|72.87|32.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w3g128)|
Llama-3-70B-Instruct|w2g64|7.86|67.64|23.2|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g64)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g64-GPTQ)\|[BitBLAS](Llama-3-70b-instruct-EfficientQAT-w2g64-BitBLAS)|
Llama-3-70B-Instruct|w2g128|8.14|67.54|22.0|[EQAT](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g128)\|[GPTQ](https://huggingface.co/ChenMnZ/Llama-3-70b-instruct-EfficientQAT-w2g128-GPTQ)\|[BitBLAS](Llama-3-70b-instruct-EfficientQAT-w2g128-BitBLAS)|
Mistral-Large-Instruct-2407|fp16|2.74|77.76|228.5|-|
Mistral-Large-Instruct-2407|w2g64|5.58|73.54|35.5|[GPTQ](https://huggingface.co/ChenMnZ/Mistral-Large-Instruct-2407-EfficientQAT-w2g64-GPTQ)

## Training
EfficientQAT involves two consecutive training phases: Block-wise training of all parameters (**Block-AP**) and end-to-end training of quantization parameters (**E2E-QP**). The detailed training script can be found in `./examples`. We give the training script examples on Llama-2-7B with w2g64 quantization in the following. 

1. Block-AP

You should modify `--model` to the folder of full-precision model  in the script before you running the following command.
```
bash examples/block_ap/Llama-2-7b/w2g64.sh
```
Specifically, the `--weight_lr` is `2e-5` for 2-bit and `1e-5` for 3-/4-bits in our experiments.

Some other important arguments:
- `--train_size`: number of training data samples, 4096 as default
- `--val_size`: number of validation data samples, 64 as default
- `--off_load_to_disk`: save training dataset to disk, saving CPU memory but may reduce training speed
- `--quant_config`: optional JSON file for per-layer overrides, mixed precision, or skip rules. When omitted, EfficientQAT keeps the original global `--wbits/--group_size` behavior.


2. E2E-QP

Then, you can load the quantized model of Block-AP for further E2E-QP. Specifically, E2E-QP can adapt to different scenarios by changing the training datasets. You should modify `--quant_model_path` to the folder of quantized model in the script before you running the following command.

1\) Train on RedPajama
```
bash examples/e2e_qp/Llama-2-7b/w2g64-redpajama.sh
``` 

2\) Train on Alpaca
```
bash examples/e2e_qp/Llama-2-7b/w2g128-redpajama.sh
```
Specifically, the `--learning_rate` is `2e-5` for 2-bit and `1e-5` for 3-/4-bits in our experiments. You can decrease the `--per_device_train_batch_size` to reduce the memory footprint during training, and making sure that `--gradient_accumulation_steps`  increases by the same multiple to maintain the same batch size.

Example per-layer quantization config:
```json
{
  "default": {
    "bits": 2,
    "group_size": 32,
    "mapping": "asymmetric",
    "granularity": "per_group"
  },
  "overrides": [
    {
      "pattern": "model.embed_tokens",
      "enabled": false,
      "bits": 16
    },
    {
      "pattern": "*.self_attn.o_proj",
      "bits": 8,
      "granularity": "per_channel",
      "mapping": "symmetric"
    }
  ]
}
```



## Inference

1. Download the pre-quantized EfficientQAT models from Huggingface
```
pip install huggingface_hub

huggingface-cli download ChenMnZ/Llama-2-7b-EfficientQAT-w2g64 --local-dir ./output/pre_quantized_models/Llama-2-7b-EfficientQAT-w2g64
```

2. Evaluate the pre-quantized EfficientQAT model
```
CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--resume_quant ./output/pre_quantized_models/Llama-2-7b-EfficientQAT-w2g64 \
--net Llama-2 \
--wbits 2 \
--group_size 64 \
--output_dir ./output/inference_results/ \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande
```


## Model Transferring
Firstly, you should install `gptqmodel` package to support GPTQ and BitBLAS quantization format:
```
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel
bash install.sh
```
- In our experiences, we test with `gptqmodel v0.9.8`.

Then, we offer three types of transferring as follows:

1. Transfer EfficientQAT checkpoints to GPTQ format
```
bash examples/model_transfer/efficientqat_to_gptq/llama-2-7b.sh
```
- **Note**: Currently [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) has overflow bugs for asymmetric quantization. Therefore, we choose the official bug-fixed version [GPTQModel](https://github.com/ModelCloud/GPTQModel) to transfer our asymmetric quantized models. Therefore, the GPTQ models provide by this repo can be only successfully loaded through [GPTQModel](https://github.com/ModelCloud/GPTQModel) otherwise [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ).


2. Transfer EfficientQAT checkpoints to BitBLAS format
```
bash examples/model_transfer/efficientqat_to_bitblas/llama-2-7b.sh
```
- Speedup has some problem, refer [this issue](https://github.com/microsoft/BitBLAS/issues/90) for details.
- The GPTQ and BitBLAS exporters still expect a uniform quantization layout. If you use per-layer overrides, generate TorchAO metadata instead of these formats.

3. Generate TorchAO / ExecuTorch quantization metadata
```
python model_transfer/efficientqat_to_torchao.py \
  --model ./output/block_ap_models/Llama-2-7b-w2g64 \
  --save_dir ./output/torchao/Llama-2-7b-w2g64
```
- This command emits `torchao_quant_manifest.json` plus the normalized `efficientqat_quant_config.json`, which can be used as the handoff layer for PT2E / ExecuTorch export pipelines.

4. Transfer fp32 datas in EfficientQAT checkpoints to half-precision counterparts.
Some of parameters are saved as fp32 for training, you can transfer them into half-precision to further reducing model size after training. 
```
bash examples/model_transfer/fp32_to_16/llama-2-7b.sh
```

## Inference of Other Formats
Below is an example to inference with GPTQ or BitBLAS quantized formats.
```Python
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

quant_dir = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-GPTQ"
# quant_dir = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-BitBLAS"
# or local path

tokenizer = AutoTokenizer.from_pretrained(quant_dir, use_fast=True)


# load quantized model to the first GPU
model = GPTQModel.from_quantized(quant_dir)

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("Model quantization is", return_tensors="pt").to(model.device))[0]))
```


## Citation
If you found this work useful, please consider citing:
```
@article{efficientqat,
  title={EfficientQAT: Efficient Quantization-Aware Training for Large Language Models},
  author={Chen, Mengzhao and Shao, Wenqi and Xu, Peng and Wang, Jiahao and Gao, Peng and Zhang, Kaipeng and Qiao, Yu and Luo, Ping},
  journal={arXiv preprint arXiv:2407.11062},
  year={2024}
}
```

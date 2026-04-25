# EfficientQAT

这个仓库当前只保留以下功能：

- Block-AP 量化训练
- E2E-QP 量化参数继续训练
- 量化模型的推理 / 生成测试
- WikiText2 PPL 测试
- 量化模型反量化并导出为标准 Hugging Face safetensors 格式

保留的主脚本：

- `main_block_ap.py`
- `main_e2e_qp.py`
- `eval_quantized_model.py`
- `model_transfer/dequantize_to_hf.py`

支持的模型类型：

- `llama`
- `qwen3`

建议显式传：

```bash
--model_kind llama
--model_kind qwen3
```

也支持：

```bash
--model_kind auto
```

## 环境

```bash
cd /root/autodl-tmp/EfficientQAT
conda activate efficientqat
```

如果需要从 Hugging Face 拉数据：

```bash
source /etc/network_turbo
```

## 1. Block-AP 量化

脚本：

```bash
python main_block_ap.py ...
```

常用参数：

- `--model`：原始模型目录
- `--net`：模型名称，用于缓存命名
- `--model_kind`：`auto / llama / qwen3`
- `--calib_dataset`：`wikitext2 / c4 / redpajama`
- `--wbits`：量化 bit 数
- `--group_size`：分组大小
- `--train_size`：校准训练样本数
- `--training_seqlen`：Block-AP 序列长度
- `--real_quant`：保存真实量化权重
- `--off_load_to_disk`：把 block cache 落盘，省内存
- `--quant_config`：可选 JSON 量化配置
- `--save_quant_dir`：量化模型输出目录
- `--output_dir`：日志目录
- `--cache_dir`：缓存目录
- `--trust_remote_code`：加载自定义模型代码

### Qwen3 示例

```bash
python main_block_ap.py \
  --model /root/autodl-tmp/Qwen3-1.7b \
  --net Qwen3-1.7b \
  --model_kind qwen3 \
  --calib_dataset wikitext2 \
  --wbits 2 \
  --group_size 32 \
  --train_size 8192 \
  --training_seqlen 2048 \
  --real_quant \
  --off_load_to_disk \
  --save_quant_dir /root/autodl-tmp/EfficientQAT/output/Qwen3-1.7b-w2g32 \
  --output_dir /root/autodl-tmp/EfficientQAT/logs/Qwen3-1.7b-w2g32 \
  --cache_dir /root/autodl-tmp/EfficientQAT/cache \
  --trust_remote_code
```

### Llama 示例

```bash
python main_block_ap.py \
  --model /path/to/model \
  --net model-name \
  --model_kind llama \
  --calib_dataset wikitext2 \
  --wbits 2 \
  --group_size 32 \
  --train_size 8192 \
  --training_seqlen 2048 \
  --real_quant \
  --off_load_to_disk \
  --save_quant_dir /path/to/output \
  --output_dir /path/to/logs \
  --cache_dir /path/to/cache
```

### 本地 WikiText2 缓存

代码会优先读取：

- `/root/autodl-tmp/EfficientQAT/cache/wikitext2_official/train.parquet`
- `/root/autodl-tmp/EfficientQAT/cache/wikitext2_official/test.parquet`

### 本地 RedPajama parquet

代码会优先读取环境变量 `EFFICIENTQAT_REDPJ_LOCAL_DIR` 指向的 parquet 目录。

```bash
export EFFICIENTQAT_REDPJ_LOCAL_DIR=/root/autodl-tmp/EfficientQAT/cache/redpajama_v2_sample_en_parquet
```

## 2. E2E-QP 继续训练

脚本：

```bash
python main_e2e_qp.py ...
```

常用参数：

- `--quant_model_path`：Block-AP 量化模型目录
- `--base_model_path`：原始模型目录
- `--model_family`：模型名称
- `--model_kind`：`auto / llama / qwen3`
- `--dataset`：`alpaca / redpajama / deita-6k / deita-10k / c4`
- `--pt_context_len`：训练上下文长度
- `--max_train_samples`：限制训练样本数
- `--wbits / --group_size`
- `--output_dir`
- `--num_train_epochs`
- `--max_steps`

### Qwen3 + deita 示例

```bash
python main_e2e_qp.py \
  --quant_model_path /root/autodl-tmp/EfficientQAT/output/Qwen3-1.7b-w2g32-8192 \
  --base_model_path /root/autodl-tmp/Qwen3-1.7b \
  --model_family Qwen3-1.7b \
  --model_kind qwen3 \
  --dataset deita-10k \
  --conv_temp llama-2 \
  --wbits 2 \
  --group_size 32 \
  --pt_context_len 2048 \
  --max_train_samples 8192 \
  --output_dir /root/autodl-tmp/EfficientQAT/output/e2e_qp_Qwen3-1.7b_w2g32_deita10k_t2048_s8192_e1 \
  --do_train True \
  --do_eval False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 2 \
  --evaluation_strategy no \
  --max_steps -1 \
  --num_train_epochs 1 \
  --bf16 \
  --optim adamw_torch \
  --trust_remote_code
```

说明：

- 如果目标是 `WikiText2 PPL`，`deita-*` 不一定更优。
- 如果目标是指令/对话效果，`deita-*` 更合适。
- `Qwen3` 当前沿用 `llama-2` 风格的 `deita` 对话模板。

## 3. PPL 与生成测试

脚本：

```bash
python eval_quantized_model.py ...
```

常用参数：

- `--model`：量化模型目录
- `--base_model_path`：原始模型目录
- `--quant_config`：量化配置 JSON
- `--model_kind`
- `--eval_ppl`
- `--wikitext_parquet`
- `--generate`
- `--max_new_tokens`
- `--trust_remote_code`

### PPL 示例

```bash
python eval_quantized_model.py \
  --model /path/to/quantized_model \
  --base_model_path /path/to/base_model \
  --quant_config /path/to/efficientqat_quant_config.json \
  --model_kind qwen3 \
  --wbits 2 \
  --group_size 32 \
  --device cuda:0 \
  --eval_ppl \
  --wikitext_parquet /root/autodl-tmp/EfficientQAT/cache/wikitext2_official/test.parquet \
  --trust_remote_code
```

### 生成测试示例

```bash
python eval_quantized_model.py \
  --model /path/to/quantized_model \
  --base_model_path /path/to/base_model \
  --quant_config /path/to/efficientqat_quant_config.json \
  --model_kind qwen3 \
  --wbits 2 \
  --group_size 32 \
  --device cuda:0 \
  --generate \
  --max_new_tokens 128 \
  --trust_remote_code
```

## 4. 反量化导出为 Hugging Face safetensors

脚本：

```bash
python model_transfer/dequantize_to_hf.py ...
```

功能：

- 加载 EfficientQAT 量化模型
- 执行 `unpack + 反量化`
- 将量化 `Linear` 恢复为普通 `nn.Linear`
- 保存为标准 Hugging Face `safetensors` 目录

常用参数：

- `--model`：量化模型目录
- `--base_model_path`：原始模型目录
- `--quant_config`
- `--model_kind`
- `--target_dtype auto|fp16|bf16|fp32`
- `--save_dir`
- `--dry_run`
- `--trust_remote_code`

### Qwen3 示例

```bash
PYTHONPATH=/root/autodl-tmp/EfficientQAT python /root/autodl-tmp/EfficientQAT/model_transfer/dequantize_to_hf.py \
  --model /root/autodl-tmp/EfficientQAT/output/Qwen3-1.7b-w2g32-8192 \
  --base_model_path /root/autodl-tmp/Qwen3-1.7b \
  --quant_config /root/autodl-tmp/EfficientQAT/output/Qwen3-1.7b-w2g32-8192/efficientqat_quant_config.json \
  --model_kind qwen3 \
  --wbits 2 \
  --group_size 32 \
  --target_dtype auto \
  --save_dir /root/autodl-tmp/EfficientQAT/output/Qwen3-1.7b-dequantized \
  --trust_remote_code
```

仅验证、不保存：

```bash
PYTHONPATH=/root/autodl-tmp/EfficientQAT python /root/autodl-tmp/EfficientQAT/model_transfer/dequantize_to_hf.py \
  --model /path/to/quantized_model \
  --base_model_path /path/to/base_model \
  --quant_config /path/to/efficientqat_quant_config.json \
  --model_kind qwen3 \
  --wbits 2 \
  --group_size 32 \
  --target_dtype auto \
  --save_dir /tmp/ignore \
  --trust_remote_code \
  --dry_run
```

## 5. 保留内容

保留的源码：

- `quantize/`
- `deita_dataset/`
- `model_transfer/dequantize_to_hf.py`

保留的主脚本：

- `main_block_ap.py`
- `main_e2e_qp.py`
- `eval_quantized_model.py`

保留的辅助文件：

- `datautils_block.py`
- `datautils_e2e.py`
- `hf_compat.py`
- `utils.py`
- `requirements.txt`
- `requirements-cuda.txt`

不会因整理而删除：

- `output/`
- `logs/`
- `cache/`

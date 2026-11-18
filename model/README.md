# Model：Qwen2.5-14B（W7900D）

| 角色 | 机器 | 说明 |
|---|---|---|
| 训练 | AIA **W7900D**（ROCm，~48GB） | 默认 `Qwen/Qwen2.5-14B-Instruct` + bf16 LoRA |
| 本机 Ollama | `qwen2.5:7b` / `qwen2.5-mc:7b` | 只做小模型联调，不是本默认训练目标 |

**不能对 Ollama GGUF 直接 LoRA。** 为何选 14B：

- 7B：稳，但 W7900D 显存浪费较多  
- **14B**：48GB 上 LoRA + gradient checkpoint 通常够用，收益更好  
- 32B：bf16 偏紧；QLoRA 依赖 bitsandbytes，ROCm 上常不稳，等 14B 跑通再试  

## 环境（W7900D Console）

```bash
# torch 已是 ROCm 版时不要重装覆盖
python3 -m pip install -r model/requirements.txt

python3 - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

默认**不要**加 `--qlora`。

## 数据 → 训练

```bash
python model/prepare_sft.py --traces "runs/*/trajectory.jsonl" --out model/sft_data

python model/train_lora.py --check-only \
  --train model/sft_data/train.jsonl --valid model/sft_data/valid.jsonl

tmux new -s train
python model/train_lora.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --train model/sft_data/train.jsonl --valid model/sft_data/valid.jsonl \
  --out model/outputs/qwen25-14b-mc-lora 2>&1 | tee model/train.log
```

显存紧时再试：`--max-length 1536` 或减小 `--grad-accum` 以外的激活（保持 microbatch=1）。

## 离线评估

把 adapter+基座起成 OpenAI-compatible 后：

```bash
python model/evaluate_offline.py \
  --data model/sft_data/valid.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model mc-14b --api-key EMPTY
```

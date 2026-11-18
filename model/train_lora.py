"""LoRA / QLoRA SFT：公开 API 动作 JSON。

Ollama 的 GGUF 不能直接训。训练基座用 HuggingFace 指令权重。

主训练机：AIA W7900D（ROCm，约 48GB）→ 默认 Qwen2.5-14B-Instruct + bf16 LoRA。
7B 留给本机 Ollama 联调；32B 需 QLoRA，ROCm 上 bitsandbytes 常不稳，暂不默认。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

# W7900D 48GB：14B LoRA 是体量与稳定的折中
DEFAULT_MODEL = 'Qwen/Qwen2.5-14B-Instruct'

def read_messages(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip(): continue
        obj = json.loads(line)
        messages = obj.get('messages')
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f'{path}:{line_no}: expected system,user,assistant')
        if [x.get('role') for x in messages] != ['system', 'user', 'assistant']:
            raise ValueError(f'{path}:{line_no}: unexpected roles')
        target = json.loads(messages[-1]['content'])
        if not isinstance(target, dict) or not isinstance(target.get('args'), dict):
            raise ValueError(f'{path}:{line_no}: malformed action')
        rows.append(messages)
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(description='Train a JSON action policy with LoRA')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='HuggingFace 基座 ID（W7900D 默认 14B）')
    parser.add_argument('--train', default='sft_data/train.jsonl')
    parser.add_argument('--valid', default='sft_data/valid.jsonl')
    parser.add_argument('--out', default='outputs/qwen25-14b-mc-lora')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--qlora', action='store_true',
                        help='4bit QLoRA（RTX 5060 8GB 等笔记本显卡需要）')
    parser.add_argument('--check-only', action='store_true',
                        help='只校验 JSON，不下载模型、不用 GPU')
    args = parser.parse_args(argv)
    if args.max_length < 256 or args.grad_accum < 1:
        parser.error('invalid length or accumulation')
    train = read_messages(args.train)
    valid = read_messages(args.valid) if Path(args.valid).exists() else []
    if not train:
        parser.error('training set is empty')
    print(json.dumps({
        'train_examples': len(train),
        'valid_examples': len(valid),
        'model': args.model,
        'qlora': args.qlora,
        'note': '训后部署 OpenAI-compatible 或自建 ollama 14B；本机 7B 仅联调基线',
    }, ensure_ascii=False))
    if args.check_only:
        return

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                                  TrainingArguments, BitsAndBytesConfig)
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    except ImportError as exc:
        raise SystemExit(
            '先装 CUDA 版 torch，再 pip install -r model/requirements.txt'
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            '当前 torch 无 CUDA（本机是 CPU wheel）。'
            'RTX 5060 需安装匹配驱动的 CUDA torch 后再训；可用 --check-only 先验数据。'
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError('tokenizer has no EOS token')

    class Examples(Dataset):
        def __init__(self, messages):
            self.samples = []
            omitted = 0
            for triplet in messages:
                prompt = tokenizer.apply_chat_template(
                    triplet[:2], tokenize=True, add_generation_prompt=True)
                answer = tokenizer.encode(triplet[2]['content'], add_special_tokens=False) + [eos]
                if len(prompt) + len(answer) > args.max_length:
                    omitted += 1
                    continue
                self.samples.append({
                    'input_ids': prompt + answer,
                    'labels': [-100] * len(prompt) + answer,
                })
            print(f'usable_examples={len(self.samples)} skipped_too_long={omitted}')
            if not self.samples:
                raise ValueError('all examples exceed --max-length')
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    def collate(batch):
        size = max(len(row['input_ids']) for row in batch)
        ids, masks, labels = [], [], []
        for row in batch:
            p = size - len(row['input_ids'])
            ids.append(row['input_ids'] + [tokenizer.pad_token_id] * p)
            masks.append([1] * len(row['input_ids']) + [0] * p)
            labels.append(row['labels'] + [-100] * p)
        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(masks),
            'labels': torch.tensor(labels),
        }

    bfloat = bool(torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bfloat else torch.float16
    load_kw = {'trust_remote_code': False}
    if args.qlora:
        load_kw['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        load_kw['device_map'] = 'auto'
    else:
        load_kw['torch_dtype'] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    ))
    model.print_trainable_parameters()
    train_ds = Examples(train)
    valid_ds = Examples(valid) if valid else None
    training = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_ratio=0.03, weight_decay=0.01, lr_scheduler_type='cosine',
        logging_steps=10, save_strategy='epoch', save_total_limit=2,
        eval_strategy='epoch' if valid_ds else 'no',
        fp16=not bfloat, bf16=bfloat, gradient_checkpointing=True,
        dataloader_num_workers=0, report_to='none', remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training, train_dataset=train_ds,
                      eval_dataset=valid_ds, data_collator=collate)
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f'LoRA adapter saved at: {args.out}')
    print('下一步：merge adapter 后部署 OpenAI-compatible 推理（见 model/README.md）')

if __name__ == '__main__':
    main()

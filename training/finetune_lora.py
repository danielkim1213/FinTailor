"""16-bit LoRA fine-tune of Qwen3.5 with Unsloth + TRL (plan section 7). Run on Colab/Kaggle GPU.

Stage 1 (free T4):  python training/finetune_lora.py --data train_mix.jsonl --model unsloth/Qwen3.5-4B
Quick timing test:  ... --max-steps 100
Stage 2 (Pro L4/A100): --model unsloth/Qwen3.5-9B --rank 32 --seq 4096 --lr 1e-4 --epochs 2

Checkpoints every 100 steps to --out (point it at Google Drive) and optionally the HF Hub.
Thinking is disabled in the chat template for training AND inference (structured JSON tasks;
keeps CPU inference fast). If you later want thinking mode, keep >=75% reasoning examples.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data")
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--out", default="/content/drive/MyDrive/fintailor/lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=10000)
    ap.add_argument("--hub-repo", help="e.g. yourname/fintailor-qwen35-4b-lora (needs HF_TOKEN)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--export", action="store_true", help="also save merged 16-bit + GGUF q4_k_m")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    try:
        from unsloth import FastModel as Loader          # handles text + multimodal checkpoints
    except ImportError:
        from unsloth import FastLanguageModel as Loader
    from trl import SFTConfig, SFTTrainer

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model, tok = Loader.from_pretrained(model_name=a.model, max_seq_length=a.seq, load_in_4bit=False,
                                        load_in_8bit=False, full_finetuning=False)
    model = Loader.get_peft_model(model, r=a.rank, lora_alpha=a.alpha, lora_dropout=0, bias="none",
                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                  use_gradient_checkpointing="unsloth", random_state=3407)
    text_tok = getattr(tok, "tokenizer", tok)  # processor -> tokenizer for multimodal checkpoints

    def fmt(batch):
        out = []
        for msgs in batch["messages"]:
            try:
                out.append(text_tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False))
            except TypeError:
                out.append(text_tok.apply_chat_template(msgs, tokenize=False))
        return {"text": out}

    ds = load_dataset("json", data_files=a.data, split="train").shuffle(seed=3407)
    ds = ds.select(range(min(a.max_examples, len(ds)))).map(fmt, batched=True, remove_columns=ds.column_names)
    ev = None
    if a.eval_data:
        ev = load_dataset("json", data_files=a.eval_data, split="train")
        ev = ev.select(range(min(300, len(ev)))).map(fmt, batched=True, remove_columns=ev.column_names)

    cfg = SFTConfig(
        output_dir=a.out, dataset_text_field="text", max_length=a.seq,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, num_train_epochs=a.epochs,
        max_steps=a.max_steps, optim="adamw_8bit", weight_decay=0.01, fp16=not bf16, bf16=bf16,
        logging_steps=10, save_steps=100, save_total_limit=3, eval_strategy="steps" if ev else "no",
        eval_steps=200, seed=3407, report_to="wandb" if a.wandb else "none",
        push_to_hub=bool(a.hub_repo), hub_model_id=a.hub_repo, hub_strategy="checkpoint" if a.hub_repo else "every_save")
    trainer = SFTTrainer(model=model, processing_class=text_tok, train_dataset=ds, eval_dataset=ev, args=cfg)
    try:  # loss on assistant tokens only
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                          response_part="<|im_start|>assistant\n")
    except Exception as e:  # noqa: BLE001
        print("train_on_responses_only unavailable:", e)
    stats = trainer.train(resume_from_checkpoint=a.resume or None)
    print(json.dumps(stats.metrics, indent=1))

    final = os.path.join(a.out, "final_adapter")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    if a.hub_repo:
        model.push_to_hub(a.hub_repo, token=os.environ.get("HF_TOKEN"))
        tok.push_to_hub(a.hub_repo, token=os.environ.get("HF_TOKEN"))
    if a.export:
        model.save_pretrained_merged(os.path.join(a.out, "merged_16bit"), tok, save_method="merged_16bit")
        model.save_pretrained_gguf(os.path.join(a.out, "gguf"), tok, quantization_method="q4_k_m")
    print("saved:", final)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from src.prompts import caption_prompt

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from .model import chat, load

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--prompt", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--seed", required=True, type=int)
args = parser.parse_args()

set_seed(args.seed)
prompt = caption_prompt(args.prompt)
train = [json.loads(line) for line in open(args.data / "train.jsonl", encoding="utf-8")]
val = [json.loads(line) for line in open(args.data / "val.jsonl", encoding="utf-8")]

model, processor = load(low_cpu_mem_usage=True)
for p in model.parameters():
    p.requires_grad = False
model.config.text_config.use_cache = False
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
))
model.print_trainable_parameters()


def collate(examples):
    (e,) = examples
    with Image.open(e["image_path"]) as im:
        image = im.convert("RGB")
    question = processor.apply_chat_template(chat(prompt), tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(chat(prompt, e["caption"]), tokenize=False)
    inputs = processor(text=[full], images=[image], return_tensors="pt")
    question_len = processor(text=[question], images=[image], return_tensors="pt")["input_ids"].shape[1]
    labels = inputs["input_ids"].clone()
    labels[:, :question_len] = -100
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    return inputs


class KeepBest(TrainerCallback):
    def __init__(self):
        best = args.out / "best.json"
        self.best = json.loads(best.read_text())["val_loss"] if best.exists() else float("inf")

    def on_evaluate(self, training_args, state, control, metrics=None, model=None, **kwargs):
        if metrics["eval_loss"] < self.best:
            self.best = metrics["eval_loss"]
            model.save_pretrained(args.out / "best")
            (args.out / "best.json").write_text(json.dumps({"epoch": state.epoch, "val_loss": self.best}))


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        logging_steps=10,
        disable_tqdm=True,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    ),
    train_dataset=train,
    eval_dataset=val,
    data_collator=collate,
    callbacks=[KeepBest()],
)

# transformers 4.45 checkpoints contain numpy RNG state that torch 2.8 won't unpickle by default
with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                                        type(np.dtype("uint32"))]):
    trainer.train(resume_from_checkpoint=any(args.out.glob("checkpoint-*")))

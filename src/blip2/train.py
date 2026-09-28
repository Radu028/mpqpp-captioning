import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import get_scheduler

from .model import load, to_cuda

EPOCHS, PATIENCE, BATCH, LR = 15, 3, 4, 1e-4

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--seed", required=True, type=int)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
train = [json.loads(line) for line in open(args.data / "train.jsonl", encoding="utf-8")]
val = [json.loads(line) for line in open(args.data / "val.jsonl", encoding="utf-8")]

model, processor = load(torch.bfloat16)
model.language_model.config.use_cache = False
image_token = model.config.image_token_index
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))
model.print_trainable_parameters()
# LoRA weights stay in fp32, the forward pass runs in bf16 autocast
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()


def loss_on(examples):
    images = []
    for e in examples:
        with Image.open(e["image_path"]) as im:
            images.append(im.convert("RGB"))
    batch = processor(images=images, text=[e["caption"] for e in examples], padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    labels[batch["input_ids"] == image_token] = -100
    batch["labels"] = labels
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**to_cuda(batch, torch.bfloat16)).loss


optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=int(0.05 * steps), num_training_steps=steps)

best, bad_epochs = float("inf"), 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    random.shuffle(train)
    for step, i in enumerate(range(0, len(train), BATCH), 1):
        loss = loss_on(train[i:i + BATCH])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss, epoch {epoch}, step {step}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % 50 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)

    model.eval()
    with torch.inference_mode():
        losses = [loss_on(val[i:i + BATCH]).item() for i in range(0, len(val), BATCH)]
    val_loss = sum(losses) / len(losses)
    if not math.isfinite(val_loss):
        raise RuntimeError(f"Non-finite validation loss, epoch {epoch}")
    print(f"epoch {epoch} val loss {val_loss:.4f}", flush=True)

    if val_loss < best:
        best, bad_epochs = val_loss, 0
        model.save_pretrained(args.out / "best")
        (args.out / "best.json").write_text(json.dumps({"epoch": epoch, "val_loss": val_loss}))
    else:
        bad_epochs += 1
        if bad_epochs == PATIENCE:
            break

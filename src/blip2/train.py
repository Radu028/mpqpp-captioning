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

EPOCHS = 15
PATIENCE = 3
BATCH = 4
LR = 1e-4

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--seed", required=True, type=int)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

with (args.data / "train.jsonl").open(encoding="utf-8") as file:
    train = [json.loads(line) for line in file]
with (args.data / "val.jsonl").open(encoding="utf-8") as file:
    val = [json.loads(line) for line in file]

model, processor = load(torch.bfloat16)
model.language_model.config.use_cache = False
image_token = model.config.image_token_index

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# Trainable weights stay in fp32. The forward pass uses bf16.
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()


def loss_on(examples):
    images = []
    for example in examples:
        with Image.open(example["image_path"]) as image:
            images.append(image.convert("RGB"))

    captions = [example["caption"] for example in examples]
    batch = processor(images=images, text=captions, padding=True, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    labels[batch["input_ids"] == image_token] = -100
    batch["labels"] = labels

    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**to_cuda(batch, torch.bfloat16)).loss


params = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR)
steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
scheduler = get_scheduler(
    "cosine", optimizer, num_warmup_steps=int(0.05 * steps), num_training_steps=steps
)

best = float("inf")
bad_epochs = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    random.shuffle(train)

    for step, i in enumerate(range(0, len(train), BATCH), 1):
        loss = loss_on(train[i : i + BATCH])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is NaN or inf at epoch {epoch}, step {step}")

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)

    model.eval()
    with torch.inference_mode():
        losses = []
        for i in range(0, len(val), BATCH):
            losses.append(loss_on(val[i : i + BATCH]).item())

    val_loss = sum(losses) / len(losses)
    if not math.isfinite(val_loss):
        raise RuntimeError(f"Validation loss is NaN or inf at epoch {epoch}")

    print(f"epoch {epoch} val loss {val_loss:.4f}", flush=True)

    if val_loss < best:
        best = val_loss
        bad_epochs = 0
        model.save_pretrained(args.out / "best")
        (args.out / "best.json").write_text(json.dumps({"epoch": epoch, "val_loss": val_loss}))
    else:
        bad_epochs += 1
        if bad_epochs == PATIENCE:
            break

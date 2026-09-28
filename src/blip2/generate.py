import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import set_seed

from .model import caption, load

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--adapter")
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()

set_seed(args.seed)
with (args.data / "test.jsonl").open(encoding="utf-8") as file:
    test = [json.loads(line) for line in file]

model, processor = load(torch.float16)
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter).eval()

tmp = args.out.with_suffix(".tmp")
with open(tmp, "w", encoding="utf-8") as f:
    for i, row in enumerate(test, 1):
        with Image.open(row["image_path"]) as im:
            text = caption(model, processor, im.convert("RGB"))

        prediction = {
            "image_id": row["image_id"],
            "reference": row["caption"],
            "prediction": text,
        }
        f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        print(f"{i}/{len(test)} {text}", flush=True)

tmp.replace(args.out)

import argparse
import json
from pathlib import Path
from src.prompts import caption_prompt

import torch
from peft import PeftModel
from PIL import Image
from transformers import set_seed

from .model import chat, load

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--prompt", required=True, type=Path)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--adapter")
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()

set_seed(args.seed)
test = [json.loads(line) for line in open(args.data / "test.jsonl", encoding="utf-8")]
model, processor = load(device_map={"": 0})
processor.tokenizer.padding_side = "left"
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
question = processor.apply_chat_template(chat(caption_prompt(args.prompt)),
                                         tokenize=False, add_generation_prompt=True)

tmp = args.out.with_suffix(".tmp")
with open(tmp, "w", encoding="utf-8") as f:
    for i, row in enumerate(test, 1):
        with Image.open(row["image_path"]) as im:
            inputs = processor(text=[question], images=[im.convert("RGB")], return_tensors="pt")
        inputs = inputs.to(model.device, torch.bfloat16)
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=128, repetition_penalty=1.1)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        f.write(json.dumps({"image_id": row["image_id"], "reference": row["caption"], "prediction": text},
                           ensure_ascii=False) + "\n")
        print(f"{i}/{len(test)} {text}", flush=True)
tmp.replace(args.out)

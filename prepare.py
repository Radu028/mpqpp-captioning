"""Builds data/<lang>/{train,val,test}.jsonl from a reviewed CSV, keeping the original split."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from src.checks import sha256
from src.images import download_images

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", nargs="+", help="One reviewed CSV, or several non-overlapping split CSVs")
parser.add_argument("--lang", required=True)
parser.add_argument("--images", default=ROOT / "images", type=Path, help="Image root containing coco/ and drawbench/ (default: images/)")
parser.add_argument("--download-images", action="store_true", help="Download and verify the fixed image set from GitHub Releases")
args = parser.parse_args()

rows, column = [], None
for path in args.csv:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in (reader.fieldnames or []) if c.endswith("_reviewed")]
        if len(columns) != 1 or not {"source", "caption_id", "caption"} <= set(reader.fieldnames or []):
            raise SystemExit("Each CSV needs source, caption_id, caption and exactly one *_reviewed column")
        if column is not None and column != columns[0]:
            raise SystemExit("All CSVs must use the same reviewed language column")
        column = columns[0]
        rows.extend(reader)
drawbench = json.loads((ASSETS / "drawbench.json").read_text())
captions = {}
seen, blank = set(), Counter()
filtered = 0
for row in rows:
    source, caption_id = row["source"], int(row["caption_id"])
    if source not in ("mscoco", "drawbench"):
        raise SystemExit(f"Unknown image source: {source}")
    key = (source, caption_id)
    if key in seen:
        raise SystemExit(f"Duplicate caption in CSV: {key}")
    seen.add(key)
    if not (row[column] or "").strip():
        blank[source] += 1
        continue
    if not (row["caption"] or "").strip():
        raise SystemExit(f"Missing English caption: {key}")
    if source == "drawbench":
        record = drawbench.get(str(caption_id))
        # Some reviewed CSVs retain this translation note. Ignore it only for
        # identity matching; keep the supplied caption unchanged for scoring.
        identity = row["caption"].strip().removeprefix("[Misspellings are intentional] ")
        if record is None or record["caption_en"].strip() != identity:
            raise SystemExit(f"Unknown or mismatched original DrawBench prompt: {caption_id}")
        images = record["images"]
        if not images:
            filtered += 1
    else:
        images = [dict(image_id=caption_id, path=f"coco/{caption_id:012d}.jpg")]
    for image in images:
        image_id = image["image_id"]
        if image_id in captions:
            raise SystemExit(f"Duplicate image ID: {image_id}")
        captions[image_id] = dict(row, relative_path=image["path"])
if blank:
    print(f"Excluded blank reviewed captions by source: {dict(blank)}")
print(f"Excluded {filtered} DrawBench prompts with SDXL score other than 2")
print(f"column {column}, {len(captions)} image-caption pairs")
expected_images = json.loads((ASSETS / "image-sha256.json").read_text())


def image_path(image_id):
    row = captions[image_id]
    path = args.images / row["relative_path"]
    candidates = [path]
    if row["source"] == "mscoco":
        candidates += [path.parent / folder / path.name for folder in ("train2017", "val2017")]
    matches = [p for p in candidates if p.is_file()]
    if len(matches) != 1:
        raise SystemExit(f"Expected one {row['relative_path']} in {args.images}; found {len(matches)}")
    if sha256(matches[0]) != expected_images[str(image_id)]:
        raise SystemExit(f"Wrong or damaged image: {matches[0]} (SHA256 mismatch)")
    return str(matches[0].resolve())


out = ROOT / "data" / args.lang
splits = json.loads((ASSETS / "split_ids.json").read_text())
all_ids = [i for ids in splits.values() for i in ids]
if len(all_ids) != len(set(all_ids)) or set(captions) - set(all_ids):
    raise SystemExit("Overlapping split IDs or CSV images outside the original splits")
if args.download_images:
    try:
        download_images(args.images, ASSETS)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
prepared = {}
for split, ids in splits.items():
    ids = [i for i in ids if i in captions]
    if not ids:
        raise SystemExit(f"No reviewed captions for {split}")
    prepared[split] = []
    for i in ids:
        prepared[split].append({"image_id": i, "image_path": image_path(i), "caption": captions[i][column],
                                "caption_en": captions[i]["caption"], "source": captions[i]["source"],
                                "caption_id": int(captions[i]["caption_id"])})

# Validate every image before replacing any prepared split.
out.mkdir(parents=True, exist_ok=True)
for split, records in prepared.items():
    with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{split}: {len(records)} {dict(Counter(r['source'] for r in records))}")
manifest = dict(csv=[dict(name=Path(p).name, sha256=sha256(p)) for p in args.csv],
                protocol="coco+drawbench-sdxl-score2", drawbench=sha256(ASSETS / "drawbench.json"),
                split_ids=sha256(ASSETS / "split_ids.json"),
                splits={split: sha256(out / f"{split}.jsonl") for split in splits},
                image_manifest=sha256(ASSETS / "image-sha256.json"))
(out / "sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

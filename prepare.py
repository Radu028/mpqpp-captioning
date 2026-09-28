import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.checks import sha256
from src.images import download_images
from src.prompts import LANGUAGES

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

def read_tsvs(folder, lang):
    if lang not in LANGUAGES:
        raise ValueError(f"No TSV files defined for language: {lang}")

    language = LANGUAGES[lang].lower()
    nested = folder / "final_translations"

    if nested.is_dir():
        if any(folder.glob(f"final_{language}_*.tsv")):
            raise ValueError(f"Keep the TSV files in one place: {folder} or {nested}")
        folder = nested

    rows = []
    paths = []

    for split in ("train", "val", "test"):
        suffix = "validation" if split == "val" else split
        path = folder / f"final_{language}_{suffix}.tsv"
        rows.extend(read_tsv(path, split))
        paths.append(path)

    return rows, paths


def read_tsv(path, split):
    rows = []

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"id", "caption", "final_translation"}
        columns = set(reader.fieldnames or [])

        if not required <= columns:
            raise ValueError(f"{path}: need id, caption and final_translation columns")

        for row in reader:
            caption_id = row["id"] or ""
            match = re.fullmatch(r"(cc|db)([0-9]{12})\.jpg", caption_id)
            if match is None:
                raise ValueError(f"{path.name}: invalid caption ID {row['id']!r}")

            translation = row["final_translation"] or ""
            if not translation.strip():
                raise ValueError(f"{path.name}: missing translation for {caption_id}")

            source = "mscoco" if match[1] == "cc" else "drawbench"
            caption = {
                "source": source,
                "caption_id": int(match[2]),
                "caption": row["caption"],
                "final_translation": translation,
                "input_split": split,
            }
            rows.append(caption)

    return rows


def read_csvs(paths):
    rows = []
    column = None

    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            reviewed = [name for name in fields if name.endswith("_reviewed")]
            required = {"source", "caption_id", "caption"}

            if len(reviewed) != 1 or not required <= fields:
                raise ValueError(
                    f"{path}: need source, caption_id, caption and one *_reviewed column"
                )
            if column is not None and column != reviewed[0]:
                raise ValueError("The CSVs use different reviewed language columns")

            column = reviewed[0]
            rows.extend(reader)

    return rows, column


def match_images(rows, column, drawbench):
    captions = {}
    seen = set()
    blank = Counter()
    filtered = 0

    for row in rows:
        source = row["source"]
        caption_id = int(row["caption_id"])
        if source not in ("mscoco", "drawbench"):
            raise ValueError(f"Unknown source: {source}")

        key = (source, caption_id)
        if key in seen:
            raise ValueError(f"Duplicate caption: {key}")
        seen.add(key)

        if not (row[column] or "").strip():
            blank[source] += 1
            continue
        if not (row["caption"] or "").strip():
            raise ValueError(f"Missing English caption: {key}")

        if source == "drawbench":
            record = drawbench.get(str(caption_id))
            # Ignore the note when matching. Keep it in the saved caption.
            text = row["caption"].strip().removeprefix("[Misspellings are intentional] ")
            if record is None or record["caption_en"].strip() != text:
                raise ValueError(f"DrawBench caption doesn't match: {caption_id}")

            images = record["images"]
            if not images:
                filtered += 1
        else:
            images = [{"image_id": caption_id, "path": f"coco/{caption_id:012d}.jpg"}]

        for image in images:
            image_id = image["image_id"]
            if image_id in captions:
                raise ValueError(f"Duplicate image: {image_id}")
            captions[image_id] = {**row, "relative_path": image["path"]}

    if blank:
        print(f"Skipped blank captions: {dict(blank)}")
    print(f"Skipped {filtered} DrawBench prompts with score != 2")
    print(f"{len(captions)} image-caption pairs from {column}")
    return captions


def image_path(folder, row, digest):
    path = folder / row["relative_path"]
    candidates = [path]
    if row["source"] == "mscoco":
        candidates += [path.parent / split / path.name for split in ("train2017", "val2017")]

    matches = [p for p in candidates if p.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Found {len(matches)} copies of {row['relative_path']} in {folder}, need one"
        )
    path = matches[0]
    if sha256(path) != digest:
        raise ValueError(f"SHA256 mismatch: {path}")

    return str(path.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", choices=LANGUAGES)
    parser.add_argument(
        "--inputs", nargs="+", type=Path, default=[ROOT / "data"],
        help="TSV folder or reviewed CSV files. Default: data/",
    )
    parser.add_argument("--images", default=ROOT / "images", type=Path, help="Default: images/")
    parser.add_argument("--download-images", action=argparse.BooleanOptionalAction,
                        default=True, help="Download missing images, default: yes")
    args = parser.parse_args()

    paths = args.inputs
    tsv_folder = len(paths) == 1 and paths[0].is_dir()
    if tsv_folder:
        rows, paths = read_tsvs(paths[0], args.lang)
        column = "final_translation"
    else:
        rows, column = read_csvs(paths)

    drawbench = json.loads((ASSETS / "drawbench.json").read_text())
    captions = match_images(rows, column, drawbench)
    hashes = json.loads((ASSETS / "image-sha256.json").read_text())
    splits = json.loads((ASSETS / "split_ids.json").read_text())

    all_ids = [image_id for ids in splits.values() for image_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Split IDs overlap")
    if set(captions) - set(all_ids):
        raise ValueError("The input has images outside the fixed splits")

    if tsv_folder:
        missing = set(all_ids) - set(captions)
        if missing:
            raise ValueError(f"Missing captions for {len(missing)} images in the fixed splits")

        for split, ids in splits.items():
            for image_id in ids:
                if captions[image_id]["input_split"] != split:
                    raise ValueError(f"Wrong input split for image {image_id}, expected {split}")

    if args.download_images:
        download_images(args.images, ASSETS)

    prepared = {}
    for split, ids in splits.items():
        records = []
        for image_id in ids:
            if image_id not in captions:
                continue

            row = captions[image_id]
            record = {
                "image_id": image_id,
                "image_path": image_path(args.images, row, hashes[str(image_id)]),
                "caption": row[column],
                "caption_en": row["caption"],
                "source": row["source"],
                "caption_id": int(row["caption_id"]),
            }
            records.append(record)

        if not records:
            raise ValueError(f"No reviewed captions for {split}")
        prepared[split] = records

    # Don't touch the old splits until all images pass the checks.
    out = ROOT / "data" / args.lang
    out.mkdir(parents=True, exist_ok=True)

    for split, records in prepared.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        counts = Counter(row["source"] for row in records)
        print(f"{split}: {len(records)} {dict(counts)}")

    input_key = "inputs" if tsv_folder else "csv"
    input_hashes = [{"name": path.name, "sha256": sha256(path)} for path in paths]
    manifest = {
        input_key: input_hashes,
        "protocol": "coco+drawbench-sdxl-score2",
        "drawbench": sha256(ASSETS / "drawbench.json"),
        "split_ids": sha256(ASSETS / "split_ids.json"),
        "splits": {split: sha256(out / f"{split}.jsonl") for split in splits},
        "image_manifest": sha256(ASSETS / "image-sha256.json"),
    }
    (out / "sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

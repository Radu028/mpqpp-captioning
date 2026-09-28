"""Small input checks before reusing a run or scoring predictions."""

import hashlib
import json
from pathlib import Path

BASELINE_SEED = 41


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["image_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty data or duplicate image IDs: {path}")
    return rows


def verify_data(root, language):
    assets = root / "assets"
    directory = root / "data" / language
    manifest = json.loads((directory / "sha256.json").read_text(encoding="utf-8"))
    if set(manifest["splits"]) != {"train", "val", "test"}:
        raise ValueError("Prepared data must contain train, val and test splits")
    if sha256(assets / "split_ids.json") != manifest["split_ids"]:
        raise ValueError("The original split list changed; prepare the data again")
    if sha256(assets / "drawbench.json") != manifest["drawbench"]:
        raise ValueError("The DrawBench mapping changed; prepare the data again")
    splits = json.loads((assets / "split_ids.json").read_text(encoding="utf-8"))
    images = json.loads((assets / "image-sha256.json").read_text(encoding="utf-8"))
    if sha256(assets / "image-sha256.json") != manifest["image_manifest"]:
        raise ValueError("The reference image checksums changed; prepare the data again")
    seen = set()
    for split, digest in manifest["splits"].items():
        path = directory / f"{split}.jsonl"
        if sha256(path) != digest:
            raise ValueError(f"Prepared {split} data changed; prepare the data again")
        for row in read_rows(path):
            image_id = row["image_id"]
            if image_id not in splits[split]:
                raise ValueError(f"Image {image_id} is not in the original {split} split")
            if image_id in seen:
                raise ValueError(f"Image {image_id} appears in multiple splits")
            seen.add(image_id)
            if sha256(row["image_path"]) != images[str(image_id)]:
                raise ValueError(f"Image content differs from the supplied image: {row['image_path']}")
    return manifest


def check_run(root, language, model, seeds, data, create=False):
    directory = root / "runs" / language / model
    record = directory / "sha256.json"
    code = [root / "src" / model / name for name in ("model.py", "train.py", "generate.py")]
    code += [root / "run.py", root / "src/checks.py", root / "requirements" / "torch.txt",
             root / "requirements" / f"{model}.txt"]
    current = dict(data=data, training_seeds=list(seeds), generation_seeds=list(seeds), evaluation_seeds=list(seeds),
                   zero_shot_generation_seed=BASELINE_SEED, zero_shot_evaluation_seed=BASELINE_SEED,
                   code={str(p.relative_to(root)): sha256(p) for p in code})
    if model == "centurio":
        from src.prompts import caption_prompt
        current["code"]["src/prompts.py"] = sha256(root / "src/prompts.py")
        current["prompt"] = sha256(root / "prompts" / f"{language}.txt")
        current["prompt_text"] = caption_prompt(root / "prompts" / f"{language}.txt")
    if record.exists():
        previous = json.loads(record.read_text(encoding="utf-8"))
        if previous != current:
            changed = [key for key in current if current[key] != previous.get(key)]
            raise ValueError(f"Inputs changed for {model}: {', '.join(changed)}. "
                             "Keep the old runs folder separately before starting a new experiment; nothing was deleted.")
    elif not create or (directory.exists() and any(directory.iterdir())) or any(
        (root / "runs" / language / "preds").glob(f"{model}-*.jsonl")
    ):
        raise ValueError(f"Cannot verify existing {model} results without {record}; keep them separately")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        with record.open("x", encoding="utf-8") as stream:
            json.dump(current, stream, indent=2)


def predictions_for(path, samples):
    rows = {row["image_id"]: row for row in read_rows(path)}
    if set(rows) != {row["image_id"] for row in samples}:
        raise ValueError(f"Missing or extra prediction IDs: {path}")
    for sample in samples:
        row = rows[sample["image_id"]]
        if row["reference"] != sample["caption"] or not isinstance(row["prediction"], str):
            raise ValueError(f"Wrong reference or invalid prediction in {path}, image {sample['image_id']}")
    return [rows[row["image_id"]]["prediction"] for row in samples]

"""Score predictions and write results.csv; run.py packages the closed logs."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from run import BASELINE_SEED, SEEDS
from .checks import check_run, predictions_for, read_rows, verify_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lang", required=True)
parser.add_argument("--models", nargs="+", choices=["blip2", "centurio"], default=["blip2", "centurio"])
args = parser.parse_args()
models = [model for model in ("blip2", "centurio") if model in args.models]
suffix = "" if len(models) == 2 else f"-{models[0]}"

root = Path(__file__).resolve().parents[1]
data_checks = verify_data(root, args.lang)
for model in models:
    check_run(root, args.lang, model, SEEDS, data_checks)
test = read_rows(root / "data" / args.lang / "test.jsonl")
refs = [r["caption"] for r in test]
sources = [r["caption_en"] for r in test]
runs = root / "runs" / args.lang
paths = [runs / "preds" / f"{model}-{kind}-s{seed}.jsonl"
         for model in models for kind in ("zero", "lora")
         for seed in ((BASELINE_SEED,) if kind == "zero" else SEEDS)]
missing = [str(path) for path in paths if not path.is_file()]
if missing:
    raise SystemExit("Missing predictions for selected models:\n" + "\n".join(missing))
hypotheses = {path: predictions_for(path, test) for path in paths}

import sacrebleu
from bert_score import BERTScorer
from comet import download_model, load_from_checkpoint
from transformers import set_seed

bert = BERTScorer(model_type="bert-base-multilingual-cased", num_layers=9, batch_size=64, device="cuda")
comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

results = []
for path in paths:
    model, kind, seed = path.stem.split("-")  # centurio-lora-s43
    hyps = hypotheses[path]

    set_seed(int(seed[1:]))
    bert_values = bert.score(hyps, refs)[2].cpu().tolist()
    comet_values = comet.predict([{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)],
                                 batch_size=32, gpus=1, progress_bar=False).scores
    for subset in ("all", "mscoco", "drawbench"):
        indices = [i for i, row in enumerate(test) if subset == "all" or row["source"] == subset]
        if not indices:
            continue
        scores = {
            "bleu": sacrebleu.corpus_bleu([hyps[i] for i in indices], [[refs[i] for i in indices]], tokenize="13a").score,
            "chrf++": sacrebleu.corpus_chrf([hyps[i] for i in indices], [[refs[i] for i in indices]], word_order=2).score,
            "bertscore": statistics.mean(bert_values[i] for i in indices),
            "comet": statistics.mean(float(comet_values[i]) for i in indices),
        }
        print(path.stem, subset, {k: round(v, 4) for k, v in scores.items()}, flush=True)
        results.append({"model": model, "kind": kind, "subset": subset, "count": len(indices),
                        "training_seed": seed[1:] if kind == "lora" else "",
                        "generation_seed": seed[1:], "evaluation_seed": seed[1:], **scores})

result_file = runs / f"results{suffix}.csv"
with open(result_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

print(f"\n{args.lang}, {len(test)} test images; one zero-shot baseline, fine-tuned mean ± std over seeds")
for model in sorted({r["model"] for r in results}):
    for subset, kind in ((s, k) for s in ("all", "mscoco", "drawbench") for k in ("zero", "lora")):
        rows = [r for r in results if r["model"] == model and r["kind"] == kind and r["subset"] == subset]
        if not rows:
            continue
        line = []
        for metric in ("bleu", "chrf++", "bertscore", "comet"):
            values = [r[metric] for r in rows]
            summary = f"{metric} {statistics.mean(values):.4f}"
            if len(values) > 1:
                summary += f" ± {statistics.stdev(values):.4f}"
            line.append(summary)
        print(f"{model:8s} {kind:4s} {subset:9s} (n={len(rows)})  " + "   ".join(line))

print(f"\nsaved {result_file}", flush=True)

import argparse
import csv
import statistics
from pathlib import Path

from run import BASELINE_SEED, SEEDS
from .checks import check_run, predictions_for, read_rows, verify_data
from .report import render_report

parser = argparse.ArgumentParser()
parser.add_argument("--lang", required=True)
parser.add_argument(
    "--models", nargs="+", choices=["blip2", "centurio"], default=["blip2", "centurio"]
)
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
paths = []
for model in models:
    paths.append(runs / "preds" / f"{model}-zero-s{BASELINE_SEED}.jsonl")
    for seed in SEEDS:
        paths.append(runs / "preds" / f"{model}-lora-s{seed}.jsonl")

missing = [str(path) for path in paths if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing predictions:\n" + "\n".join(missing))
predictions = {path: predictions_for(path, test) for path in paths}

import sacrebleu
from bert_score import BERTScorer
from comet import download_model, load_from_checkpoint
from transformers import set_seed

bert = BERTScorer(
    model_type="bert-base-multilingual-cased", num_layers=9, batch_size=64, device="cuda"
)
comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

results = []
for path in paths:
    model, kind, seed = path.stem.split("-")  # centurio-lora-s43
    hyps = predictions[path]

    set_seed(int(seed[1:]))
    bert_values = bert.score(hyps, refs)[2].cpu().tolist()

    comet_input = []
    for source, caption, reference in zip(sources, hyps, refs):
        comet_input.append({"src": source, "mt": caption, "ref": reference})
    comet_values = comet.predict(comet_input, batch_size=32, gpus=1, progress_bar=False).scores

    for subset in ("all", "mscoco", "drawbench"):
        indices = [
            i for i, row in enumerate(test)
            if subset == "all" or row["source"] == subset
        ]
        if not indices:
            continue

        captions = [hyps[i] for i in indices]
        references = [refs[i] for i in indices]
        scores = {
            "bleu": sacrebleu.corpus_bleu(captions, [references], tokenize="13a").score,
            "chrf++": sacrebleu.corpus_chrf(captions, [references], word_order=2).score,
            "bertscore": statistics.mean(bert_values[i] for i in indices),
            "comet": statistics.mean(float(comet_values[i]) for i in indices),
        }
        print(path.stem, subset, {k: round(v, 4) for k, v in scores.items()}, flush=True)

        result = {
            "model": model,
            "kind": kind,
            "subset": subset,
            "count": len(indices),
            "training_seed": seed[1:] if kind == "lora" else "",
            "generation_seed": seed[1:],
            "evaluation_seed": seed[1:],
            **scores,
        }
        results.append(result)

result_file = runs / f"results{suffix}.csv"
with open(result_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

report_file = runs / f"results{suffix}.md"
report = render_report(args.lang, results)
report_file.write_text(report, encoding="utf-8")
print(report)
print(f"\nsaved {result_file}", flush=True)
print(f"saved {report_file}", flush=True)

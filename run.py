"""One zero-shot baseline, then LoRA training and captions for seeds 41, 43, 44.

Rerun the same command after a failure, finished steps are skipped.
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from src.checks import BASELINE_SEED, check_run, verify_data

ROOT = Path(__file__).resolve().parent
SEEDS = (41, 43, 44)


def venv_python(name):
    env = ROOT / f".venv-{name}"
    return env / "Scripts" / "python.exe" if os.name == "nt" else env / "bin" / "python"


def step(title, command, log):
    print(f"\n== {title}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TQDM_DISABLE="1")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n== {title}\n")
        process = subprocess.Popen([str(c) for c in command], cwd=ROOT, env=env, text=True, encoding="utf-8",
                                   errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout:
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
        if process.wait():
            sys.exit(f"{title} failed, see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--models", nargs="+", choices=["blip2", "centurio"], default=["blip2", "centurio"],
                        help="Run only the selected models; default: both")
    args = parser.parse_args()
    models = [model for model in ("blip2", "centurio") if model in args.models]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Use only the first available GPU, respecting an existing allocation.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    data = ROOT / "data" / args.lang
    prompt = ROOT / "prompts" / f"{args.lang}.txt"
    runs = ROOT / "runs" / args.lang
    if not data.exists():
        sys.exit(f"{data} is missing, run prepare.py first")
    if "centurio" in models and not prompt.exists():
        sys.exit(f"{prompt} is missing")
    print("Checking data and image SHA256...", flush=True)
    data_checks = verify_data(ROOT, args.lang)
    for model in models:
        check_run(ROOT, args.lang, model, SEEDS, data_checks, create=True)
    (runs / "preds").mkdir(parents=True, exist_ok=True)
    (runs / "logs").mkdir(exist_ok=True)

    for model in models:
        python = venv_python(model)
        extra = ["--prompt", prompt] if model == "centurio" else []
        zero = runs / "preds" / f"{model}-zero-s{BASELINE_SEED}.jsonl"
        if not zero.exists():
            step(f"{model} zero-shot baseline", [python, "-m", f"src.{model}.generate",
                 "--data", data, *extra, "--seed", BASELINE_SEED, "--out", zero],
                 runs / "logs" / f"{model}-zero.log")
        for seed in SEEDS:
            out = runs / model / f"seed{seed}"
            lora = runs / "preds" / f"{model}-lora-s{seed}.jsonl"
            log = runs / "logs" / f"{model}-s{seed}.log"
            common = ["--data", data, *extra, "--seed", seed]

            if not (out / "done").exists():
                out.mkdir(parents=True, exist_ok=True)
                step(f"{model} {seed} training", [python, "-m", f"src.{model}.train", *common, "--out", out], log)
                (out / "done").touch()
            if not lora.exists():
                step(f"{model} {seed} fine-tuned",
                     [python, "-m", f"src.{model}.generate", *common, "--adapter", out / "best", "--out", lora], log)

    suffix = "" if len(models) == 2 else f"-{models[0]}"
    step("scoring", [venv_python("eval"), "-m", "src.evaluate", "--lang", args.lang, "--models", *models],
         runs / "logs" / f"evaluate{suffix}.log")
    print(f"\nsaved {pack_results(args.lang, models)}", flush=True)


def pack_results(language, models):
    """Archive results only after scoring and its log have both closed."""
    runs = ROOT / "runs" / language
    suffix = "" if len(models) == 2 else f"-{models[0]}"
    archive = ROOT / f"results-{language}{suffix}.zip"
    files = [runs / f"results{suffix}.csv", runs / "logs" / f"evaluate{suffix}.log"]
    for model in models:
        files.append(runs / model / "sha256.json")
        files.extend([runs / "preds" / f"{model}-zero-s{BASELINE_SEED}.jsonl",
                      runs / "logs" / f"{model}-zero.log"])
        for seed in SEEDS:
            files.append(runs / "preds" / f"{model}-lora-s{seed}.jsonl")
            files.extend([runs / "logs" / f"{model}-s{seed}.log", runs / model / f"seed{seed}" / "best.json"])
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "data" / language / "sha256.json", "data-sha256.json")
        for name in ("src/evaluate.py", "requirements/eval.txt"):
            z.write(ROOT / name, f"scoring-code/{name}")
        for path in files:
            z.write(path, path.relative_to(runs))
    temporary.replace(archive)
    return archive


if __name__ == "__main__":
    main()

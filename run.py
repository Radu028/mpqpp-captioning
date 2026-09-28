import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from src.checks import BASELINE_SEED, check_run, verify_data
from src.prompts import LANGUAGES

ROOT = Path(__file__).resolve().parent
SEEDS = (41, 43, 44)


def venv_python(name):
    env = ROOT / f".venv-{name}"
    if os.name == "nt":
        return env / "Scripts" / "python.exe"
    return env / "bin" / "python"


def step(title, command, log):
    print(f"\n{title}", flush=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TQDM_DISABLE="1")

    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{title}\n")
        process = subprocess.Popen(
            [str(c) for c in command],
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout:
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)

        if process.wait():
            sys.exit(f"{title} failed, see {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", choices=LANGUAGES)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["blip2", "centurio"],
        default=["blip2", "centurio"],
        help="Models to run, default: both",
    )
    args = parser.parse_args()
    models = [model for model in ("blip2", "centurio") if model in args.models]

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # One GPU, even if the machine has more.
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus.split(",")[0]

    data = ROOT / "data" / args.lang
    runs = ROOT / "runs" / args.lang

    if not data.exists():
        sys.exit(f"{data} is missing, run prepare.py first")

    print("Checking captions and images...", flush=True)
    data_checks = verify_data(ROOT, args.lang)
    for model in models:
        check_run(ROOT, args.lang, model, SEEDS, data_checks, create=True)

    (runs / "preds").mkdir(parents=True, exist_ok=True)
    (runs / "logs").mkdir(exist_ok=True)

    for model in models:
        python = venv_python(model)
        inputs = ["--data", data]
        if model == "centurio":
            inputs += ["--lang", args.lang]

        generate = [python, "-m", f"src.{model}.generate", *inputs]
        train = [python, "-m", f"src.{model}.train", *inputs]

        zero = runs / "preds" / f"{model}-zero-s{BASELINE_SEED}.jsonl"
        if not zero.exists():
            command = generate + ["--seed", BASELINE_SEED, "--out", zero]
            step(f"{model}: zero-shot", command, runs / "logs" / f"{model}-zero.log")

        for seed in SEEDS:
            out = runs / model / f"seed{seed}"
            lora = runs / "preds" / f"{model}-lora-s{seed}.jsonl"
            log = runs / "logs" / f"{model}-s{seed}.log"

            if not (out / "done").exists():
                out.mkdir(parents=True, exist_ok=True)
                command = train + ["--seed", seed, "--out", out]
                step(f"{model}: train seed {seed}", command, log)
                (out / "done").touch()

            if not lora.exists():
                command = generate + ["--seed", seed, "--adapter", out / "best", "--out", lora]
                step(f"{model}: captions for seed {seed}", command, log)

    suffix = "" if len(models) == 2 else f"-{models[0]}"
    step(
        "scoring",
        [venv_python("eval"), "-m", "src.evaluate", "--lang", args.lang, "--models", *models],
        runs / "logs" / f"evaluate{suffix}.log",
    )
    print(f"\nsaved {pack_results(args.lang, models)}", flush=True)


def pack_results(language, models):
    runs = ROOT / "runs" / language
    suffix = "" if len(models) == 2 else f"-{models[0]}"
    archive = ROOT / f"results-{language}{suffix}.zip"

    files = [
        runs / f"results{suffix}.csv",
        runs / f"results{suffix}.md",
        runs / "logs" / f"evaluate{suffix}.log",
    ]
    for model in models:
        files += [
            runs / model / "sha256.json",
            runs / "preds" / f"{model}-zero-s{BASELINE_SEED}.jsonl",
            runs / "logs" / f"{model}-zero.log",
        ]
        for seed in SEEDS:
            files += [
                runs / "preds" / f"{model}-lora-s{seed}.jsonl",
                runs / "logs" / f"{model}-s{seed}.log",
                runs / model / f"seed{seed}" / "best.json",
            ]

    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "data" / language / "sha256.json", "data-sha256.json")
        for name in ("src/evaluate.py", "src/report.py", "requirements/eval.txt"):
            z.write(ROOT / name, f"scoring-code/{name}")
        for path in files:
            z.write(path, path.relative_to(runs))
    temporary.replace(archive)
    return archive


if __name__ == "__main__":
    main()

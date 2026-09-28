# Multilingual PQPP image captioning

Compare zero-shot and LoRA fine-tuned BLIP-2 and Centurio in your language.
The examples use Italian. Replace only `it` with your language code from the table.

## 1. Setup

Requirements: Python 3.11, git, one NVIDIA GPU with 32 GB VRAM and about
100 GB of free disk.

```bash
git clone https://github.com/Radu028/mpqpp-captioning.git
cd mpqpp-captioning
python3.11 install.py
```

Run the remaining commands from this folder. You don't need to activate the
virtual environments yourself. On Windows, use `py -3.11`.

## 2. Prepare the data

Download `final_translations.zip` from the Google Drive link shared with the
team. Unzip it, then put the extracted `final_translations` folder in `data/`:

```text
data/final_translations/final_italian_train.tsv
data/final_translations/final_italian_validation.tsv
data/final_translations/final_italian_test.tsv
```

Keep all three files for your language unchanged. Together they contain
10,200 captions. You can leave the other languages in the same folder.

| Language | Code |
|----------|------|
| Danish | `da` |
| French | `fr` |
| Hindi | `hi` |
| Italian | `it` |

```bash
python3.11 prepare.py it
```

This downloads the 1.9 GB image bundle, checks it and creates the fixed splits
in `data/it/`. You don't need to download images or split the data yourself.
It reads only your language and keeps 10,000 COCO images and 188 DrawBench
SDXL images with score 2.

Expected counts: **train 6,080 / validation 2,034 / test 2,074**.
If they differ or captions are blank, check with Radu Popa before training.

## 3. Run

```bash
python3.11 run.py it
```

The language code selects the files and Centurio prompt automatically.
For `it`, the prompt is `Briefly describe the image in Italian.`
The instruction stays in English. BLIP-2 has no text prompt.

Runs both models on one GPU, including generation and scoring:

- One zero-shot run per model, generation/evaluation seed 41, greedy decoding
  (no random sampling, so the same model and inputs are expected to produce the
  same captions regardless of the generation seed).
- Fine-tuning with training seeds 41, 43 and 44, each with the matching
  generation/evaluation seed. BLIP-2 uses rank 16 and Centurio uses rank 8.

BLIP-2 zero-shot generates English captions, scored against your translations.

To run only one model, add `--models blip2` or `--models centurio`.
If a run stops, run the same command again. It skips finished steps.
BLIP-2 restarts unfinished training, while Centurio resumes from its checkpoint.
If you change the TSV files, run `prepare.py` again. SHA256 checks prevent reusing
results from different inputs.

## 4. Results

Open `runs/it/results.md` for zero-shot scores, fine-tuned mean ± std and
deltas, using BLEU, chrF++, BERTScore and COMET. Per-seed scores are in
`runs/it/results.csv`.

Send **`results-it.zip`** privately to the team. It contains the reports,
predictions and translated test references. Keep `runs/` until we confirm we've received everything.

With `--models centurio`, the files are named `results-it-centurio.zip`,
`results-centurio.md` and `results-centurio.csv`. BLIP-2 follows the same naming pattern.

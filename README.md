# Multilingual PQPP captioning instructions

**Deliverable: `results-<lang>.zip`**

| Model    | Checkpoint                     | LoRA on                                 |
|----------|--------------------------------|-----------------------------------------|
| BLIP-2   | Salesforce/blip2-opt-6.7b-coco | q/v of the language model               |
| Centurio | WueNLP/centurio_qwen           | attention and MLP of the language model |

Both models caption the same test images. Their captions are scored
against your reviewed captions with BLEU, chrF++, BERTScore and COMET.

In this doc Italian is the running example. Replace `review_italian_final.csv`,
`it` and `Italian` with your own file, language code and language name.

## The two configurations

Every model runs in two configurations. The comparison between them is the
result of the study.

|   | Model | Additional fine-tuning | What it measures |
|---|-------|------------|------------------|
| 1 | original | none | **zero-shot**: how well the model captions in your language out of the box |
| 2 | LoRA fine-tuned | your reviewed captions | **fine-tuned**: how much your reviewed captions help |

Configuration 1 runs once per model and language, with generation/evaluation
seed 41. Decoding is greedy, without random sampling. Configuration 2 uses
training seeds 41, 43 and 44, with matching generation/evaluation seeds. All
three adapters are compared against the same zero-shot baseline.

**Already fixed (language-independent):**

- the train / validation / test split from the Romanian experiment
- the images: 10000 from COCO and 188 from DrawBench (two SDXL images for each prompt with score 2)
- the training settings at the end of this doc

You need one NVIDIA GPU with 32 GB (we used an RTX 5090), Python 3.11, git and
about 100 GB of free disk.

## Steps

Run every command from the `mpqpp-captioning` folder, in this order. On Windows
write `py -3.11` instead of `python3.11`.

### 1. Setup

```
git clone https://github.com/Radu028/mpqpp-captioning.git
cd mpqpp-captioning
python3.11 install.py
```

This creates one virtual environment per model and one for scoring, with the
CUDA 12.8 build of PyTorch.

### 2. Prepare your captions and images

You do **not** have to split your captions into train / validation / test. One
file holding all of them is the normal case; the script works out which split
each image belongs to. Copy your CSV into the `mpqpp-captioning` folder, then:

```
python3.11 prepare.py review_italian_final.csv --lang it --download-images
```

It prints how many images ended up in each split. The first run also downloads
the images (1.9 GB) and checks them; later runs skip the download.
Only images with non-empty reviewed captions in your CSV are included, so
counts can differ between languages. Their original split assignments stay fixed.

**What the file needs.** Use the CSV we sent you as it is. It has:

- `source` and `caption_id`, which identify the image
- `caption`, the original English caption
- exactly one `caption_<language>_reviewed` column, for example `caption_italian_reviewed`

### 3. Centurio instruction

Centurio gets the same English instruction for every language:

```
<image_placeholder>
Briefly describe the image in Italian.
```

`prompts/it.txt` holds just the language name, `Italian`. Danish, Hindi, Italian
and Romanian are already there. For another language, create
`prompts/<your code>.txt` with its English name in one word, for example
`German`. Keep the instruction in English; change only the language name.

### 4. Run

```
python3.11 run.py --lang it
```

For each model this captions the test images once with the original model, then
fine-tunes it with seeds 41, 43 and 44 and captions the test images after each
run. At the end it scores everything and writes `results-it.zip`.

Runtime depends on your GPU and language. On a remote Linux server, run inside
tmux or screen so disconnecting does not stop it. If it stops, run the same
command again; finished steps are skipped. An interrupted BLIP-2 training run
restarts that seed; Centurio resumes from its latest checkpoint.

To run only one model, add `--models centurio` or `--models blip2`.
If you change the captions, run `prepare.py` again. The SHA256 checks reject
reuse of results when the prepared data, images, prompt or model code changes.

### 5. Scores

The scores are at the end of `runs/it/logs/evaluate.log`: one zero-shot score
and fine-tuned mean ± std across three seeds, for all test images and separately for COCO and
DrawBench.

BLIP-2 zero-shot captions are in English; they are still scored against your
target-language references.

## What to send back

`mpqpp-captioning/results-it.zip`. Keep the `runs/` folder until we confirm we
have everything.

With `--models`, the ZIP and scoring log include the selected model in their
names, for example `results-it-centurio.zip` and `evaluate-centurio.log`.

## Training settings

Please don't change these, otherwise the languages can't be compared.

|               | BLIP-2                       | Centurio                                       |
|---------------|------------------------------|------------------------------------------------|
| LoRA          | rank 16, alpha 32, q/v       | rank 8, alpha 16, attention + MLP              |
| Batch         | 4                            | 1, gradient accumulation 2                     |
| Learning rate | 1e-4, cosine, 5% warmup      | 1e-4, cosine, 10% warmup                       |
| Epochs        | up to 15, early stop after 3 | 5                                              |
| Kept adapter  | lowest validation loss       | lowest validation loss                         |
| Decoding      | greedy, max 128 tokens       | greedy, max 128 tokens, repetition penalty 1.1 |

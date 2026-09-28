"""CPU checks for changed inputs, image identity and prediction completeness."""
import csv
import json
import runpy
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.checks import check_run, predictions_for, read_rows, sha256, verify_data
from src.prompts import caption_prompt

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (41, 43, 44)


class InputChecks(unittest.TestCase):
    def test_fixed_english_instruction_changes_only_target_language(self):
        for code, language in (("ro", "Romanian"), ("it", "Italian"), ("da", "Danish"), ("hi", "Hindi")):
            self.assertEqual(caption_prompt(ROOT / "prompts" / f"{code}.txt"),
                             f"<image_placeholder>\nBriefly describe the image in {language}.")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "assets").mkdir()
        self.data = self.root / "data/it"
        self.data.mkdir(parents=True)
        self.images = self.root / "images/coco"
        self.images.mkdir(parents=True)
        digests = {}
        for i, split in enumerate(("train", "val", "test"), 1):
            image = self.images / f"{i:012d}.jpg"
            image.write_bytes(f"reference image {i}".encode())
            digests[str(i)] = sha256(image)
            row = dict(image_id=i, caption=f"Reviewed {i}", caption_en=f"English {i}", image_path=str(image))
            (self.data / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        (self.root / "assets/split_ids.json").write_text(json.dumps(dict(train=[1], val=[2], test=[3])))
        (self.root / "assets/image-sha256.json").write_text(json.dumps(digests))
        (self.root / "assets/drawbench.json").write_text("{}")
        self.manifest = dict(csv="source-hash", split_ids=sha256(self.root / "assets/split_ids.json"),
            drawbench=sha256(self.root / "assets/drawbench.json"),
            image_manifest=sha256(self.root / "assets/image-sha256.json"),
            splits={s: sha256(self.data / f"{s}.jsonl") for s in ("train", "val", "test")})
        (self.data / "sha256.json").write_text(json.dumps(self.manifest))
        for model in ("blip2", "centurio"):
            for name in ("model.py", "train.py", "generate.py"):
                path = self.root / "src" / model / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# unchanged code\n")
        (self.root / "requirements").mkdir()
        for name in ("torch", "blip2", "centurio"):
            (self.root / f"requirements/{name}.txt").write_text("pinned dependencies")
        for name in ("run.py", "src/checks.py", "src/prompts.py"):
            (self.root / name).write_text("# unchanged runner\n")
        (self.root / "prompts").mkdir()
        (self.root / "prompts/it.txt").write_text("Italian")

    def test_unchanged_inputs_can_be_reused(self):
        self.assertEqual(verify_data(self.root, "it"), self.manifest)
        check_run(self.root, "it", "blip2", SEEDS, self.manifest, create=True)
        check_run(self.root, "it", "blip2", SEEDS, self.manifest)

    def test_data_and_image_changes_are_rejected(self):
        image = self.images / "000000000001.jpg"
        image.write_bytes(b"different image under the same filename")
        with self.assertRaisesRegex(ValueError, "Image content differs"):
            verify_data(self.root, "it")
        with (self.data / "train.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Prepared train data changed"):
            verify_data(self.root, "it")

    def test_changed_model_prompt_or_reprepared_data_preserves_original_record(self):
        check_run(self.root, "it", "centurio", SEEDS, self.manifest, create=True)
        record = self.root / "runs/it/centurio/sha256.json"
        original = record.read_bytes()
        prompt = self.root / "prompts/it.txt"
        prompt.write_text("changed prompt")
        with self.assertRaisesRegex(ValueError, "prompt"):
            check_run(self.root, "it", "centurio", SEEDS, self.manifest)
        prompt.write_text("Italian")
        with self.assertRaisesRegex(ValueError, "data"):
            check_run(self.root, "it", "centurio", SEEDS, dict(self.manifest, csv="new source"))
        (self.root / "src/centurio/train.py").write_text("changed recipe")
        with self.assertRaisesRegex(ValueError, "code"):
            check_run(self.root, "it", "centurio", SEEDS, self.manifest)
        self.assertEqual(record.read_bytes(), original)

    def test_results_without_a_record_are_not_adopted(self):
        directory = self.root / "runs/it/preds"
        directory.mkdir(parents=True)
        (directory / "blip2-zero-s41.jsonl").write_text("old result")
        with self.assertRaisesRegex(ValueError, "Cannot verify"):
            check_run(self.root, "it", "blip2", SEEDS, self.manifest, create=True)

    def test_predictions_require_exact_ids_and_references(self):
        samples = read_rows(self.data / "test.jsonl")
        path = self.root / "predictions.jsonl"
        row = dict(image_id=3, reference="Reviewed 3", prediction="")
        path.write_text(json.dumps(row) + "\n")
        self.assertEqual(predictions_for(path, samples), [""])
        path.write_text(json.dumps(dict(row, reference="Old caption")) + "\n")
        with self.assertRaisesRegex(ValueError, "Wrong reference"):
            predictions_for(path, samples)
        path.write_text((json.dumps(row) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            predictions_for(path, samples)
        path.write_text(json.dumps(dict(row, image_id=99)) + "\n")
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            predictions_for(path, samples)

    def test_prepare_includes_four_drawbench_images_and_checks_before_writing(self):
        splits = dict(train=[1, 1000000, 1000001, 1000002, 1000003], val=[2], test=[3])
        (self.root / "assets/split_ids.json").write_text(json.dumps(splits))
        digests = json.loads((self.root / "assets/image-sha256.json").read_text())
        draw_images = []
        for number in (4, 5, 6, 7):
            path = self.root / f"images/drawbench/0/image_{number}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"original DrawBench {number}".encode())
            image_id = 1000000 + number - 4
            digests[str(image_id)] = sha256(path)
            draw_images.append(dict(image_id=image_id, path=str(path.relative_to(self.root / "images"))))
        (self.root / "assets/image-sha256.json").write_text(json.dumps(digests))
        (self.root / "assets/drawbench.json").write_text(json.dumps({"0": dict(caption_en="DrawBench source", images=draw_images)}))
        csv_file = self.root / "reviewed.csv"
        with csv_file.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["source", "caption_id", "caption", "caption_italian_reviewed"])
            writer.writeheader()
            for i in (1, 2, 3):
                writer.writerow(dict(source="mscoco", caption_id=i, caption=f"English {i}", caption_italian_reviewed=f"Reviewed {i}"))
            writer.writerow(dict(source="drawbench", caption_id=0, caption="[Misspellings are intentional] DrawBench source", caption_italian_reviewed="Reviewed prompt"))
        code = compile((ROOT / "prepare.py").read_text(), str(ROOT / "prepare.py"), "exec")
        def prepare():
            with patch.object(sys, "argv", ["prepare.py", str(csv_file), "--lang", "it"]):
                exec(code, {"__file__": str(self.root / "prepare.py"), "__name__": "__main__"})
        prepare()
        verify_data(self.root, "it")
        train = read_rows(self.data / "train.jsonl")
        self.assertEqual([r["image_id"] for r in train], splits["train"])
        self.assertEqual([r["caption"] for r in train[1:]], ["Reviewed prompt"] * 4)
        self.assertEqual(train[1]["caption_en"], "[Misspellings are intentional] DrawBench source")
        original = (self.data / "train.jsonl").read_bytes()
        (self.root / "images/drawbench/0/image_7.png").write_bytes(b"wrong bytes")
        with self.assertRaisesRegex(SystemExit, "SHA256 mismatch"):
            prepare()
        self.assertEqual((self.data / "train.jsonl").read_bytes(), original)

    def test_shipped_drawbench_mapping_keeps_only_sdxl_score_two(self):
        splits = json.loads((ROOT / "assets/split_ids.json").read_text())
        images = json.loads((ROOT / "assets/image-sha256.json").read_text())
        draw = json.loads((ROOT / "assets/drawbench.json").read_text())
        self.assertEqual({k: len(v) for k, v in splits.items()}, dict(train=6080, val=2034, test=2074))
        ids = [i for values in splits.values() for i in values]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(map(str, ids)), set(images))
        self.assertEqual(set(draw), set(map(str, range(200))))
        with (ROOT / "assets/drawbench-sdxl-scores.csv").open(newline="") as stream:
            scores = {r["Prompts"].strip(): float(r["score"]) for r in csv.DictReader(stream)}
        selected = 0
        for prompt, record in draw.items():
            score = scores[record["caption_en"].strip()]
            self.assertEqual(record["sdxl_score"], score)
            group = {r["image_id"] for r in record["images"]}
            if score != 2:
                self.assertEqual(group, set())
                continue
            selected += 1
            self.assertEqual(group, {1000000 + int(prompt) * 4 + j for j in (0, 1)})
            self.assertEqual(sum(group <= set(values) for values in splits.values()), 1)
            self.assertEqual({r["path"] for r in record["images"]},
                             {f"drawbench/{prompt}/image_{j}.png" for j in (4, 5)})
        self.assertEqual(selected, 94)

    def test_runner_checks_inputs_before_launching_and_keeps_one_gpu(self):
        import os
        namespace = runpy.run_path(str(ROOT / "run.py"))
        main = namespace["main"]
        main.__globals__["ROOT"] = self.root
        calls = []
        main.__globals__["step"] = lambda title, command, log: calls.append((title, os.environ["CUDA_VISIBLE_DEVICES"]))
        main.__globals__["pack_results"] = lambda language, models: "test-results.zip"
        with patch.object(sys, "argv", ["run.py", "--lang", "it", "--models", "blip2"]), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
            main()
            self.assertEqual(len(calls), 8)
            self.assertEqual(sum("zero-shot" in title for title, _ in calls), 1)
            self.assertTrue(all(gpu == "2" for _, gpu in calls))
            calls.clear()
            (self.root / "src/blip2/train.py").write_text("changed recipe")
            with self.assertRaisesRegex(ValueError, "code"):
                main()
            self.assertEqual(calls, [])

    def test_result_archive_includes_closed_scoring_log_and_only_selected_model(self):
        import run
        runs = self.root / "runs/it"
        files = ["results-blip2.csv", "blip2/sha256.json", "preds/blip2-zero-s41.jsonl", "logs/blip2-zero.log"]
        for seed in SEEDS:
            files += [f"preds/blip2-lora-s{seed}.jsonl"]
            files += [f"logs/blip2-s{seed}.log", f"blip2/seed{seed}/best.json"]
        for name in files:
            path = runs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        (self.root / "src/evaluate.py").write_text("# scorer\n")
        (self.root / "requirements/eval.txt").write_text("# requirements\n")
        summary = "mean ± std: BLEU 9.5 ± 0.2"
        log = runs / "logs/evaluate-blip2.log"
        with patch.object(run, "ROOT", self.root):
            run.step("scoring", [sys.executable, "-c", f"print({summary!r})"], log)
            archive = run.pack_results("it", ["blip2"])
        self.assertEqual(archive.name, "results-it-blip2.zip")
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(z.read("logs/evaluate-blip2.log"), log.read_bytes())
            self.assertIn(summary, z.read("logs/evaluate-blip2.log").decode())
            self.assertIn("results-blip2.csv", z.namelist())
            self.assertFalse(any("centurio" in name for name in z.namelist()))
            self.assertEqual([n for n in z.namelist() if "-zero-s" in n], ["preds/blip2-zero-s41.jsonl"])


if __name__ == "__main__":
    unittest.main()

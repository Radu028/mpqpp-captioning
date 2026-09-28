import unittest

from src.report import render_report


class ReportTests(unittest.TestCase):
    def test_mean_std_and_delta(self):
        rows = []
        for kind, seed, value in (
            ("zero", "", 5),
            ("lora", "41", 2),
            ("lora", "43", 4),
            ("lora", "44", 6),
        ):
            rows.append(
                {
                    "model": "centurio",
                    "kind": kind,
                    "training_seed": seed,
                    "subset": "all",
                    "count": 10,
                    "bleu": value,
                    "chrf++": value,
                    "bertscore": value / 10,
                    "comet": value / 10,
                }
            )
        report = render_report("it", rows)
        self.assertIn("| Centurio | BLEU | 5.0000 | 4.0000 ± 2.0000 | -1.0000 |", report)
        self.assertIn("| Centurio | COMET | 0.5000 | 0.4000 ± 0.2000 | -0.1000 |", report)
        self.assertIn("Training seeds: 41, 43, 44", report)
        self.assertNotIn("## DrawBench", report)
        self.assertNotIn("BLIP-2", report)

import statistics


def render_report(language, results):
    lines = [
        f"# Captioning results: {language}",
        "",
        "Fine-tuned: mean ± sample std across seeds. Δ: fine-tuned minus zero-shot.",
        "Higher is better. BLEU and chrF++ are out of 100. BERTScore and COMET are unscaled.",
        "",
    ]
    metrics = (
        ("bleu", "BLEU"),
        ("chrf++", "chrF++"),
        ("bertscore", "BERTScore"),
        ("comet", "COMET"),
    )
    for subset, title in (
        ("all", "All test images"),
        ("mscoco", "COCO"),
        ("drawbench", "DrawBench"),
    ):
        rows = [r for r in results if r["subset"] == subset]
        if not rows:
            continue

        lines += [
            f"## {title}",
            "",
            f"Test images: {rows[0]['count']}",
            "",
            "| Model | Metric | Zero-shot | Fine-tuned, mean ± std | Δ |",
            "|---|---|---:|---:|---:|",
        ]
        for model in sorted({r["model"] for r in rows}):
            model_rows = [row for row in rows if row["model"] == model]
            (baseline,) = [row for row in model_rows if row["kind"] == "zero"]
            tuned = [row for row in model_rows if row["kind"] == "lora"]
            name = {"blip2": "BLIP-2", "centurio": "Centurio"}[model]

            for key, label in metrics:
                values = [r[key] for r in tuned]
                mean = statistics.mean(values)
                score = f"{mean:.4f}"
                if len(values) > 1:
                    score += f" ± {statistics.stdev(values):.4f}"

                delta = mean - baseline[key]
                lines.append(
                    f"| {name} | {label} | {baseline[key]:.4f} | {score} | {delta:+.4f} |"
                )
        lines.append("")

    seeds = sorted({int(r["training_seed"]) for r in results if r["kind"] == "lora"})
    lines += [
        "## Run details",
        "",
        f"Training seeds: {', '.join(map(str, seeds))}. Same seeds for generation and scoring.",
        "Zero-shot: one run per model, generation/evaluation seed 41.",
        "Scores for each seed are in the CSV.",
        "",
    ]
    if any(r["model"] == "blip2" for r in results):
        lines += [
            "BLIP-2 zero-shot captions are English, scored against your reviewed captions.",
            "",
        ]
    return "\n".join(lines)

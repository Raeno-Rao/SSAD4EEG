"""Run paper/appendix component, covariance, and ODE-step ablations.

Inference-only ODE sweeps reuse independently trained four-step full-model
checkpoints. Retrained sweeps fit a separate model at each step count.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.ssad_loso import DATASETS, LOSOWindows, channel_stats
from models.ssad import SSAD
from scripts.run_ssad_loso import VARIANTS, benchmark_latency, evaluate, run_fold, seed_all, split_subjects


FAMILIES = {
    "components": ["backbone", "gate", "iso_sde", "aniso_no_score", "aniso_score", "score_gur", "full"],
    "controlled": ["identity_iso", "filter_iso", "identity_aniso", "full", "residual", "isotropic_gur", "no_gur", "no_sa"],
    "covariance": ["cov_isotropic", "cov_random", "cov_fixed", "full"],
    "alignment": ["align_off", "align_on"],
    "frozen_gate": ["cov_isotropic", "cov_random", "cov_fixed", "full"],
}
STEPS = (1, 2, 4, 10, 20)
METRICS = ("accuracy", "balanced_accuracy", "weighted_f1", "recovery_mse", "spectral_snr_db",
           "task_band_score_energy_pct", "recovery_distance", "latency_ms_per_sample")
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")


def summarize(records):
    paired = {}
    for record in records:
        if record["dataset"] == "deap":
            key = (record["variant"], record["steps"], record["test_subject"])
            paired.setdefault(key, {})[record["task"]] = record
    records = list(records)
    for pair in paired.values():
        if "valence" in pair and "arousal" in pair:
            combined = {key: pair["valence"][key] for key in ("dataset", "variant", "steps", "test_subject")}
            combined["task"] = "task_average"
            for metric in METRICS:
                if metric in pair["valence"] and metric in pair["arousal"]:
                    combined[metric] = (pair["valence"][metric] + pair["arousal"][metric]) / 2
            records.append(combined)
    rows = []
    keys = sorted({(r["dataset"], r["task"], r["variant"], r["steps"]) for r in records})
    for dataset, task, variant, steps in keys:
        selected = [r for r in records if (r["dataset"], r["task"], r["variant"], r["steps"]) == (dataset, task, variant, steps)]
        row = {"dataset": dataset, "task": task, "variant": variant, "steps": steps, "subjects": len(selected)}
        for metric in METRICS:
            values = [r[metric] for r in selected if metric in r]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        for band, name in enumerate(BAND_NAMES):
            values = [r["raw_latent_band_power_correlation"][band] for r in selected
                      if "raw_latent_band_power_correlation" in r
                      and r["raw_latent_band_power_correlation"][band] is not None]
            if values:
                row[f"{name}_correlation_mean"] = float(np.mean(values))
                row[f"{name}_correlation_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return rows


def interaction_effects(records):
    records = list(records)
    deap_pairs = {}
    for record in records:
        if record["dataset"] == "deap" and record["variant"] in {"identity_iso", "filter_iso", "identity_aniso", "full"}:
            key = (record["variant"], record["test_subject"])
            deap_pairs.setdefault(key, {})[record["task"]] = record["balanced_accuracy"]
    for (variant, subject), pair in deap_pairs.items():
        if "valence" in pair and "arousal" in pair:
            records.append({"dataset": "deap", "task": "task_average", "test_subject": subject,
                            "variant": variant, "balanced_accuracy": (pair["valence"] + pair["arousal"]) / 2})
    by_subject = {}
    for record in records:
        if record["variant"] in {"identity_iso", "filter_iso", "identity_aniso", "full"}:
            key = (record["dataset"], record["task"], record["test_subject"])
            by_subject.setdefault(key, {})[record["variant"]] = record["balanced_accuracy"]
    effects = {}
    for (dataset, task, subject), values in by_subject.items():
        if len(values) == 4:
            effect = values["full"] - values["filter_iso"] - values["identity_aniso"] + values["identity_iso"]
            effects.setdefault((dataset, task), []).append({"subject": subject, "interaction_pp": effect * 100})
    return [{"dataset": dataset, "task": task, "subjects": values,
             "mean_pp": float(np.mean([v["interaction_pp"] for v in values])),
             "std_pp": float(np.std([v["interaction_pp"] for v in values], ddof=1)) if len(values) > 1 else None}
            for (dataset, task), values in sorted(effects.items())]


def inference_sweep(args, dataset, task, subject, stats=None):
    seed_all(args.seed + subject)
    project = Path(__file__).resolve().parents[1]
    trained = Path(args.checkpoint_root) / dataset / task / f"subject{subject:02d}"
    result_file = trained / "result.json"
    checkpoint = trained / "best.pt"
    if not result_file.exists() or not checkpoint.exists():
        raise FileNotFoundError(f"Four-step checkpoint and result required at {trained}")
    metadata = json.loads(result_file.read_text(encoding="utf-8"))
    if (metadata.get("variant") != "full" or metadata.get("steps") != 4
            or metadata.get("seed") != args.seed
            or metadata.get("alignment_weight") != args.alignment_weight):
        raise ValueError(f"Checkpoint metadata does not match full four-step model, seed, and alignment weight: {result_file}")
    train_subjects, validation_subjects = split_subjects(dataset, subject, args.seed)
    if metadata["train_subjects"] != train_subjects or metadata["validation_subjects"] != validation_subjects:
        raise ValueError(f"Checkpoint has different LOSO split: {result_file}")
    mean, std = stats if stats is not None else channel_stats(project / "data", dataset, train_subjects)
    test = LOSOWindows(project / "data", dataset, [subject], mean, std, task)
    loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.workers)
    spec = DATASETS[dataset]
    device = torch.device(args.device)
    model = SSAD(spec["channels"], spec["classes"], len(train_subjects), spec["rate"], steps=4).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    rows = []
    for steps in STEPS:
        model.steps = steps
        metrics = evaluate(model, loader, device, recovery=True,
                           task_band=(8., 30.) if dataset == "bciiv2a" else (4., 13.))
        rows.append({"dataset": dataset, "task": task, "test_subject": subject,
                     "variant": "fixed_checkpoint", "steps": steps, "nfe": steps,
                     "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                     "precision": "float32", "latency_batch_size": 1,
                     "latency_ms_per_sample": benchmark_latency(model, test[0][0], device), **metrics})
    return rows


def format_ode_tables(summary, family):
    tables = []
    for dataset, task in (("bciiv2a", "motor_imagery"), ("deap", "task_average")):
        rows = [row for row in summary if row["dataset"] == dataset and row["task"] == task]
        if len(rows) != len(STEPS) or any(row["subjects"] != DATASETS[dataset]["subjects"] for row in rows):
            continue
        rows.sort(key=lambda row: row["steps"])
        lines = ["\\begin{table}[t]", "\\centering",
                 "\\caption{ODE-step ablation on " + dataset + " (" + family.replace("_", " ") + ").}",
                 "\\begin{tabular}{rcccc}", "\\toprule",
                 "Steps & NFE & BA (\\%) & WF1 (\\%) & Latency (ms/sample) " + "\\\\", "\\midrule"]
        for row in rows:
            def cell(metric, scale=1.):
                return f"{row[metric + '_mean'] * scale:.2f} $\\pm$ {row[metric + '_std'] * scale:.2f}"
            lines.append(f"{row['steps']} & {row['steps']} & {cell('balanced_accuracy', 100)} & "
                         f"{cell('weighted_f1', 100)} & {cell('latency_ms_per_sample')} " + "\\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        tables.append("\n".join(lines))
    return "\n\n".join(tables) + "\n" if tables else ""


def write_results(root, records):
    root.mkdir(parents=True, exist_ok=True)
    (root / "per_subject.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = summarize(records)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if root.name == "controlled":
        (root / "interaction.json").write_text(json.dumps(interaction_effects(records), indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in summary for key in row))
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    if root.name in {"ode_inference", "ode_retrain"}:
        table = format_ode_tables(summary, root.name)
        if table:
            (root / "ode_steps_table.tex").write_text(table, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=[*FAMILIES, "ode_inference", "ode_retrain"], required=True)
    parser.add_argument("--dataset", choices=["bciiv2a", "deap", "both"], default="both")
    parser.add_argument("--task", choices=["valence", "arousal", "both"], default="both")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=5e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.,
                        help="Use 0 for manuscript variants; set a positive weight for the latent-band extension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parents[1] / "outputs" / "ssad_ablations"))
    parser.add_argument("--checkpoint-root", default=str(Path(__file__).resolve().parents[1] / "outputs" / "ssad"))
    parser.add_argument("--frozen-reference")
    args = parser.parse_args()
    if args.family == "frozen_gate" and not args.frozen_reference:
        parser.error("--frozen-reference is required for frozen_gate")
    datasets = list(DATASETS) if args.dataset == "both" else [args.dataset]
    output_root = Path(args.output_root).resolve() / args.family
    records = []
    stats_cache = {}
    for dataset in datasets:
        tasks = ["motor_imagery"] if dataset == "bciiv2a" else (["valence", "arousal"] if args.task == "both" else [args.task])
        subjects = [args.fold] if args.fold is not None else range(1, DATASETS[dataset]["subjects"] + 1)
        for task in tasks:
            for subject in subjects:
                if not 1 <= subject <= DATASETS[dataset]["subjects"]:
                    parser.error(f"Invalid fold {subject} for {dataset}")
                key = (dataset, subject)
                if key not in stats_cache:
                    train_subjects, _ = split_subjects(dataset, subject, args.seed)
                    stats_cache[key] = channel_stats(Path(__file__).resolve().parents[1] / "data", dataset, train_subjects)
                if args.family == "ode_inference":
                    records.extend(inference_sweep(args, dataset, task, subject, stats_cache[key]))
                    write_results(output_root, records)
                    continue
                variants = FAMILIES.get(args.family, ["full"])
                step_counts = STEPS if args.family == "ode_retrain" else (4,)
                for variant in variants:
                    for steps in step_counts:
                        run_args = argparse.Namespace(**vars(args))
                        run_args.variant = variant
                        if args.family == "frozen_gate":
                            run_args.variant = "frozen_" + variant
                        run_args.steps = steps
                        if args.family == "alignment":
                            run_args.alignment_weight = 0. if variant == "align_off" else (args.alignment_weight or 0.01)
                        run_args.recovery = args.family in {"covariance", "ode_retrain"}
                        run_args.benchmark = args.family == "ode_retrain"
                        run_args.output_root = str(output_root / variant / f"k{steps:02d}")
                        result = run_fold(run_args, dataset, task, subject, stats_cache[key])
                        records.append(result)
                        write_results(output_root, records)
    print(f"Wrote {len(records)} subject-variant results to {output_root}")


if __name__ == "__main__":
    main()

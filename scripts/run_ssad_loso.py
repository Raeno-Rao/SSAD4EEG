"""Reproducible SSAD LOSO runs for BCI IV-2a and DEAP.

Examples:
  python scripts/run_ssad_loso.py --dataset bciiv2a --fold 1
  python scripts/run_ssad_loso.py --dataset deap --task both
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from datasets.ssad_loso import DATASETS, LOSOWindows, channel_stats
from models.ssad import SSAD, band_log_power


VARIANTS = {
    "full": {},
    "backbone": {"filtering": False, "noise_enabled": False, "refinement": "none", "gur_mode": "off", "subject_alignment": False},
    "gate": {"noise_enabled": False, "refinement": "none", "gur_mode": "off", "subject_alignment": False},
    "iso_sde": {"covariance_mode": "isotropic", "refinement": "none", "gur_mode": "off", "subject_alignment": False},
    "aniso_no_score": {"refinement": "none", "gur_mode": "off", "subject_alignment": False},
    "aniso_score": {"gur_mode": "off", "subject_alignment": False},
    "score_gur": {"subject_alignment": False},
    "identity_iso": {"filtering": False, "covariance_mode": "isotropic"},
    "filter_iso": {"covariance_mode": "isotropic"},
    "identity_aniso": {"filtering": False},
    "residual": {"refinement": "residual"},
    "isotropic_gur": {"gur_mode": "isotropic"},
    "no_gur": {"gur_mode": "off"},
    "no_sa": {"subject_alignment": False},
    "cov_isotropic": {"covariance_mode": "isotropic"},
    "cov_random": {"covariance_mode": "random"},
    "cov_fixed": {"covariance_mode": "fixed"},
    "align_off": {},
    "align_on": {},
    "frozen_cov_isotropic": {"covariance_mode": "isotropic", "gur_mode": "off"},
    "frozen_cov_random": {"covariance_mode": "random", "gur_mode": "off"},
    "frozen_cov_fixed": {"covariance_mode": "fixed", "gur_mode": "off"},
    "frozen_full": {"gur_mode": "off"},
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_subjects(dataset, test_subject, seed):
    subjects = [s for s in range(1, DATASETS[dataset]["subjects"] + 1) if s != test_subject]
    rng = np.random.default_rng(seed + test_subject)
    rng.shuffle(subjects)
    validation_count = 1 if dataset == "bciiv2a" else 3
    return sorted(subjects[validation_count:]), sorted(subjects[:validation_count])


@torch.no_grad()
def benchmark_latency(model, sample, device, warmup=10, repetitions=30):
    model.eval()
    sample = sample[None].to(device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000 * (time.perf_counter() - start) / repetitions


@torch.no_grad()
def evaluate(model, loader, device, diagnostics=False, recovery=False, task_band=None):
    model.eval()
    true, predicted, raw_bands, latent_bands = [], [], [], []
    recovery_values = {key: [] for key in ("recovery_mse", "spectral_snr_db", "task_band_score_energy_pct", "recovery_distance")}
    for x, y, _ in loader:
        x = x.to(device)
        output = model(x, return_details=recovery)
        true.extend(y.tolist())
        predicted.extend(output["logits"].argmax(1).cpu().tolist())
        if diagnostics:
            raw_bands.append(band_log_power(x, model.sampling_rate).cpu())
            latent_bands.append(band_log_power(model.encoder(x), model.sampling_rate).cpu())
        if recovery:
            clean, perturbed, refined = (output[key] for key in ("clean", "perturbed", "refined"))
            error = refined - clean
            recovery_values["recovery_mse"].extend(error.square().mean(dim=(1, 2)).cpu().tolist())
            clean_fft = torch.fft.fft(clean, dim=-1, norm="ortho")
            error_fft = torch.fft.fft(error, dim=-1, norm="ortho")
            snr = 10 * torch.log10(clean_fft.abs().square().sum(dim=(1, 2)) /
                                   error_fft.abs().square().sum(dim=(1, 2)).clamp_min(1e-12))
            recovery_values["spectral_snr_db"].extend(snr.cpu().tolist())
            distance = error.flatten(1).norm(dim=1) / (perturbed - clean).flatten(1).norm(dim=1).clamp_min(1e-12)
            recovery_values["recovery_distance"].extend(distance.cpu().tolist())
            if output["score_fields"]:
                freqs = torch.fft.rfftfreq(clean.shape[-1], d=1. / model.sampling_rate).to(device)
                band = (freqs >= task_band[0]) & (freqs <= task_band[1])
                spectra = torch.stack([torch.fft.rfft(field, dim=-1).abs().square().sum(dim=1)
                                       for field in output["score_fields"]]).sum(dim=0)
                ratio = 100 * spectra[:, band].sum(dim=-1) / spectra.sum(dim=-1).clamp_min(1e-12)
                recovery_values["task_band_score_energy_pct"].extend(ratio.cpu().tolist())
    metrics = {
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy_score(true, predicted),
        "weighted_f1": f1_score(true, predicted, average="weighted", zero_division=0),
        "n_windows": len(true),
    }
    if diagnostics:
        raw = torch.cat(raw_bands).numpy()
        latent = torch.cat(latent_bands).numpy()
        correlations = []
        for band in range(raw.shape[1]):
            a, b = raw[:, band], latent[:, band]
            correlations.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else None)
        metrics["raw_latent_band_power_correlation"] = correlations
        metrics["gate_centers_hz"] = model.geometry.centers.detach().cpu().tolist()
        metrics["gate_widths_hz"] = model.geometry.log_widths.exp().detach().cpu().tolist()
    if recovery:
        metrics.update({key: float(np.mean(values)) for key, values in recovery_values.items() if values})
    return metrics


def run_fold(args, dataset, task, test_subject, stats=None):
    seed_all(args.seed + test_subject)
    root = Path(__file__).resolve().parents[1]
    train_subjects, val_subjects = split_subjects(dataset, test_subject, args.seed)
    mean, std = stats if stats is not None else channel_stats(root / "data", dataset, train_subjects)
    domain_map = {sid: idx for idx, sid in enumerate(train_subjects)}
    train = LOSOWindows(root / "data", dataset, train_subjects, mean, std, task, domain_map)
    validation = LOSOWindows(root / "data", dataset, val_subjects, mean, std, task)
    test = LOSOWindows(root / "data", dataset, [test_subject], mean, std, task)
    device = torch.device(args.device)
    spec = DATASETS[dataset]
    model = SSAD(spec["channels"], spec["classes"], len(train_subjects), spec["rate"],
                 steps=args.steps, **VARIANTS[args.variant]).to(device)
    if getattr(args, "frozen_reference", None):
        reference = Path(args.frozen_reference) / dataset / task / f"subject{test_subject:02d}"
        metadata = json.loads((reference / "result.json").read_text(encoding="utf-8"))
        if metadata["train_subjects"] != train_subjects or metadata["validation_subjects"] != val_subjects:
            raise ValueError("Frozen reference uses a different subject split")
        state = torch.load(reference / "best.pt", map_location=device, weights_only=True)
        own = model.state_dict()
        for name in own:
            if name.startswith(("encoder.", "geometry.")):
                own[name] = state[name]
        model.load_state_dict(own)
        for parameter in list(model.encoder.parameters()) + list(model.geometry.parameters()):
            parameter.requires_grad_(False)
    gate_params = list(model.geometry.parameters())
    gate_ids = {id(p) for p in gate_params}
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if id(p) not in gate_ids], "lr": args.lr},
        {"params": gate_params, "lr": args.gate_lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.workers)
    output_dir = Path(args.output_root).resolve() / dataset / task / f"subject{test_subject:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ba, stale = -1., 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y, subject in train_loader:
            x, y, subject = x.to(device), y.to(device), subject.to(device)
            output = model(x, training_losses=True, deterministic=False)
            loss = (nn.functional.cross_entropy(output["logits"], y)
                    + output["score_loss"]
                    + 1e-3 * output["gur_loss"]
                    + (0.1 * nn.functional.cross_entropy(output["domain_logits"], subject) if model.subject_alignment else 0.)
                    + args.alignment_weight * output["align_loss"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        ba = val_metrics["balanced_accuracy"]
        print(f"{dataset}/{task}/S{test_subject:02d} epoch {epoch + 1}: loss={np.mean(losses):.4f} val_BA={ba:.4f}", flush=True)
        if ba > best_ba:
            best_ba, stale = ba, 0
            torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device, weights_only=True))
    task_band = (8., 30.) if dataset == "bciiv2a" else (4., 13.)
    test_metrics = evaluate(model, test_loader, device, diagnostics=True,
                            recovery=args.recovery, task_band=task_band)
    result = {
        "dataset": dataset, "task": task, "test_subject": test_subject,
        "train_subjects": train_subjects, "validation_subjects": val_subjects,
        "best_validation_balanced_accuracy": best_ba,
        "epochs_run": epoch + 1, "seed": args.seed,
        "alignment_weight": args.alignment_weight, "variant": args.variant, "steps": args.steps,
        "frozen_reference": getattr(args, "frozen_reference", None),
        **test_metrics,
    }
    if args.benchmark:
        result["latency_ms_per_sample"] = benchmark_latency(model, test[0][0], device)
        result["nfe"] = model.steps if model.refinement != "none" else 0
        result["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        result["precision"] = "float32"
        result["latency_batch_size"] = 1
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bciiv2a", "deap", "both"], default="both")
    parser.add_argument("--task", choices=["valence", "arousal", "both"], default="both")
    parser.add_argument("--fold", type=int, help="Run only one held-out subject")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=5e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--recovery", action="store_true", help="Compute Appendix E recovery metrics on the test subject")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parents[1] / "outputs" / "ssad"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    datasets = list(DATASETS) if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        tasks = ["motor_imagery"] if dataset == "bciiv2a" else (["valence", "arousal"] if args.task == "both" else [args.task])
        folds = [args.fold] if args.fold is not None else range(1, DATASETS[dataset]["subjects"] + 1)
        task_summaries = {}
        for task in tasks:
            results = []
            for subject in folds:
                if not 1 <= subject <= DATASETS[dataset]["subjects"]:
                    parser.error(f"Invalid fold {subject} for {dataset}")
                results.append(run_fold(args, dataset, task, subject))
            if args.fold is None:
                summary = {metric: {"mean": float(np.mean([r[metric] for r in results])),
                                    "std": float(np.std([r[metric] for r in results], ddof=1))}
                           for metric in ("accuracy", "balanced_accuracy", "weighted_f1")}
                path = Path(args.output_root).resolve() / dataset / task / "summary.json"
                path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"{dataset}/{task}: {summary}")
                task_summaries[task] = summary
        if dataset == "deap" and args.fold is None and len(task_summaries) == 2:
            average = {metric: float(np.mean([task_summaries[task][metric]["mean"] for task in tasks]))
                       for metric in ("accuracy", "balanced_accuracy", "weighted_f1")}
            path = Path(args.output_root).resolve() / dataset / "task_average.json"
            path.write_text(json.dumps(average, indent=2), encoding="utf-8")
            print(f"deap task average: {average}")


if __name__ == "__main__":
    main()

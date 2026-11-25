# SSAD LOSO experiments

This is a new implementation based on the method in `v1_main (SSAD).tex`.
It does not reuse the older `FAIBNet` results in `outputs/subject*`.

## Run

Use a Python environment with PyTorch, NumPy, scikit-learn, and the packages
in `requirements.txt`. The project already contains BCI IV-2a subject arrays
under `data/bciiv2a` and DEAP source files under `data/deap`. DEAP subjects
are converted individually to memory-mapped `.npy` files on first access.

```powershell
python scripts/run_ssad_loso.py --dataset bciiv2a
python scripts/run_ssad_loso.py --dataset deap --task both
```

Use `--fold 1 --epochs 1` only for a pipeline smoke test. Full runs default
to 200 epochs with patience 20. Results and best checkpoints go to
`outputs/ssad/<dataset>/<task>/subjectNN/`; complete runs also create
`summary.json`. DEAP's `task_average.json` averages the valence and arousal
means. `--alignment-weight 0` is the matching ablation for the added spectral
alignment term. Keep the same seed and splits for that comparison.

## Protocol and interpretation

- LOSO holds out one subject for testing. Validation subjects are chosen only
  from the remaining subjects (one on BCI, three on DEAP). Channel mean and
  standard deviation are computed from training subjects only. Early stopping
  uses validation balanced accuracy; testing happens once with the selected
  checkpoint.
- BCI uses 22 channels at 250 Hz, one 1000-sample trial per example, and four
  classes. DEAP uses the first 32 EEG channels at 128 Hz. The 3-second baseline
  is dropped, and each 60-second trial is retained as one 7680-sample input.
  Valence and arousal are binarized at rating 5. DEAP metrics are trial-level.
- The encoder preserves the temporal sample grid before gating. The FFT grid
  therefore uses the original sampling rate (250 or 128 Hz). The five Gaussian
  bases start at 2, 6, 10, 20, and 35 Hz, as in the manuscript. The gate acts
  on the learned sequence, not the raw EEG.
- The optional alignment loss compares *relative log band powers* of raw EEG and
  ungated latent features in delta, theta, alpha, beta, and gamma ranges. It
  encourages the latent representation to retain the physical frequency
  organization despite nonlinear encoding. Each fold records raw-to-latent
  band-power correlations on held-out data and learned gate centers/widths.
  These are diagnostics, not proof that every latent channel represents a
  physiological rhythm. Report them alongside the zero-alignment ablation.
- The implementation includes anisotropic VE perturbation, preconditioned
  score matching, a three-scale temporal score U-Net, four-step probability
  flow refinement, GUR, and subject-adversarial alignment. The fixed inference
  noise seed makes evaluation reproducible. Validation allocation,
  and the added alignment term are explicit implementation choices because
  the manuscript does not completely specify them. New results should be
  identified as this implementation rather than copied from the manuscript.

## Paper and appendix ablations

The ablation driver trains each variant independently, using the same LOSO
split, training schedule, seed, and validation-only checkpoint selection.
For the manuscript-style runs, its default alignment weight is **zero**;
the latent-band alignment extension is a separate `alignment` family.

```powershell
python scripts/run_ssad_ablations.py --family components --dataset both
python scripts/run_ssad_ablations.py --family controlled --dataset both
python scripts/run_ssad_ablations.py --family covariance --dataset both
python scripts/run_ssad_ablations.py --family alignment --dataset both
python scripts/run_ssad_ablations.py --family frozen_gate --dataset both --frozen-reference outputs/ssad
```

`components` corresponds to the BCI/DEAP columns of main Table 2.
`controlled` corresponds to Appendix D's filtering-by-covariance interaction,
residual-refiner, isotropic-GUR, no-GUR, and no-SA controls. `covariance`
corresponds to main Table 3 / Appendix E, including matched-trace isotropic,
random, fixed canonical-band, and learned covariance. Its results include
recovery MSE, spectral SNR, task-band score energy, and recovery distance.
Sleep-EDF is not included because its data pipeline is absent from this repo.
These scripts must be run to obtain new values; the numbers printed in the
manuscript are not treated as results from this implementation.

## ODE step count

```powershell
python scripts/run_ssad_ablations.py --family ode_retrain --dataset both
python scripts/run_ssad_ablations.py --family ode_inference --dataset both --checkpoint-root outputs/ssad
```

Both families compare 1, 2, 4, 10, and 20 Euler steps, with
`NFE = steps`. `ode_retrain` fits a separate model at each count and uses
the same step count for training and testing. `ode_inference` loads a fully
trained four-step `full` checkpoint for each fold and changes only the solver
step count at inference. It checks the checkpoint's seed and split metadata.
For the inference sweep, train the four-step checkpoints first with
`run_ssad_loso.py --alignment-weight 0`; use those as `--checkpoint-root`.
The two sweeps answer different questions and should be reported separately.

The driver writes per-subject JSON and mean/standard-deviation CSV/JSON under
`outputs/ssad_ablations/<family>/`. DEAP includes a per-subject average of
valence and arousal before aggregation. Once every fold for all five step
counts is present, `ode_steps_table.tex` is generated with BA, WF1, NFE, and
measured single-sample latency. Latency uses batch size one, warmup, and GPU
synchronization. Hardware, precision, and batch size should accompany any
paper latency claim. `--fold 1 --epochs 1` is only a script smoke test.

#!/usr/bin/env python3
"""
Convert DEAP dataset from .dat (pickled dicts) to per-subject .npy files.

Expected input layout (./deap):
    s01.dat, s02.dat, ..., s32.dat  (case-insensitive variants also accepted)
    Each .dat is a pickled dict with keys:
        'data'   : ndarray, shape (40, 40, 8064)  [trials, channels, time]
        'labels' : ndarray, shape (40, 4)         [valence, arousal, dominance, liking]

Output layout (./data/deap):
    deap_S001_trials.npy  (40, 40, 8064)
    deap_S001_labels.npy  (40, 4)
    ...
    deap_S032_trials.npy
    deap_S032_labels.npy
"""

import os
import pickle
import numpy as np


def convert_deap(input_dir: str = "./deap", output_dir: str = "./data/deap"):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 33):
        sid_str = f"{i:02d}"
        # Try common DEAP file naming patterns
        candidates = [
            os.path.join(input_dir, f"s{sid_str}.dat"),
            os.path.join(input_dir, f"S{sid_str}.dat"),
            os.path.join(input_dir, f"subject{sid_str}.dat"),
            os.path.join(input_dir, f"participant{sid_str}.dat"),
            os.path.join(input_dir, f"{sid_str}.dat"),
        ]

        filepath = None
        for cand in candidates:
            if os.path.exists(cand):
                filepath = cand
                break

        if filepath is None:
            raise FileNotFoundError(
                f"Could not find DEAP data file for subject {i:03d} in {input_dir}. "
                f"Tried: {candidates}"
            )

        with open(filepath, "rb") as f:
            subject_dict = pickle.load(f, encoding="latin1")

        data = subject_dict["data"]    # (40, 40, 8064)
        labels = subject_dict["labels"]  # (40, 4)

        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)

        if data.ndim != 3:
            raise ValueError(f"Subject {i:03d}: data must be 3D, got shape {data.shape}")
        if labels.ndim != 2:
            raise ValueError(f"Subject {i:03d}: labels must be 2D, got shape {labels.shape}")
        if data.shape[0] != labels.shape[0]:
            raise ValueError(f"Subject {i:03d}: trial count mismatch")

        out_prefix = f"deap_S{i:03d}"
        np.save(os.path.join(output_dir, f"{out_prefix}_trials.npy"), data)
        np.save(os.path.join(output_dir, f"{out_prefix}_labels.npy"), labels)

        print(f"Saved {out_prefix}: data={data.shape}, labels={labels.shape}")


if __name__ == "__main__":
    convert_deap()
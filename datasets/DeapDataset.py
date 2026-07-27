import os
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class DeapDataset(Dataset):
    """
    Read subject-wise .npy files saved as:
        deap_S001_trials.npy
        deap_S001_labels.npy
        ...
        deap_S032_trials.npy
        deap_S032_labels.npy

    Each trials file is expected to be:
        (N, C, T)  —  N trials, C channels (40 for DEAP: 32 EEG + 8 peripheral), T time samples

    This dataset returns:
        x, y
    where
        x: FloatTensor [C, T]   (after optional baseline removal and length padding/cropping)
        y: LongTensor []          (binary label: 0=Low, 1=High)
    """

    def __init__(
        self,
        root: str,
        subjects: Optional[Iterable[int]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        target_length: int = 7680,
        label_dim: int = 0,            # 0=valence, 1=arousal, 2=dominance, 3=liking
        binary_threshold: float = 5.0, # > threshold -> High (1), <= threshold -> Low (0)
        remove_baseline: bool = True,
        baseline_length: int = 384,    # 3 seconds * 128 Hz
        eeg_only: bool = False,        # if True, keep only first 32 channels
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
    ):
        super().__init__()
        self.root = root
        self.target_length = target_length
        self.label_dim = label_dim
        self.binary_threshold = binary_threshold
        self.remove_baseline = remove_baseline
        self.baseline_length = baseline_length
        self.eeg_only = eeg_only
        self.dtype = dtype
        self.label_dtype = label_dtype

        if subjects is None:
            subjects = list(range(1, 33))
        self.subjects = sorted(list(subjects))

        all_x = []
        all_y = []

        for sid in self.subjects:
            prefix = f"deap_S{sid:03d}"
            x_path = os.path.join(root, f"{prefix}_trials.npy")
            y_path = os.path.join(root, f"{prefix}_labels.npy")

            if not os.path.exists(x_path):
                raise FileNotFoundError(f"Missing trials file: {x_path}")
            if not os.path.exists(y_path):
                raise FileNotFoundError(f"Missing labels file: {y_path}")

            x = np.load(x_path).astype(np.float32)
            y = np.load(y_path).astype(np.float32)

            if x.ndim != 3:
                raise ValueError(f"{x_path} must be 3D, got shape {x.shape}")

            # Heuristic transpose: if time < channels, likely (N, T, C)
            if x.shape[1] > x.shape[2]:
                x = np.transpose(x, (0, 2, 1))

            # Optional: keep only EEG channels (first 32)
            if self.eeg_only:
                x = x[:, :32, :]

            # Remove baseline (first 3 seconds at 128 Hz)
            if self.remove_baseline:
                if x.shape[2] > self.baseline_length:
                    x = x[:, :, self.baseline_length:]
                else:
                    raise ValueError(
                        f"Data length {x.shape[2]} <= baseline length {self.baseline_length}. "
                        f"Set remove_baseline=False if baseline already removed."
                    )

            # Pad / crop to target length
            if x.shape[2] > self.target_length:
                x = x[:, :, : self.target_length]
            elif x.shape[2] < self.target_length:
                pad = self.target_length - x.shape[2]
                x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")

            # Select target dimension and binarize
            if y.ndim == 2:
                y = y[:, self.label_dim]
            elif y.ndim != 1:
                raise ValueError(f"{y_path} must be 1D or 2D, got shape {y.shape}")

            # DEAP labels are 1-9; binarize at threshold (default 5)
            y = (y > self.binary_threshold).astype(np.int64)

            if y.min() < 0:
                raise ValueError(f"Labels must be non-negative, got min={y.min()}")

            if len(x) != len(y):
                raise ValueError(
                    f"Sample count mismatch for subject {sid}: x={len(x)}, y={len(y)}"
                )

            all_x.append(x)
            all_y.append(y)

        if len(all_x) == 0:
            raise ValueError(f"No valid subject data found in {root}")

        self.x = np.concatenate(all_x, axis=0)
        self.y = np.concatenate(all_y, axis=0)

        # Trial-wise normalization: per trial, per channel along time axis
        trial_mean = self.x.mean(axis=2, keepdims=True)
        trial_std = self.x.std(axis=2, keepdims=True)
        self.x = (self.x - trial_mean) / (trial_std + 1e-6)

        # After trial-wise norm, global mean/std are conceptually 0/1
        self.mean = 0.0
        self.std = 1.0

    def get_stats(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=self.dtype)
        y = torch.tensor(self.y[index], dtype=self.label_dtype)
        return x, y
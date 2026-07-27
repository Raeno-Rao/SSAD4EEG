import os
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BCIIV2aDataset(Dataset):
    """
    Read subject-wise .npy files saved as:
        bci_S001_trials.npy
        bci_S001_labels.npy
        ...
        bci_S009_trials.npy
        bci_S009_labels.npy

    Each trials file is expected to be:
        (N, C, T) or (N, T, C)

    This dataset returns:
        x, y
    where
        x: FloatTensor [C, T]
        y: LongTensor []
    """

    def __init__(
        self,
        root: str,
        subjects: Optional[Iterable[int]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        target_length: int = 1000,
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
    ):
        super().__init__()
        self.root = root
        self.target_length = target_length
        self.dtype = dtype
        self.label_dtype = label_dtype

        if subjects is None:
            subjects = list(range(1, 10))
        self.subjects = sorted(list(subjects))

        all_x = []
        all_y = []

        for sid in self.subjects:
            prefix = f"bci_S{sid:03d}"
            x_path = os.path.join(root, f"{prefix}_trials.npy")
            y_path = os.path.join(root, f"{prefix}_labels.npy")

            if not os.path.exists(x_path):
                raise FileNotFoundError(f"Missing trials file: {x_path}")
            if not os.path.exists(y_path):
                raise FileNotFoundError(f"Missing labels file: {y_path}")

            x = np.load(x_path).astype(np.float32)
            y = np.load(y_path).astype(np.int64)

            if x.ndim != 3:
                raise ValueError(f"{x_path} must be 3D, got shape {x.shape}")

            # Make sure shape is (N, C, T)
            if x.shape[1] != 22 and x.shape[2] == 22:
                x = np.transpose(x, (0, 2, 1))

            if x.shape[1] != 22:
                raise ValueError(f"{x_path} must have 22 channels, got shape {x.shape}")

            # Pad / crop to target length
            if x.shape[2] > target_length:
                x = x[:, :, :target_length]
            elif x.shape[2] < target_length:
                pad = target_length - x.shape[2]
                x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")

            # Make labels 0-based if needed
            if y.min() == 1:
                y = y - 1

            if y.min() < 0:
                raise ValueError(f"Labels must be non-negative after normalization, got min={y.min()}")

            if len(x) != len(y):
                raise ValueError(f"Sample count mismatch for subject {sid}: x={len(x)}, y={len(y)}")

            all_x.append(x)
            all_y.append(y)

        if len(all_x) == 0:
            raise ValueError(f"No valid subject data found in {root}")

        self.x = np.concatenate(all_x, axis=0)
        self.y = np.concatenate(all_y, axis=0)

         # trial-wise normalization

        trial_mean = self.x.mean(axis=2, keepdims=True)
        trial_std = self.x.std(axis=2, keepdims=True)

        self.x = (
            self.x - trial_mean
        ) / (trial_std + 1e-6)

        self.mean = 0.0
        self.std = 1.0

    def get_stats(self):
        return 0.0, 1.0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=self.dtype)
        y = torch.tensor(self.y[index], dtype=self.label_dtype)
        return x, y
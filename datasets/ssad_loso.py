"""Lazy, subject-indexed EEG windows for leakage-free LOSO experiments."""

from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


DATASETS = {
    "bciiv2a": {"subjects": 9, "channels": 22, "rate": 250, "length": 1000, "classes": 4},
    "deap": {"subjects": 32, "channels": 32, "rate": 128, "length": 7680, "classes": 2},
}


def subject_paths(root, dataset, subject):
    prefix = ("bci" if dataset == "bciiv2a" else "deap") + f"_S{subject:03d}"
    data_dir = Path(root) / dataset
    x_path = data_dir / f"{prefix}_trials.npy"
    y_path = data_dir / f"{prefix}_labels.npy"
    if dataset == "deap" and (not x_path.exists() or not y_path.exists()):
        source = data_dir / f"s{subject:02d}.dat"
        if not source.exists():
            raise FileNotFoundError(f"Missing {source} or converted {x_path}")
        with source.open("rb") as handle:
            raw = pickle.load(handle, encoding="latin1")
        np.save(x_path, np.asarray(raw["data"], dtype=np.float32))
        np.save(y_path, np.asarray(raw["labels"], dtype=np.float32))
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing subject {subject}: {x_path} / {y_path}")
    return x_path, y_path


def channel_stats(root, dataset, subjects):
    """Compute per-channel moments on training subjects only."""
    channels = DATASETS[dataset]["channels"]
    total = np.zeros(channels, dtype=np.float64)
    squares = np.zeros(channels, dtype=np.float64)
    count = 0
    for subject in subjects:
        x_path, _ = subject_paths(root, dataset, subject)
        data = np.load(x_path, mmap_mode="r")
        for trial in data:
            eeg = np.asarray(trial[:channels, 384:] if dataset == "deap" else trial[:channels], dtype=np.float64)
            if eeg.shape[0] != channels:
                raise ValueError(f"Expected {channels} EEG channels in {x_path}")
            total += eeg.sum(axis=1)
            squares += np.square(eeg).sum(axis=1)
            count += eeg.shape[-1]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class LOSOWindows(Dataset):
    def __init__(self, root, dataset, subjects, mean, std, task="valence", domain_map=None):
        self.dataset = dataset
        self.spec = DATASETS[dataset]
        self.mean = mean[:, None]
        self.std = std[:, None]
        self.task = task
        self.domain_map = domain_map or {}
        self.files = {}
        self.labels = {}
        self.index = []
        length = self.spec["length"]
        for sid in subjects:
            x_path, y_path = subject_paths(root, dataset, sid)
            x = np.load(x_path, mmap_mode="r")
            labels = np.load(y_path)
            if dataset == "deap":
                labels = (labels[:, 0 if task == "valence" else 1] > 5.).astype(np.int64)
                available = x.shape[-1] - 384
                windows = available // length
            else:
                labels = labels.astype(np.int64).reshape(-1)
                if labels.min() == 1:
                    labels -= 1
                windows = 1
            if x.shape[0] != len(labels) or windows < 1:
                raise ValueError(f"Invalid trials, labels or length for subject {sid}")
            self.files[sid] = x
            self.labels[sid] = labels
            self.index.extend((sid, trial, window) for trial in range(len(labels)) for window in range(windows))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        sid, trial, window = self.index[index]
        start = (384 if self.dataset == "deap" else 0) + window * self.spec["length"]
        stop = start + self.spec["length"]
        x = np.asarray(self.files[sid][trial, :self.spec["channels"], start:stop], dtype=np.float32)
        if x.shape[-1] < self.spec["length"]:
            x = np.pad(x, ((0, 0), (0, self.spec["length"] - x.shape[-1])))
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.copy()), int(self.labels[sid][trial]), self.domain_map.get(sid, 0)

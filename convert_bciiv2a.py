import os
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_labels_to_zero_based(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.int64)
    uniq = np.unique(y)

    if np.array_equal(uniq, np.array([0, 1, 2, 3])):
        return y
    if np.array_equal(uniq, np.array([1, 2, 3, 4])):
        return y - 1

    raise ValueError(f"Unexpected label values: {uniq}")


def build_loso_split(test_subject: int, n_subjects: int = 9, val_subject: int = None):
    subjects = list(range(n_subjects))
    assert test_subject in subjects

    if val_subject is None:
        val_subject = (test_subject + 1) % n_subjects
        if val_subject == test_subject:
            val_subject = (val_subject + 1) % n_subjects

    assert val_subject != test_subject

    train_subjects = [s for s in subjects if s not in [test_subject, val_subject]]
    val_subjects = [val_subject]
    test_subjects = [test_subject]

    return train_subjects, val_subjects, test_subjects


class BCIMergedDataset(Dataset):
    """
    使用 preprocess_bci_iv2a_local.py 处理后的 merged T+E 数据
    文件格式:
      bci_S001_trials.npy
      bci_S001_labels.npy

    返回:
      x, y, subject_id
    """

    def __init__(self, root, subjects=None, mean=None, std=None):
        self.root = root
        self.data = []
        self.labels = []
        self.subject_ids = []

        files = sorted([
            f for f in os.listdir(root)
            if f.startswith("bci_") and f.endswith("_trials.npy")
        ])

        for f in files:
            sid = int(f.split("_S")[1][:3]) - 1  # 0-based

            if subjects is not None and sid not in subjects:
                continue

            x = np.load(os.path.join(root, f)).astype(np.float32)
            y = np.load(os.path.join(root, f.replace("_trials", "_labels"))).astype(np.int64)
            y = normalize_labels_to_zero_based(y)

            self.data.append(x)
            self.labels.append(y)
            self.subject_ids.append(np.full((len(x),), sid, dtype=np.int64))

        if len(self.data) == 0:
            raise ValueError(f"No data found in {root} for subjects={subjects}")

        self.data = np.concatenate(self.data, axis=0)      # (N, 22, 1000)
        self.labels = np.concatenate(self.labels, axis=0)  # (N,)
        self.subject_ids = np.concatenate(self.subject_ids, axis=0)

        if mean is None:
            self.mean = float(self.data.mean())
            self.std = float(self.data.std())
        else:
            self.mean = float(mean)
            self.std = float(std)

        self.std = max(self.std, 1e-6)
        self.data = (self.data - self.mean) / self.std

        if self.labels.min() < 0 or self.labels.max() > 3:
            raise ValueError(f"Labels must be in [0,3], got [{self.labels.min()}, {self.labels.max()}]")

    def get_stats(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        s = torch.tensor(self.subject_ids[idx], dtype=torch.long)
        return x, y, s
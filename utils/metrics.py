import torch

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)


def compute_metrics(logits, labels):
    pred = torch.argmax(logits, dim=1)

    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average="macro")
    kappa = cohen_kappa_score(labels, pred)

    return {
        "acc": acc,
        "f1": f1,
        "kappa": kappa,
    }
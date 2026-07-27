#trainers/trainer.py
import os

from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils.metrics import compute_metrics
import json

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        beta,
        device,
        save_dir,
    ):
        self.model = model
        self.optimizer = optimizer
        self.beta = beta
        self.device = device
        self.save_dir = save_dir

        self.best_acc = 0

        os.makedirs(self.save_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.save_dir, "best.pt")

        # History for plotting
        self.history = {
            "epochs": [],
            "kl": [],
            "accuracy": [],
            "train_loss": [],
        }

    def train_epoch(self, loader):
        self.model.train()

        total_loss = 0

        for x, y in tqdm(loader):
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)

            logits = out["logits"]
            kl = out["kl"]

            cls_loss = F.cross_entropy(logits, y)
            loss = cls_loss + self.beta * kl

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        all_logits = []
        all_labels = []
        all_kl = []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)

            logits = out["logits"]
            kl = out["kl"]

            all_logits.append(logits)
            all_labels.append(y)
            all_kl.append(kl)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_kl = torch.stack(all_kl)

        metrics = compute_metrics(all_logits, all_labels)
        metrics["kl"] = all_kl.mean().item()

        return metrics

    def record_epoch(self, epoch, train_loss, metrics):
        """Record metrics for current epoch."""
        self.history["epochs"].append(epoch + 1)
        self.history["train_loss"].append(train_loss)
        self.history["kl"].append(metrics.get("kl", 0.0))
        self.history["accuracy"].append(metrics["acc"])

    def save_training_curves(self):
        """Save training curves as images."""
        import matplotlib.pyplot as plt
        import numpy as np

        epochs = self.history["epochs"]
        kl_values = self.history["kl"]
        acc_values = self.history["accuracy"]

        # Plot KL vs Epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, kl_values, "b-", linewidth=2, marker="o", markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("KL Divergence", fontsize=12)
        plt.title("KL Divergence vs Epoch", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        kl_path = os.path.join(self.save_dir, "kl_curve.png")
        plt.savefig(kl_path, dpi=100)
        plt.close()
        print(f"Saved KL curve: {kl_path}")

        # Plot Accuracy vs Epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, acc_values, "g-", linewidth=2, marker="o", markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Accuracy vs Epoch", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        acc_path = os.path.join(self.save_dir, "accuracy_curve.png")
        plt.savefig(acc_path, dpi=100)
        plt.close()
        print(f"Saved Accuracy curve: {acc_path}")

    def save_best(self, metrics):
        acc = metrics["acc"]

        if acc > self.best_acc:
            self.best_acc = acc

            torch.save(self.model.state_dict(), self.ckpt_path)

            result_path = os.path.join(self.save_dir, "result.json")
            with open(result_path, "w") as f:
                json.dump(
                    {
                        "acc": float(metrics["acc"]),
                        "f1": float(metrics["f1"]),
                        "kappa": float(metrics["kappa"]),
                    },
                    f,
                    indent=2,
                )

            print(f"Saved best checkpoint: {acc:.4f}")



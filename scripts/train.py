#train.py
import hydra
import sys
import os

# Ensure project root is on sys.path so imports work when running from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets.BCIIV2aDataset import BCIIV2aDataset
from models.faib_net import FAIBNet
from trainers.trainer import Trainer
from utils.seed import set_seed
from utils.visualize import plot_gate

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    device = cfg.train.device
    # Resolve data root relative to project root (hydra may change cwd)
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', cfg.data.root))

    train_set = BCIIV2aDataset(
        root=data_root,
        subjects=cfg.experiment.train_subjects,
    )
    mean, std = train_set.get_stats()

    test_set = BCIIV2aDataset(
        root=data_root,
        subjects=cfg.experiment.test_subjects,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    model = FAIBNet(
        in_channels=cfg.model.in_channels,
        seq_len=cfg.model.seq_len,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
        num_bands=cfg.model.num_bands,
        noise_scale=cfg.model.noise_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        beta=cfg.model.beta,
        device=device,
        save_dir=cfg.train.save_dir,
    )

    for epoch in range(cfg.train.epochs):
        train_loss = trainer.train_epoch(train_loader)

        metrics = trainer.evaluate(test_loader)

        trainer.record_epoch(epoch, train_loss, metrics)

        trainer.save_best(metrics)

        print(
            f"Epoch {epoch+1} | "
            f"loss={train_loss:.4f} | "
            f"acc={metrics['acc']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"kappa={metrics['kappa']:.4f}"
        )

    trainer.save_training_curves()

    gate = model.gate.build_gate()

    plot_gate(gate)


if __name__ == "__main__":
    main()
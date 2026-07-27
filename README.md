FAIBD-v3.0/

README.md

requirements.txt

configs/
    config.yaml
    model/
    train/
    dataset/

datasets/
    BCIIV2aDataset.py
    DEAPDataset.py
    CHBMITDataset.py
    transforms.py

models/

    spectral_gate.py

    variational_ib.py

    diffusion.py

    s4_backbone.py

    diffusion_head.py

    attention_pool.py

    classifier.py

    faib_model.py

losses/

    diffusion_loss.py

    ib_loss.py

    total_loss.py

engine/

    trainer.py

    evaluator.py

    earlystop.py

utils/

    ema.py

    scheduler.py

    metrics.py

    logger.py

    seed.py

    visualization.py

analysis/

    gate_analysis.py

    attention_analysis.py

    statistics.py

scripts/

    train.py

    run_loso.py

    run_deap.py

    run_chb.py

    visualize_gate.py

outputs/
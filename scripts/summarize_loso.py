import os
import json
import numpy as np

accs = []
f1s = []
kappas = []

for sid in range(1,10):

    path = (
        f"D:\Desktop\AAAI-Manifold diffusion for EEG detection\EEGDM-v2.0\outputs/subject{sid}/result.json"
    )

    with open(path) as f:
        r = json.load(f)

    accs.append(r["acc"])
    f1s.append(r["f1"])
    kappas.append(r["kappa"])

print()

print("ACC")
print(np.mean(accs), np.std(accs))

print("F1")
print(np.mean(f1s), np.std(f1s))

print("KAPPA")
print(np.mean(kappas), np.std(kappas))
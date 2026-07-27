import matplotlib.pyplot as plt
import numpy as np


def plot_gate(gate, save_path="gate.png"):
    gate = gate.detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    fs = 250
    T = 1000

    freqs = np.fft.rfftfreq(T, d=1 / fs)
    plt.plot(freqs, gate)
    plt.xlim([0, 50])

    plt.xlabel("Frequency Bin")
    plt.ylabel("Gate Value")

    plt.title("Learned Spectral Gate")

    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
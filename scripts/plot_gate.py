#scripts/plot_gate.py
import numpy as np
import matplotlib.pyplot as plt

gate = np.load("gate_subject1.npy")

fs = 250
T = 1000

freqs = np.fft.rfftfreq(T, d=1/fs)

plt.figure(figsize=(10,4))

plt.plot(freqs, gate)
plt.xlim([0,50])

plt.xlabel("Frequency Bin")
plt.ylabel("Gate")

plt.title("Learned Spectral Gate")

plt.grid()

plt.show()
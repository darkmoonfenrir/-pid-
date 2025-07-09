import numpy as np
import matplotlib.pyplot as plt
import csv

# ====== Step 1: Load Data ======
data = []
with open("data.csv", newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        data.append({
            "time": float(row[0]),
            "temperature": float(row[1]),
            "volte": float(row[2])
        })

times = np.array([d["time"] for d in data])
temperatures = np.array([d["temperature"] for d in data])

# ====== Step 2: Set FOPTD Parameters ======
# Modify these manually to test fit
K = 35.8816         # Try: 23.35, or 9.88, or 0.99
T = 3105         # Try: 2928 or 3105
L = 110            # Try: 0 or 100~200
y0 = temperatures[0]  # Initial temperature

# ====== Step 3: Compute Theoretical Response with Delay ======
y_theory = np.zeros_like(times)
for i, t in enumerate(times):
    if t < L:
        y_theory[i] = y0
    else:
        y_theory[i] = y0 + K * (1 - np.exp(-(t - L) / T))

# ====== Step 4: Plot Comparison ======
plt.figure(figsize=(10, 6))
plt.plot(times, temperatures, label="Actual Data")
plt.plot(times, y_theory, '--', label=f"FOPTD Model (K={K}, T={T}, L={L})")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Actual vs FOPTD Theoretical Response")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit

# ==== Step 1: 读取CSV数据 ====
data = []
with open("data.csv", newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过表头
    for row in reader:
        data.append({
            "time": float(row[0]),
            "temperature": float(row[1]),
            "volte": float(row[2])
        })

# ==== Step 2: 数据提取 ====
times = np.array([d["time"] for d in data])
temperatures = np.array([d["temperature"] for d in data])
y0 = temperatures[0]  # 初始温度作为偏置

# ==== Step 3: 定义带纯滞后的 FOPTD 模型 ====
def foptd_model(t, K, T, L):
    """一阶迟滞系统响应模型（带延迟）"""
    response = np.piecewise(
        t,
        [t < L, t >= L],
        [lambda t: y0, lambda t: y0 + K * (1 - np.exp(-(t - L) / T))]
    )
    return response

# ==== Step 4: 设置初始参数（手动选取） ====
K0, T0, L0 = 35.88, 3105, 110  # 初始猜测
initial_guess = [K0, T0, L0]

# ==== Step 5: 进行非线性最小二乘拟合 ====
popt, pcov = curve_fit(foptd_model, times, temperatures, p0=initial_guess, maxfev=10000)
K_fit, T_fit, L_fit = popt

# ==== Step 6: 生成拟合曲线 ====
y_fit = foptd_model(times, K_fit, T_fit, L_fit)

# ==== Step 7: 计算拟合误差 ====
mse = np.mean((temperatures - y_fit) ** 2)

# ==== Step 8: 打印拟合结果 ====
print("=== Nonlinear Fitting Result ===")
print(f"Fitted K = {K_fit:.4f}")
print(f"Fitted T = {T_fit:.4f} s")
print(f"Fitted L = {L_fit:.4f} s")
print(f"Mean Squared Error (MSE) = {mse:.6f}")

# ==== Step 9: 可视化对比 ====
plt.figure(figsize=(10, 6))
plt.plot(times, temperatures, label="Actual Data")
plt.plot(times, y_fit, '--', label="Fitted FOPTD Model")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Actual vs Fitted Response (FOPTD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

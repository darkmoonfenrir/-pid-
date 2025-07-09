import csv
import numpy as np

# ========== 一阶系统响应基本结构 ==========
def compute_two_point_method(times, temps, input_step):
    y0 = temps[0]
    y_ss = np.max(temps)
    delta_y = y_ss - y0

    # 增益估算
    K = delta_y / input_step

    # 63.2%时间点
    y_target_632 = y0 + 0.632 * delta_y
    t_632 = None
    for i in range(1, len(temps)):
        if temps[i] >= y_target_632:
            t_632 = times[i] - times[0]
            break

    # 10%-90%上升时间法估 T
    y10 = y0 + 0.1 * delta_y
    y90 = y0 + 0.9 * delta_y
    t_10, t_90 = None, None
    for i in range(1, len(temps)):
        if temps[i] >= y10 and t_10 is None:
            t_10 = times[i]
        if temps[i] >= y90:
            t_90 = times[i]
            break

    if t_10 is not None and t_90 is not None:
        tau_10_90 = (t_90 - t_10) / 2.2
    else:
        tau_10_90 = None

    # 平均时间常数
    if t_632 and tau_10_90:
        tau_avg = (t_632 + tau_10_90) / 2
    else:
        tau_avg = t_632 or tau_10_90

    return K, tau_avg


def compute_smith_method(times, temps, input_step):
    y0 = temps[0]
    y_ss = np.max(temps)
    delta_y = y_ss - y0
    K = delta_y / input_step

    # 取最大斜率点（用差分法估算）
    dy_dt = np.gradient(temps, times)
    max_slope_index = np.argmax(dy_dt)
    max_slope = dy_dt[max_slope_index]
    t_inflect = times[max_slope_index]
    y_inflect = temps[max_slope_index]

    # 切线回推到 y0 得到 L
    L = t_inflect - (y_inflect - y0) / max_slope

    # 时间常数 T 估计（以一阶系统最大斜率为参考）
    T = delta_y / max_slope

    return K, T, L

# ========== 数据读取 ==========
data = []
try:
    with open("data.csv", newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            data.append({"time": float(row[0]), "temperature": float(row[1]), "volte": float(row[2])})
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# ========== 数据提取 ==========
times = np.array([d["time"] for d in data])
temps = np.array([d["temperature"] for d in data])
volts = np.array([d["volte"] for d in data])
input_step = 3.5

# ========== Two-Point Method ==========
K_two, T_two = compute_two_point_method(times, temps, input_step)
print("=== Two-Point Method Estimate ===")
print(f"K = {K_two:.4f}")
print(f"T = {T_two:.4f} s")
print(f"L = 0.0000 s   (Assumed no delay)")
print("")

# ========== Smith Method ==========
K_smith, T_smith, L_smith = compute_smith_method(times, temps, input_step)
print("=== Smith Method Estimate ===")
print(f"K = {K_smith:.4f}")
print(f"T = {T_smith:.4f} s")
print(f"L = {L_smith:.4f} s")

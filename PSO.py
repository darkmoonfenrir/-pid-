import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from tqdm import tqdm

# 系统模型 G(s)
K_plant = 36.1072
tau = 3272.4787
L = 68.2818
plant_nominal = ctrl.tf([K_plant], [tau, 1])
pade_num, pade_den = ctrl.pade(L, 2)
delay = ctrl.tf(pade_num, pade_den)
G = ctrl.series(delay, plant_nominal)

# 控制目标参数
T_init = 16.85
T_target = 35.0
T_step = T_target - T_init
t_sim = np.linspace(0, 20000, 10000)
u_input = np.ones_like(t_sim) * T_step

# PID参数范围
param_bounds = {
    "Kp": (0.0, 1.0),
    "Ki": (0.0, 0.001),
    "Kd": (0.0, 500.0)
}

# 仿真与性能指标
def simulate_pid(Kp, Ki, Kd):
    try:
        C = ctrl.tf([Kd, Kp, Ki], [1, 0])
        sys_cl = ctrl.feedback(ctrl.series(C, G), 1)
        t, y = ctrl.forced_response(sys_cl, T=t_sim, U=u_input)
        y += T_init

        overshoot = (np.max(y) - T_target) / T_target * 100
        steady_state_error = abs(y[-1] - T_target)
        tolerance = 0.05 * T_target
        indices_outside = np.where(np.abs(y - T_target) > tolerance)[0]

        if len(indices_outside) == 0:
            settling_time = 0
        else:
            last_outside_index = indices_outside[-1]
            settling_time = t[last_outside_index + 1] if last_outside_index < len(t) - 1 else np.inf

        return overshoot, steady_state_error, settling_time
    except Exception:
        return np.inf, np.inf, np.inf

# 适应度函数
def fitness(Kp, Ki, Kd):
    overshoot, sse, ts = simulate_pid(Kp, Ki, Kd)
    if np.isinf(overshoot) or np.isinf(sse) or np.isinf(ts):
        return 1e6
    return abs(overshoot)*1000 + sse*10000 + ts

# ========== 初始化种群 ========== #
seeds = np.array([
    [0.5, 0.0001, 50],
    [0.7, 0.00005, 30],
    [0.3, 0.0002, 70],
    [0.6, 0.00015, 40]
])
noise_scale = np.array([0.1, 0.00005, 5.0])
population = []

per_seed = 6  # 每个种子生成6个扰动样本
for seed in seeds:
    population.append(seed)  # 保留种子自身
    for _ in range(per_seed):
        perturb = noise_scale * (2 * np.random.rand(3) - 1)
        p = seed + perturb
        p[0] = np.clip(p[0], *param_bounds["Kp"])
        p[1] = np.clip(p[1], *param_bounds["Ki"])
        p[2] = np.clip(p[2], *param_bounds["Kd"])
        population.append(p)

population = np.array(population)
pop_size = population.shape[0]  # 动态匹配真实粒子数量
velocity = np.zeros_like(population)

# ========== 初始化历史最优 ========== #
pbest = population.copy()
pbest_scores = np.array([fitness(*p) for p in population])
gbest_idx = np.argmin(pbest_scores)
gbest = pbest[gbest_idx].copy()
gbest_score = pbest_scores[gbest_idx]

# ========== PSO 参数 ========== #
w = 0.7
c1 = 1.5
c2 = 1.5
max_iter = 100

# ========== 粒子群优化 ========== #
for iteration in tqdm(range(1, max_iter + 1), desc="优化PID"):
    for i in range(pop_size):
        r1, r2 = np.random.rand(), np.random.rand()
        velocity[i] = (w * velocity[i]
                       + c1 * r1 * (pbest[i] - population[i])
                       + c2 * r2 * (gbest - population[i]))
        population[i] += velocity[i]

        # 边界限制
        population[i][0] = np.clip(population[i][0], *param_bounds["Kp"])
        population[i][1] = np.clip(population[i][1], *param_bounds["Ki"])
        population[i][2] = np.clip(population[i][2], *param_bounds["Kd"])

        score = fitness(*population[i])
        if score < pbest_scores[i]:
            pbest[i] = population[i].copy()
            pbest_scores[i] = score
            if score < gbest_score:
                gbest = population[i].copy()
                gbest_score = score

    # 每10代输出top10
    if iteration % 10 == 0:
        sorted_idx = np.argsort(pbest_scores)
        print(f"Iteration {iteration}: Top 10 Scores")
        for rank, idx in enumerate(sorted_idx[:10], 1):
            p = pbest[idx]
            s = pbest_scores[idx]
            print(f"# {rank} | Score: {s:.4f} | Kp={p[0]:.4f}, Ki={p[1]:.6f}, Kd={p[2]:.2f}")

# ========== 输出最优解 ========== #
os, sse, ts = simulate_pid(*gbest)
print("\n 最佳PID参数:")
print(f"Kp = {gbest[0]:.4f}, Ki = {gbest[1]:.6f}, Kd = {gbest[2]:.2f}")
print(f"超调量: {os:.2f}%")
print(f"稳态误差: {sse:.4f} °C")
print(f"调节时间: {ts:.2f} 秒")

# ========== 绘图 ========== #
plt.figure(figsize=(10, 5))
C_opt = ctrl.tf([gbest[2], gbest[0], gbest[1]], [1, 0])
sys_cl = ctrl.feedback(ctrl.series(C_opt, G), 1)
t_out, y_out = ctrl.forced_response(sys_cl, T=t_sim, U=u_input)
y_out += T_init

plt.plot(t_out, y_out, label='最优PID响应')
plt.axhline(T_target, color='r', linestyle='--', label='目标温度')
plt.xlabel("时间 (秒)")
plt.ylabel("温度 (°C)")
plt.title("PID闭环响应（粒子群优化）")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

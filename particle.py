import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from tqdm import tqdm

# 被控对象 G(s)
K_plant = 36.1072
tau = 3272.4787
L = 68.2818
plant_nominal = ctrl.tf([K_plant], [tau, 1])
pade_num, pade_den = ctrl.pade(L, 2)
delay = ctrl.tf(pade_num, pade_den)
G = ctrl.series(delay, plant_nominal)

# 控制目标和仿真参数
T_init = 16.85
T_target = 35.0
T_step = T_target - T_init
t_sim = np.linspace(0, 20000, 10000)
u_input = np.ones_like(t_sim) * T_step

# PID参数边界
param_bounds = {
    "Kp": (0.0, 1.0),
    "Ki": (0.0, 0.001),
    "Kd": (0.0, 500.0)
}

def simulate_pid(Kp, Ki, Kd):
    """仿真PID控制系统，返回超调量、稳态误差、调节时间"""
    try:
        C = ctrl.tf([Kd, Kp, Ki], [1, 0])
        sys_cl = ctrl.feedback(ctrl.series(C, G), 1)
        t, y = ctrl.forced_response(sys_cl, T=t_sim, U=u_input)
        y = y + T_init

        overshoot = (np.max(y) - T_target) / T_target * 100
        steady_state_error = abs(y[-1] - T_target)
        tolerance = 0.05 * T_target
        indices_outside = np.where(np.abs(y - T_target) > tolerance)[0]

        if len(indices_outside) == 0:
            settling_time = 0
        else:
            last_outside_index = indices_outside[-1]
            if last_outside_index == len(y) - 1:
                settling_time = np.inf
            else:
                settling_time = t[last_outside_index + 1]

        return overshoot, steady_state_error, settling_time
    except Exception:
        # 任何异常，返回大惩罚
        return np.inf, np.inf, np.inf

def fitness(Kp, Ki, Kd):
    """适应度函数，越小越好"""
    overshoot, sse, ts = simulate_pid(Kp, Ki, Kd)
    if np.isinf(overshoot) or np.isinf(sse) or np.isinf(ts):
        return 1e6  # 非法解惩罚
    # 目标：无超调（绝对值），小稳态误差，快调节时间
    # 这里用加权和
    score = abs(overshoot)*1000 + sse*10000 + ts
    return score

# 以优良参数为基准初始化族群
base = np.array([0.5, 0.0001, 50.0])
pop_size = 30
noise_scale = np.array([0.1, 0.00005, 5.0])

population = [base]
for _ in range(pop_size - 1):
    perturb = noise_scale * (2 * np.random.rand(3) - 1)
    particle = base + perturb
    # 限制边界
    particle[0] = np.clip(particle[0], *param_bounds["Kp"])
    particle[1] = np.clip(particle[1], *param_bounds["Ki"])
    particle[2] = np.clip(particle[2], *param_bounds["Kd"])
    population.append(particle)
population = np.array(population)

# PSO参数
w = 0.7  # 惯性权重
c1 = 1.5  # 个体学习因子
c2 = 1.5  # 社会学习因子
max_iter = 100

# 初始化速度、个体最优和全局最优
velocity = np.zeros_like(population)
pbest = population.copy()
pbest_scores = np.array([fitness(*p) for p in population])
gbest_index = np.argmin(pbest_scores)
gbest = pbest[gbest_index].copy()
gbest_score = pbest_scores[gbest_index]

# 迭代优化
for iteration in tqdm(range(1, max_iter + 1), desc="优化PID"):
    for i in range(pop_size):
        r1, r2 = np.random.rand(), np.random.rand()
        velocity[i] = (w * velocity[i] +
                       c1 * r1 * (pbest[i] - population[i]) +
                       c2 * r2 * (gbest - population[i]))

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

    # 每10代打印Top10
    if iteration % 10 == 0:
        sorted_indices = np.argsort(pbest_scores)
        print(f"Iteration {iteration}: Top 10 Scores")
        for rank, idx in enumerate(sorted_indices[:10], 1):
            p = pbest[idx]
            s = pbest_scores[idx]
            print(f"# {rank} | Score: {s:.4f} | Kp={p[0]:.4f}, Ki={p[1]:.6f}, Kd={p[2]:.2f}")

# 最终最优结果展示及绘图
os, sse, ts = simulate_pid(*gbest)
print("\n最佳PID参数:")
print(f"Kp={gbest[0]:.4f}, Ki={gbest[1]:.6f}, Kd={gbest[2]:.2f}")
print(f"超调量: {os:.2f}%, 稳态误差: {sse:.4f} °C, 调节时间: {ts:.2f} 秒")

plt.figure(figsize=(10, 5))
C_opt = ctrl.tf([gbest[2], gbest[0], gbest[1]], [1, 0])
sys_opt = ctrl.feedback(ctrl.series(C_opt, G), 1)
t_opt, y_opt = ctrl.forced_response(sys_opt, T=t_sim, U=u_input)
y_opt = y_opt + T_init
plt.plot(t_opt, y_opt, label='最优PID响应曲线')
plt.axhline(T_target, color='r', linestyle='--', label='目标温度')
plt.xlabel('时间 (秒)')
plt.ylabel('温度 (°C)')
plt.title('粒子群优化后的PID控制响应')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# 被控对象 G(s)
K_plant = 36.1072
tau = 3272.4787
L = 68.2818
plant_nominal = ctrl.tf([K_plant], [tau, 1])
pade_num, pade_den = ctrl.pade(L, 2)
delay = ctrl.tf(pade_num, pade_den)
G = ctrl.series(delay, plant_nominal)

# 控制目标
T_init = 16.85
T_target = 35.0
T_step = T_target - T_init
t_sim = np.linspace(0, 20000, 10000)
u_input = np.ones_like(t_sim) * T_step

def simulate_pid(Kp, Ki, Kd):
    # 构造PID控制器
    C = ctrl.tf([Kd, Kp, Ki], [1, 0])
    sys_cl = ctrl.feedback(ctrl.series(C, G), 1)

    # 仿真响应
    t, y = ctrl.forced_response(sys_cl, T=t_sim, U=u_input)
    y = y + T_init  # 转换成实际温度

    # 计算指标
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

    # 先打印参数和指标
    print(f"超调量: {overshoot:.2f}%")
    print(f"稳态误差: {steady_state_error:.4f} °C")
    print(f"调节时间: {settling_time:.2f} 秒")

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(t, y, label='系统温度响应')
    plt.axhline(T_target, color='r', linestyle='--', label='目标温度')
    plt.xlabel('时间 (秒)')
    plt.ylabel('温度 (°C)')
    plt.title(f'PID控制响应\nKp={Kp}, Ki={Ki}, Kd={Kd}')
    plt.grid(True)
    plt.legend()
    plt.show()

    return overshoot, steady_state_error, settling_time

# 测试示例
Kp, Ki, Kd = 1.0, 0.000302, 37.52
os, sse, ts = simulate_pid(Kp, Ki, Kd)

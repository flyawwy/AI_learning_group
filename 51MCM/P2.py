import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pyswarm import pso
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx', sheet_name='表2 (Table 2)')
times = df['时间 t (Time t)'].values
F_true = df['主路5的车流量 (Traffic flow on the Main road 5)'].values

# 定义Huber损失函数（鲁棒优化）
def huber_loss(y_true, y_pred, delta=10.0):
    error = y_true - y_pred
    return np.where(np.abs(error) <= delta, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))

# 定义支路流量函数
# 支路1（稳定）
def branch1(t, C1):
    return C1 if np.isscalar(t) else np.full_like(t, C1)

# 支路2（两段线性增长 + 中间稳定）
def branch2(t, a1, b1, a2, b2, t0, t1, C2):
    result = np.zeros_like(t)
    result[t <= t0] = a1 * t[t <= t0] + b1
    result[(t > t0) & (t < t1)] = C2
    result[t >= t1] = a2 * t[t >= t1] + b2
    return result

# 支路3（先线性增长后稳定）
def branch3(t, a3, b3, t2, C3):
    result = np.zeros_like(t)

    # 线性增长部分
    linear_growth_mask = t < t2
    result[linear_growth_mask] = a3 * t[linear_growth_mask] + b3

    # 稳定部分
    stable_mask = t >= t2
    result[stable_mask] = C3

    return result

# 支路4（周期性规律）
def branch4(t, A, B, omega, phi):
    return A * np.sin(omega * t + phi) + B

# 总流量模型
def total_flow(t, params):
    C1, a1, b1, a2, b2, t0, t1, C2, a3, b3, t2, C3, A, B, omega, phi = params
    t_delayed = t - 2  # 支路1和2延迟2分钟
    return (
        branch1(t_delayed, C1) +
        branch2(t_delayed, a1, b1, a2, b2, t0, t1, C2) +
        branch3(t, a3, b3, t2, C3) +  # 使用新的支路3函数
        branch4(t, A, B, omega, phi)
    )

# 目标函数
def objective(params, times, F_measured):
    predicted = total_flow(times, params)
    loss = huber_loss(F_measured, predicted, delta=10.0).sum()
    return loss

# 参数边界约束
bounds = [
    (5, 15),        # C1 (支路1稳定值)
    (0.2, 0.5),     # a1 (支路2第一段斜率)
    (10, 30),       # b1 (支路2第一段截距)
    (0.2, 0.5),     # a2 (支路2第三段斜率)
    (10, 30),       # b2 (支路2第三段截距)
    (45, 55),       # t0 (支路2第一段结束时间)
    (70, 78),       # t1 (支路2第三段开始时间)
    (30, 60),       # C2 (支路2中间稳定值)
    (0.2, 0.5),     # a3 (支路3线性增长斜率)
    (10, 30),       # b3 (支路3线性增长截距)
    (60, 80),       # t2 (支路3转折点)
    (30, 60),       # C3 (支路3稳定值)
    (3, 10),        # A (支路4振幅)
    (20, 35),       # B (支路4基线)
    (0.3, 0.6),     # omega (支路4频率)
    (-np.pi, np.pi) # phi (支路4相位角)
]

# 动态惯性权重的PSO优化
def pso_with_adaptive_inertia():
    best_error = np.inf
    best_params = None

    for omega in [0.9, 0.75, 0.6, 0.45, 0.3]:
        xopt_pso, fopt_pso = pso(
            objective,
            lb=[b[0] for b in bounds],
            ub=[b[1] for b in bounds],
            args=(times, F_true),
            swarmsize=200,
            omega=omega,
            minfunc=1e-8,
            minstep=1e-8,
            maxiter=1500,
        )

        result_local = minimize(
            objective,
            xopt_pso,
            args=(times, F_true),
            method='L-BFGS-B',
            bounds=bounds
        )

        if result_local.fun < best_error:
            best_error = result_local.fun
            best_params = result_local.x

    return best_params, best_error

# 执行优化
best_params, best_error = pso_with_adaptive_inertia()

print("最终最优参数:", best_params)
print("最终最小误差:", best_error)

# 计算指定时刻的支路流量值
def compute_branch_flows(t, params):
    C1, a1, b1, a2, b2, t0, t1, C2, a3, b3, t2, C3, A, B, omega, phi = params

    t_scalar = False
    if np.isscalar(t):
        t_scalar = True
        t_array = np.array([t])
    else:
        t_array = np.array(t)

    t_delayed = t_array - 2

    flows = {
        "Branch1": branch1(t_delayed, C1),
        "Branch2": branch2(t_delayed, a1, b1, a2, b2, t0, t1, C2),
        "Branch3": branch3(t_array, a3, b3, t2, C3),
        "Branch4": branch4(t_array, A, B, omega, phi)
    }

    if t_scalar:
        return {k: v[0] for k, v in flows.items()}
    else:
        return flows

# 计算7:30（t=30）和8:30（t=90）的支路流量值
t_7_30 = 30  # 7:30 = 30分钟
t_8_30 = 90  # 8:30 = 90分钟

flow_7_30 = compute_branch_flows(t_7_30, best_params)
flow_8_30 = compute_branch_flows(t_8_30, best_params)

print("7:30 各支路流量值:", flow_7_30)
print("8:30 各支路流量值:", flow_8_30)

# 绘制主路流量拟合结果
predicted_F = total_flow(times, best_params)
plt.figure(figsize=(12, 6))
plt.plot(times, F_true, label='实际主路流量')
plt.plot(times, predicted_F, label='拟合主路流量')

plt.xlabel('时间（分钟）')
plt.ylabel('流量（辆/2分钟）')
plt.legend()
plt.title('主路及各支路流量拟合结果（优化后）')
plt.grid()
plt.show()

# 残差分析
residuals = F_true - predicted_F
plt.figure(figsize=(12, 6))
plt.plot(times, residuals, label='残差', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('时间（分钟）')
plt.ylabel('残差')
plt.title('残差分布')
plt.legend()
plt.grid()
plt.show()

# 误差分布直方图
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('残差')
plt.ylabel('频率')
plt.title('残差分布直方图')
plt.grid()
plt.show()
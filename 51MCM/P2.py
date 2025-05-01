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
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    return np.where(np.abs(error) <= delta, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))

# 定义支路流量函数（向量化）
# 支路流量函数（修复 branch1）
def branch1(t, C1):
    return C1 if np.isscalar(t) else np.full_like(t, C1)

def branch2(t, a1, b1, t0, C2, smooth_width=10):
    x = (t - t0) / smooth_width
    sigmoid = 1 / (1 + np.exp(-x))
    return (1 - sigmoid) * (a1 * t + b1) + sigmoid * C2

def branch3(t, a2, b2, t1, C3, smooth_width=10):
    x = (t - t1) / smooth_width
    sigmoid = 1 / (1 + np.exp(-x))
    return (1 - sigmoid) * (a2 * t + b2) + sigmoid * C3

def branch4(t, A, B, omega, phi):
    return A * np.sin(omega * t + phi) + B

# 总流量模型
def total_flow(t, params):
    C1, a1, b1, t0, C2, a2, b2, t1, C3, A, B, omega, phi = params
    t_delayed = t - 2  # 支路1和2延迟2分钟
    return (
        branch1(t_delayed, C1) +
        branch2(t_delayed, a1, b1, t0, C2) +
        branch3(t, a2, b2, t1, C3) +
        branch4(t, A, B, omega, phi)
    )

# 目标函数（向量化计算）
def objective(params, times, F_measured):
    predicted = total_flow(times, params)
    loss = huber_loss(F_measured, predicted, delta=5.0).sum()
    return loss

# 参数边界约束（合并冗余范围）
bounds = [
    (3, 20),       # C1
    (0.2, 0.6),    # a1
    (5, 60),       # b1
    (50, 80),      # t0
    (40, 120),     # C2
    (0.2, 0.7),    # a2
    (5, 60),       # b2
    (70, 100),     # t1
    (40, 120),     # C3
    (5, 60),       # A
    (25, 65),      # B
    (0.2, 0.5),    # omega
    (-np.pi, np.pi) # phi
]

# 动态惯性权重的PSO优化（简化逻辑）
def pso_with_adaptive_inertia():
    best_error = np.inf
    best_params = None

    for omega in [0.9, 0.7, 0.5, 0.3]:  # 固定惯性权重列表
        xopt_pso, fopt_pso = pso(
            objective,
            lb=[b[0] for b in bounds],
            ub=[b[1] for b in bounds],
            args=(times, F_true),
            swarmsize=150,  # 减少种群规模
            omega=omega,
            minfunc=1e-8,
            minstep=1e-8,
            maxiter=1000,
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
    C1, a1, b1, t0, C2, a2, b2, t1, C3, A, B, omega, phi = params
    t_delayed = t - 2
    return {
        "Branch1": branch1(t_delayed, C1),
        "Branch2": branch2(t_delayed, a1, b1, t0, C2),
        "Branch3": branch3(t, a2, b2, t1, C3),
        "Branch4": branch4(t, A, B, omega, phi)
    }

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

# 计算并绘制各支路流量曲线
branch_flows = {
    "Branch1": total_flow(times, best_params * np.array([1] + [0]*12)),  # Branch1 only
    "Branch2": total_flow(times, best_params * np.array([0,1,1,1,1] + [0]*8)),  # Branch2 only
    "Branch3": total_flow(times, best_params * np.array([0]*5 + [1,1,1,1] + [0]*4)),  # Branch3 only
    "Branch4": total_flow(times, best_params * np.array([0]*9 + [1,1,1,1]))   # Branch4 only
}
for name, flow in branch_flows.items():
    plt.plot(times, flow, label=f'支路流量 - {name}')

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
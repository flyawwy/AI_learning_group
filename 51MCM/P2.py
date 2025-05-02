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

# 支路流量函数定义
def branch1(t, C1):
    return C1 if np.isscalar(t) else np.full_like(t, C1)

def branch2(t, a1, b1, a2):
    C2 = a1 * 24 + b1 # 稳定阶段的车流量
    result = np.zeros_like(t)

    mask_growth = (t <= 24)
    mask_stable = (t > 24) & (t <= 37)
    mask_decrease = (t > 37)

    result[mask_growth] = a1 * t[mask_growth] + b1
    result[mask_stable] = C2
    # 确保线性减少阶段在 t=37 处与稳定阶段的值相等
    result[mask_decrease] = a2 * (t[mask_decrease] - 37) + C2

    return result

def branch3(t, a3, b3, t2, C3):
    result = np.zeros_like(t, dtype=float)

    # 确保在 t2 时刻的车流量等于 C3
    C2_at_t2 = a3 * t2 + b3

    linear_growth_mask = t < t2
    stable_mask = t >= t2

    # 在线性增长阶段，使用原始公式
    result[linear_growth_mask] = a3 * t[linear_growth_mask] + b3

    # 在稳定阶段，直接设置为 C2_at_t2 而不是 C3，从而保证连续性
    result[stable_mask] = C2_at_t2

    return result

# 傅里叶级数支路4
# params_fourier: [N, T, A0, A1, ..., AN, B1, ..., BN]
def branch4(t, params_fourier):
    N = int(round(params_fourier[0]))  # 阶数取整
    T = params_fourier[1]
    A0 = params_fourier[2]
    A = params_fourier[3:3+N]
    B = params_fourier[3+N:3+2*N]
    result = np.full_like(t, A0, dtype=float)
    for n in range(1, N+1):
        result += A[n-1] * np.cos(2 * np.pi * n * t / T) + B[n-1] * np.sin(2 * np.pi * n * t / T)
    return result

# 修改total_flow，适配新的支路4
# params = [C1, a1, b1, a2, a3, b3, t2, C3, N, T, A0, A1...AN, B1...BN]
def total_flow(t, params):
    C1, a1, b1, a2, a3, b3, t2, C3 = params[:8]
    N = int(round(params[8]))
    T = params[9]
    A0 = params[10]
    A = params[11:11+N]
    B = params[11+N:11+2*N]
    params_fourier = [N, T, A0] + list(A) + list(B)
    t_delayed = np.maximum(0, t - 2)
    return (
        branch1(t_delayed, C1) +
        branch2(t_delayed, a1, b1, a2) +
        branch3(t, a3, b3, t2, C3) +
        branch4(t, params_fourier)
    )

def objective(params, times, F_measured):
    predicted = total_flow(times, params)
    loss = huber_loss(F_measured, predicted, delta=10.0).sum()
    return loss

# 参数边界约束
bounds = [
    (5, 10),
    (0.4, 0.5),
    (10, 20),
    (0.2, 0.3),
    (0.4, 0.5),
    (10, 20),
    (35, 45),
    (15, 20),
    (3, 10),  # N的边界
    (10, 45),  # T的边界
    (0, 10),  # A0的边界
    (-10, 10),  # A1的边界
    (-10, 10),  # A2的边界
    (-10, 10),  # A3的边界
    (-10, 10),  # A4的边界
    (-10, 10),  # A5的边界
    (-10, 10),  # A6的边界
    (-10, 10),  # A7的边界
    (-10, 10),  # A8的边界
    (-10, 10),  # A9的边界
    (-10, 10),  # A10的边界
    (-10, 10),  # B1的边界
    (-10, 10),  # B2的边界
    (-10, 10),  # B3的边界
    (-10, 10),  # B4的边界
    (-10, 10),  # B5的边界
    (-10, 10),  # B6的边界
    (-10, 10),  # B7的边界
    (-10, 10),  # B8的边界
    (-10, 10),  # B9的边界
    (-10, 10),  # B10的边界
]

initial_guess = [5.0008, 0.4548, 10., 0.2, 0.4856, 10.5713, 39.4208, 15., 5, 25, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def optimize():
    xopt_pso, fopt_pso = pso(objective, lb=[b[0] for b in bounds], ub=[b[1] for b in bounds],
                             args=(times, F_true), swarmsize=200, omega=0.5, minfunc=1e-8, maxiter=1500)

    result_local = minimize(objective, xopt_pso, args=(times, F_true), method='L-BFGS-B', bounds=bounds)
    return result_local.x, result_local.fun

best_params, best_error = optimize()

print("最终最优参数:", best_params)
print("最终最小误差:", best_error)

def compute_branch_flows(t, params):
    t_array = np.array([t]) if np.isscalar(t) else np.array(t)
    N = int(round(params[8]))
    T = params[9]
    A0 = params[10]
    A = params[11:11+N]
    B = params[11+N:11+2*N]
    params_fourier = [N, T, A0] + list(A) + list(B)
    flows = {
        "Branch1": branch1(t_array - 2, params[0]),
        "Branch2": branch2(t_array - 2, *params[1:4]),
        "Branch3": branch3(t_array, *params[4:8]),
        "Branch4": branch4(t_array, params_fourier)
    }
    return {k: v[0] if np.isscalar(t) else v for k, v in flows.items()}

flow_7_30 = compute_branch_flows(15, best_params)
flow_8_30 = compute_branch_flows(45, best_params)

print("7:30 各支路流量值:", flow_7_30)
print("8:30 各支路流量值:", flow_8_30)

def format_expression_branch1(C1):
    return f"支路1: f(t) = {C1:.2f} （稳定）"

def format_expression_branch2(a1, b1, a2):
    C2 = a1 * 24 + b1
    return (f"支路2: f(t) = \\begin{{cases}} "
            f"{a1:.2f}t + {b1:.2f}, & t < 24 \\\\ "
            f"{C2:.2f}, & 24 \\leq t < 37 \\\\ "
            f"-{a2:.2f}t + {a2*37 + C2:.2f}, & t \\geq 37 "
            "\\end{cases}")

def format_expression_branch3(a3, b3, t2, C3):
    return (f"支路3: f(t) = \\begin{{cases}} "
            f"{a3:.2f}t + {b3:.2f}, & t < {int(t2)} \\\\ "
            f"{C3:.2f}, & t \\geq {int(t2)} "
            "\\end{cases}")

def format_expression_branch4(N, T, A0, A, B):
    terms = [f"{A0:.2f}"]
    for n in range(1, N+1):
        terms.append(f"{A[n-1]:.2f} \\cos\\left(\\frac{{2\\pi {n} t}}{{{T}}}\\right)")
        terms.append(f"{B[n-1]:.2f} \\sin\\left(\\frac{{2\\pi {n} t}}{{{T}}}\\right)")
    return f"支路4: f(t) = " + " + ".join(terms)

C1_opt, a1_opt, b1_opt, a2_opt, a3_opt, b3_opt, t2_opt, C3_opt = best_params[:8]
N_opt = int(round(best_params[8]))
T_opt = best_params[9]
A0_opt = best_params[10]
A_opt = best_params[11:11+N_opt]
B_opt = best_params[11+N_opt:11+2*N_opt]

print("\n--- 各支路车流量函数表达式 ---")
print(format_expression_branch1(C1_opt))
print(format_expression_branch2(a1_opt, b1_opt, a2_opt))
print(format_expression_branch3(a3_opt, b3_opt, t2_opt, C3_opt))
print(format_expression_branch4(N_opt, T_opt, A0_opt, A_opt, B_opt))

# 绘图
predicted_F = total_flow(times, best_params)
plt.figure(figsize=(12, 6))
plt.plot(times, F_true, label='实际主路流量', color='blue')
plt.plot(times, predicted_F, label='拟合主路流量', linestyle='--', color='green')

flows = compute_branch_flows(times, best_params)
for name, flow in flows.items():
    plt.plot(times, flow, label=name)

plt.xlabel('时间（分钟）')
plt.ylabel('流量（辆/2分钟）')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('主路及各支路流量拟合结果（优化后）')
plt.grid()
plt.tight_layout()
plt.show()

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

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('残差')
plt.ylabel('频率')
plt.title('残差分布直方图')
plt.grid()
plt.show()
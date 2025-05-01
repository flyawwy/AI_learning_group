import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx', sheet_name='表1 (Table 1)')
times = df['时间 t (Time t)'].values
main_road_flow = df['主路3的车流量 (Traffic flow on the Main road 3)'].values

def model(params, t):
    """
    模型函数：计算给定参数下的预测值。
    :param params: 参数列表 [a1, b1, a2, b2, c2, d2, t0] 对应支路1和支路2的参数
    :param t: 时间序列
    :return: 主路车流量预测值
    """
    a1, b1, a2, b2, c2, d2, t0 = params
    branch1_flow = a1 * np.array(t) + b1
    branch2_flow = np.where(np.array(t) < t0, a2 * np.array(t) + b2, c2 * np.array(t) + d2)
    return branch1_flow + branch2_flow

def residuals(params, times, main_road_flow):
    """
    残差函数：计算实际值与模型预测值之间的差异。
    :param params: 参数列表
    :param times: 时间序列
    :param main_road_flow: 主路车流量的实际观测值
    :return: 残差数组
    """
    return model(params, times) - np.array(main_road_flow)

# 定义目标函数（MSE损失）
def objective(params):
    return np.sum(residuals(params, times, main_road_flow) ** 2)

# 增加支路2负值惩罚项的目标函数
def objective_with_penalty(params):
    loss = np.sum(residuals(params, times, main_road_flow) ** 2)
    # 提取参数
    a2, b2, c2, d2, t0 = params[2], params[3], params[4], params[5], params[6]
    t_arr = np.array(times)
    # 支路2下降阶段
    t_after = t_arr[t_arr >= t0]
    q2_after = c2 * t_after + d2
    penalty = np.sum(np.abs(q2_after[q2_after < 0])) * 10
    return loss + penalty

# 参数边界
bounds = [
    (0.1, None),    # a1 >= 0.1
    (0, None),      # b1 >= 0
    (0.1, None),    # a2 >= 0.1
    (0, None),      # b2 >= 0
    (-2, -0.1),     # c2 in [-2, -0.1]
    (0, None),      # d2 >= 0
    (20, 40)        # t0 in [20, 40]
]

# 新的初始猜测
initial_params = [0.6, 0, 0.8, 15, -1, 75, 30]

# 执行带约束优化（使用带惩罚项的目标函数）
result = minimize(objective_with_penalty, initial_params, bounds=bounds)
print("最优参数:", result.x)

# 计算并输出支路1和支路2的车流量
opt_params = result.x
a1, b1, a2, b2, c2, d2, t0 = opt_params
branch1_flow = a1 * np.array(times) + b1
branch2_flow = np.where(np.array(times) < t0, a2 * np.array(times) + b2, c2 * np.array(times) + d2)

print("支路1车流量:", branch1_flow)
print("支路2车流量:", branch2_flow)
print(f"支路1车流量表达式: Q1(t) = {a1:.2f} * t + {b1:.2f}")
print(f"支路2车流量表达式: Q2(t) = {a2:.2f} * t + {b2:.2f} (t < {t0:.2f}) 或 Q2(t) = {c2:.2f} * t + {d2:.2f} (t >= {t0:.2f})")

# 误差分析
mse = np.mean((main_road_flow - (branch1_flow + branch2_flow)) ** 2)
mae = np.mean(np.abs(main_road_flow - (branch1_flow + branch2_flow)))
residual = (branch1_flow + branch2_flow) - main_road_flow
print(f"均方误差 (MSE): {mse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print("残差数组:", residual)

# 残差分布可视化
plt.figure(figsize=(10, 4))
plt.plot(times, residual, marker='o', linestyle='-', color='purple')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("时间（分钟，7:00起计）")
plt.ylabel("残差 (预测-观测)")
plt.title("残差分布图")
plt.grid(True)
plt.show()

# 可视化对比
plt.figure(figsize=(12, 6))
plt.plot(times, main_road_flow, label="主路观测车流量", color='black')
plt.plot(times, branch1_flow, label="支路1估计车流量", linestyle='--')
plt.plot(times, branch2_flow, label="支路2估计车流量", linestyle='--')
plt.plot(times, branch1_flow + branch2_flow, label="支路1+2总和", color='green', linestyle=':')
plt.legend()
plt.xlabel("时间（分钟，7:00起计）")
plt.ylabel("车流量")
plt.title("主路车流量与支路估计车流量对比")
plt.grid(True)
plt.show()
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ==================== 1. 数据读取与预处理 ====================
try:
    df = pd.read_excel('C:\\Users\Lenovo\Desktop\\2025-51MCM-Problem A\附件(Attachment).xlsx', sheet_name='表3 (Table 3)')
    print("=== 数据读取成功 ===")
    print("前5行数据样例：")
    print(df.head())
except Exception as e:
    print(f"数据读取失败: {e}")
    exit()

# 检查数据列
required_columns = ['时间 t (Time t)', '主路4的车流量 (Traffic flow on the Main road 4)']
if not all(col in df.columns for col in required_columns):
    print("错误：数据列缺失，请检查Excel文件")
    print("现有列名:", df.columns)
    exit()

times = df['时间 t (Time t)'].values*2
main_road_flow = df['主路4的车流量 (Traffic flow on the Main road 4)'].values

print(f"\n数据统计：")
print(f"时间范围：{times.min()} 到 {times.max()} 分钟")
print(f"车流量范围：{main_road_flow.min():.1f} 到 {main_road_flow.max():.1f}")

# ==================== 2. 模型定义 ====================
def branch1_flow(t, params):
    """支路1：五段分段函数"""
    a1, b1, a2, b2, c1, a3, b3 = params[:7]
    return np.piecewise(t,
                        [t < 24, (t >= 24) & (t < 48), (t >= 48) & (t < 72), (t >= 72) & (t < 96), t >= 96],
                        [0,
                         lambda x: a1 * (x - 24) + b1,
                         lambda x: a2 * (x - 48) + b2,
                         lambda x: c1,
                         lambda x: a3 * (x - 96) + b3]
                        )

def branch2_flow(t):
    """支路2：三段分段函数（固定形式）"""
    return np.piecewise(t,
                        [t < 72, (t >= 72) & (t < 96), t >= 96],
                        [lambda x: 0.5 * x,
                         lambda x: 36,
                         lambda x: -0.5 * (x - 96) + 36]
                        )

def branch3_flow(t, params):
    """支路3：信号灯控制周期函数"""
    amp, phase, t_g = params[7], params[8], params[9]
    flow = np.zeros_like(t)
    for i, time in enumerate(t):
        if 0 <= (time - t_g) % 18 < 10:  # 绿灯期10分钟
            flow[i] = amp * np.sin(2 * np.pi * (time - t_g) / 18 + phase)
    return flow

def total_model(params, t):
    """综合流量模型"""
    # 计算各支路流量
    b1 = branch1_flow(t - 2, params)  # 支路1有2分钟延迟
    b2 = branch2_flow(t - 2)  # 支路2有2分钟延迟
    b3 = branch3_flow(t, params)  # 支路3无延迟

    # 处理延迟后的边界值
    b1 = np.interp(t - 2, t, branch1_flow(t, params), left=0, right=0)
    b2 = np.interp(t - 2, t, branch2_flow(t), left=0, right=0)

    return b1 + b2 + b3

# ==================== 3. 参数优化 ====================
print("\n=== 开始参数优化 ===")
initial_params = [1.0, 20, -0.5, 50, 30, -0.3, 40, 10, 0, 6]  # 初始猜测
bounds = (
    [0, 0, -np.inf, 0, 0, -np.inf, 0, 0, -np.inf, 0],  # 下限
    [np.inf] * 10  # 上限
)

result = least_squares(
    lambda p, t, y: total_model(p, t) - y,
    initial_params,
    args=(times, main_road_flow),
    bounds=bounds,
    verbose=1
)
opt_params = result.x

print("\n=== 优化结果 ===")
param_names = ['a1', 'b1', 'a2', 'b2', 'c1', 'a3', 'b3', 'amplitude', 'phase', 't_g']
for name, value in zip(param_names, opt_params):
    print(f"{name}: {value:.4f}")

# ==================== 4. 结果可视化 ====================
plt.figure(figsize=(12, 6))
plt.rc("font",family="DengXian")
plt.rcParams['axes.unicode_minus']=False

# 主路拟合对比
plt.subplot(3, 1, 1)
plt.plot(times, main_road_flow, 'k-', label='观测值', linewidth=2)
plt.plot(times, total_model(opt_params, times), 'r--', label='拟合值', linewidth=1.5)
plt.title("主路车流量 - 观测值与拟合值对比", fontsize=12)
plt.xlabel("时间（7:00后分钟数）", fontsize=10)
plt.ylabel("车流量", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)

# 各支路流量
plt.subplot(3, 1, 2)
plt.plot(times, branch1_flow(times, opt_params), label='支路1', linestyle='-')
plt.plot(times, branch2_flow(times), label='支路2', linestyle='--')
plt.plot(times, branch3_flow(times, opt_params), label='支路3', linestyle=':')
plt.title("各支路车流量变化", fontsize=12)
plt.xlabel("时间（7:00后分钟数）", fontsize=10)
plt.ylabel("车流量", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)

# 残差分析
plt.subplot(3, 1, 3)
residuals = main_road_flow - total_model(opt_params, times)
plt.stem(times, residuals, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.axhline(0, color='r', linestyle='--')
plt.title("残差分布", fontsize=12)
plt.xlabel("时间（7:00后分钟数）", fontsize=10)
plt.ylabel("残差", fontsize=10)
plt.grid(True)

plt.tight_layout()
plt.show()

# ==================== 5. 输出各支路表达式 ====================
a1, b1, a2, b2, c1, a3, b3, amp, phase, t_g = opt_params

print("\n" + "=" * 60)
print(" " * 20 + "各支路车流量表达式")
print("=" * 60)

print(f"""
【支路1】五段分段线性函数：
Q₁(t) = 
⎧
⎪   0,                                 当 t < 12 分钟
⎪
⎨   {a1:.4f} × (t - 12) + {b1:.4f},    当 12 ≤ t < 36 分钟
⎪
⎪   {a2:.4f} × (t - 36) + {b2:.4f},    当 36 ≤ t < 60 分钟
⎪
⎩   {c1:.4f},                          当 60 ≤ t < 84 分钟
   {a3:.4f} × (t - 84) + {b3:.4f},    当 t ≥ 84 分钟
""")

print(f"""
【支路2】三段分段函数（固定形式）：
Q₂(t) = 
⎧
⎪   0.5 × t,                          当 t < 72 分钟
⎨
⎪   36,                               当 72 ≤ t < 96 分钟
⎪
⎩   -0.5 × (t - 96) + 36,             当 t ≥ 96 分钟
""")

print(f"""
【支路3】信号灯控制周期函数：
Q₃(t) = 
⎧
⎪   {amp:.4f} × sin[2π(t - {t_g:.4f})/18 + {phase:.4f}],  当 (t - {t_g:.4f}) mod 18 ∈ [0,10) 分钟
⎨
⎪   0,                                     其他时间（红灯期）
⎩
""")

print(f"""
【主路4】流量关系式：
Q₄(t) = Q₁(t - 2) + Q₂(t - 2) + Q₃(t)
（支路1和支路2有2分钟延迟）
""")

# ==================== 6. 模型评估 ====================
rmse = np.sqrt(np.mean(residuals ** 2))
r_squared = 1 - np.sum(residuals ** 2) / np.sum((main_road_flow - np.mean(main_road_flow)) ** 2)

print("\n" + "=" * 60)
print(" " * 20 + "模型评估指标")
print("=" * 60)
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"决定系数 (R²): {r_squared:.4f}")
print("=" * 60)

# ==================== 7. 关键时间点流量计算 ====================
def calculate_flow(t_minutes):
    """计算指定时刻的各支路流量"""
    t = float(t_minutes)
    b1 = branch1_flow(np.array([t - 2]), opt_params)[0]  # 支路1有2分钟延迟
    b2 = branch2_flow(np.array([t - 2]))[0]  # 支路2有2分钟延迟
    b3 = branch3_flow(np.array([t]), opt_params)[0]  # 支路3无延迟

    return {
        '支路1': b1,
        '支路2': b2,
        '支路3': b3,
        '总和': b1 + b2 + b3
    }

print("\n关键时间点车流量：")
for time_point in [30, 90]:  # 7:30和8:30对应t=30和t=90
    flow = calculate_flow(time_point)
    print(f"\n时间 t = {time_point} 分钟（{7 + time_point // 60}:{time_point % 60:02d}）")
    print(f"支路1流量: {flow['支路1']:.4f}")
    print(f"支路2流量: {flow['支路2']:.4f}")
    print(f"支路3流量: {flow['支路3']:.4f}")
    print(f"流量总和: {flow['总和']:.4f}")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pyswarm import pso
import matplotlib.pyplot as plt

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
_df = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx', sheet_name='表2 (Table 2)')
times = _df['时间 t (Time t)'].values
F_true = _df['主路5的车流量 (Traffic flow on the Main road 5)'].values

def huber_loss(y_true, y_pred, delta=10.0):
    error = y_true - y_pred
    return np.where(np.abs(error) <= delta, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))

def branch1(t, C1):
    return C1 if np.isscalar(t) else np.full_like(t, C1)

def branch2(t, a1, b1, a2):
    C2 = a1 * 24 + b1
    result = np.zeros_like(t)
    mask_growth = (t <= 24)
    mask_stable = (t > 24) & (t <= 37)
    mask_decrease = (t > 37)
    result[mask_growth] = a1 * t[mask_growth] + b1
    result[mask_stable] = C2
    result[mask_decrease] = a2 * (t[mask_decrease] - 37) + C2
    return result

def branch3(t, a3, b3, t2):
    result = np.zeros_like(t, dtype=float)
    C2_at_t2 = a3 * t2 + b3
    linear_growth_mask = t < t2
    stable_mask = t >= t2
    result[linear_growth_mask] = a3 * t[linear_growth_mask] + b3
    result[stable_mask] = C2_at_t2
    return result

def branch4(t, params_fourier):
    N = int(round(params_fourier[0]))
    T = params_fourier[1]
    A0 = params_fourier[2]
    A = params_fourier[3:3+N]
    B = params_fourier[3+N:3+2*N]
    result = np.full_like(t, A0, dtype=float)
    for n in range(1, N+1):
        result += A[n-1] * np.cos(2 * np.pi * n * t / T) + B[n-1] * np.sin(2 * np.pi * n * t / T)
    return result

def total_flow(t, params):
    C1, a1, b1, a2, a3, b3, t2 = params[:7]
    N = int(round(params[7]))
    T = params[8]
    A0 = params[9]
    A = params[10:10+N]
    B = params[10+N:10+2*N]
    params_fourier = [N, T, A0] + list(A) + list(B)
    t_delayed = np.maximum(0, t - 2)
    return (
        branch1(t_delayed, C1) +
        branch2(t_delayed, a1, b1, a2) +
        branch3(t, a3, b3, t2) +
        branch4(t, params_fourier)
    )

def objective(params, times, F_measured):
    predicted = total_flow(times, params)
    return huber_loss(F_measured, predicted, delta=10.0).sum()

# 支路流量分解

def compute_branch_flows(t, params):
    t_array = np.array([t]) if np.isscalar(t) else np.array(t)
    N = int(round(params[7]))
    T = params[8]
    A0 = params[9]
    A = params[10:10+N]
    B = params[10+N:10+2*N]
    params_fourier = [N, T, A0] + list(A) + list(B)
    flows = {
        "Branch1": branch1(t_array - 2, params[0]),
        "Branch2": branch2(t_array - 2, *params[1:4]),
        "Branch3": branch3(t_array, *params[4:7]),
        "Branch4": branch4(t_array, params_fourier)
    }
    return {k: v[0] if np.isscalar(t) else v for k, v in flows.items()}

# 公式格式化

def format_expression_branch1(C1):
    return f"支路1: f(t) = {C1:.2f} （稳定）"

def format_expression_branch2(a1, b1, a2):
    C2 = a1 * 24 + b1
    return (f"支路2: f(t) = \\begin{{cases}} "
            f"{a1:.2f}t + {b1:.2f}, & t < 24 \\\\ "
            f"{C2:.2f}, & 24 \\leq t < 37 \\\\ "
            f"-{a2:.2f}t + {a2*37 + C2:.2f}, & t \\geq 37 "
            "\\end{cases}")

def format_expression_branch3(a3, b3, t2):
    C2_at_t2 = a3 * t2 + b3
    return (f"支路3: f(t) = \\begin{{cases}} "
            f"{a3:.2f}t + {b3:.2f}, & t < {int(t2)} \\\\ "
            f"{C2_at_t2:.2f}, & t \\geq {int(t2)} "
            "\\end{cases}")

def format_expression_branch4(N, T, A0, A, B):
    terms = [f"{A0:.2f}"]
    for n in range(1, N+1):
        terms.append(f"{A[n-1]:.2f} \\cos\\left(\\frac{{2\\pi {n} t}}{{{T}}}\\right)")
        terms.append(f"{B[n-1]:.2f} \\sin\\left(\\frac{{2\\pi {n} t}}{{{T}}}\\right)")
    return f"支路4: f(t) = " + " + ".join(terms)

def plot_and_save():
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
    plt.title('主路及各支路流量拟合结果')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./P2/主路及各支路流量拟合结果.png')
    plt.close()

    residuals = F_true - predicted_F
    plt.figure(figsize=(12, 6))
    plt.plot(times, residuals, label='残差', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('时间（分钟）')
    plt.ylabel('残差')
    plt.title('残差分布')
    plt.legend()
    plt.grid()
    plt.savefig('./P2/残差分布.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('频率')
    plt.title('残差分布直方图')
    plt.grid()
    plt.savefig('./P2/残差分布直方图.png')
    plt.close()

    # 输出公式到 result.md
    result_lines = [
        '# Result',
        '',
        f'$ {format_expression_branch1(C1_opt)} $',
        f'$ {format_expression_branch2(a1_opt, b1_opt, a2_opt)} $',
        f'$ {format_expression_branch3(a3_opt, b3_opt, t2_opt)} $',
        f'$ {format_expression_branch4(N_opt, T_opt, A0_opt, A_opt, B_opt)} $',
        '',
        '## 7:30 各支路流量值',
        '',
    ]
    for k, v in flow_7_30.items():
        result_lines.append(f'- {k}: {v:.2f}')
    result_lines.append('')
    result_lines.append('## 8:30 各支路流量值')
    result_lines.append('')
    for k, v in flow_8_30.items():
        result_lines.append(f'- {k}: {v:.2f}')
    result_lines.append('')
    with open('./P2/result.md', 'w', encoding='utf-8') as f:
        for line in result_lines:
            f.write(line + '\n')

if __name__ == "__main__":
    import argparse
    import os

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='P2问题求解工具')
    parser.add_argument('--p5', action='store_true', help='启用P5问题求解（关键采样点分析）')
    parser.add_argument('--method', type=str, default='combined',
                        choices=['derivative', 'peaks', 'breakpoints', 'combined'],
                        help='采样点识别方法（仅在启用P5时有效）')

    args = parser.parse_args()

    # 参数边界和初始猜测
    _bounds = [
        (5, 10), (0.4, 0.5), (10, 20), (0.2, 0.3), (0.4, 0.5), (10, 20), (35, 45),
        (3, 10), (10, 45), (0, 10), *[(-10, 10)]*20
    ]
    _initial_guess = [5.0008, 0.4548, 10., 0.2, 0.4856, 10.5713, 39.4208, 5, 25, 5] + [1]*20

    def optimize():
        xopt_pso, _ = pso(objective, lb=[b[0] for b in _bounds], ub=[b[1] for b in _bounds],
                          args=(times, F_true), swarmsize=200, omega=0.5, minfunc=1e-8, maxiter=1500)
        result_local = minimize(objective, xopt_pso, args=(times, F_true), method='L-BFGS-B', bounds=_bounds)
        return result_local.x, result_local.fun

    # 创建P2目录（如果不存在）
    os.makedirs('./P2', exist_ok=True)

    # 常规P2问题求解
    best_params, best_error = optimize()
    flow_7_30 = compute_branch_flows(15, best_params)
    flow_8_30 = compute_branch_flows(45, best_params)
    C1_opt, a1_opt, b1_opt, a2_opt, a3_opt, b3_opt, t2_opt = best_params[:7]
    N_opt = int(round(best_params[7]))
    T_opt = best_params[8]
    A0_opt = best_params[9]
    A_opt = best_params[10:10+N_opt]
    B_opt = best_params[10+N_opt:10+2*N_opt]

    # 生成常规结果
    plot_and_save()

    # 如果启用了P5问题求解
    if args.p5:
        print("\n正在进行P5问题求解（关键采样点分析）...")
        from P5.key_sampling_points import KeySamplingPointsAnalyzer

        # 创建关键采样点分析器
        analyzer = KeySamplingPointsAnalyzer(data_source='P2')

        # 识别关键采样点
        analyzer.identify_key_points(method=args.method)

        # 评估采样点
        evaluation_results = analyzer.evaluate_sampling_points(original_params=best_params)

        # 可视化结果
        analyzer.visualize_sampling_points(evaluation_results)

        # 生成报告
        analyzer.generate_report(evaluation_results)

        print(f"P5分析完成！关键采样点数量：{len(analyzer.sampling_indices)}")
        print(f"采样点时间索引：{analyzer.sampling_indices}")
        print(f"采样点时间值：{analyzer.sampling_points}")
        print(f"结果已保存到 ./P5/P2/ 目录")
    print("最终最优参数:", best_params)
    print("最终最小误差:", best_error)
    print("7:30 各支路流量值:", flow_7_30)
    print("8:30 各支路流量值:", flow_8_30)
    print("\n--- 各支路车流量函数表达式 ---")
    print(format_expression_branch1(C1_opt))
    print(format_expression_branch2(a1_opt, b1_opt, a2_opt))
    print(format_expression_branch3(a3_opt, b3_opt, t2_opt))
    print(format_expression_branch4(N_opt, T_opt, A0_opt, A_opt, B_opt))
    plot_and_save()
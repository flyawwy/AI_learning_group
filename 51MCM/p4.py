import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

## 可视化配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TrafficAnalysisSystem:
    def __init__(self):
        self._setup_parameters()
        self._load_dataset()

    def _setup_parameters(self):
        """配置系统参数"""
        self.RED_DURATION = 4
        self.GREEN_DURATION = 5
        self.CYCLE_LENGTH = self.RED_DURATION + self.GREEN_DURATION
        # FIRST_GREEN将作为可优化参数，在optimize_model中设置
        # 这里只是初始化变量，但不赋初始值

    def _load_dataset(self):
        """加载并预处理交通流量数据（方法1：数据平滑）"""
        df = pd.read_excel('.\\2025-51MCM-Problem A\附件(Attachment).xlsx',
                           sheet_name='表4 (Table 4)')
        time_points = df['时间 t (Time t)'].values
        flow_data = df['主路4的车流量 (Traffic flow on the Main road 4)'].values

        # 方法1：添加数据平滑处理（3点移动平均）
        flow_data = pd.Series(flow_data).rolling(window=3, center=True, min_periods=1).mean().values

        self.dataset = pd.DataFrame({
            '时间点': time_points,
            '时间索引': range(60),
            '主路流量': flow_data
        })
        self.time_idx = self.dataset['时间索引'].values
        self.actual_flow = self.dataset['主路流量'].values

    def _check_signal_state(self, t_val):
        """检查交通信号状态"""
        elapsed = t_val - self.FIRST_GREEN
        if elapsed < 0:
            adjusted = elapsed % self.CYCLE_LENGTH
            return adjusted >= -self.GREEN_DURATION
        return (elapsed % self.CYCLE_LENGTH) < self.GREEN_DURATION

    def _get_signal_states(self, t_array):
        """获取信号状态序列"""
        return np.array([self._check_signal_state(t) for t in t_array])

    def _calculate_branch_flows(self, t_array, params):
        """计算各支路流量"""
        # 参数解包
        # 最后一个参数为FIRST_GREEN
        self.FIRST_GREEN = params[-1]

        # 其他参数解包
        a_params = params[:10]
        b_params = params[10:15]
        c_params = params[15:-1]

        # 支路1流量计算
        flow1 = self._compute_flow1(t_array, a_params)

        # 支路2流量计算
        flow2 = self._compute_flow2(t_array, b_params)

        # 支路3流量计算
        flow3 = self._compute_flow3(t_array, c_params)

        return flow1, flow2, flow3

    def _compute_flow1(self, t, params):
        """计算支路1流量"""
        a1, a2, a3, a4, a5, a6, brk1, brk2, brk3, brk4 = params
        result = np.zeros_like(t, dtype=float)

        mask = t < brk1
        result[mask] = 0

        mask = (t >= brk1) & (t < brk2)
        result[mask] = a1 * (t[mask] - brk1) + a2

        mask = (t >= brk2) & (t < brk3)
        result[mask] = a3 * (t[mask] - brk2) + a4

        mask = (t >= brk3) & (t < brk4)
        result[mask] = a5

        mask = (t >= brk4)
        result[mask] = a6 * (t[mask] - brk4)

        return np.maximum(result, 0)

    def _compute_flow2(self, t, params):
        """计算支路2流量"""
        b1, b2, b3, b4, b5 = params
        result = np.zeros_like(t, dtype=float)
        brk5 = 17
        brk6 = 35

        mask = t <= brk5
        result[mask] = b1 * t[mask] + b2

        mask = (t > brk5) & (t <= brk6)
        result[mask] = b3

        mask = t > brk6
        result[mask] = b4 * (t[mask] - brk6) + b5

        return np.maximum(result, 0)

    def _compute_flow3(self, t, params):
        """计算支路3流量"""
        signal_states = self._get_signal_states(t)
        result = np.zeros_like(t, dtype=float)

        cycle_params = [(params[2 * i], params[2 * i + 1]) for i in range(7)]
        cycle_starts = [self.FIRST_GREEN + i * self.CYCLE_LENGTH for i in range(7)]

        for idx, start in enumerate(cycle_starts):
            mask = (t >= start) & (t < start + self.GREEN_DURATION) & signal_states
            slope, intercept = cycle_params[idx]
            result[mask] = slope * (t[mask] - start) + intercept

        return np.maximum(result, 0)

    def _calculate_main_flow(self, t_array, params):
        """计算主路流量"""
        f1, f2, f3 = self._calculate_branch_flows(t_array, params)

        # 移除支路1和2的延迟处理，直接使用原始流量
        # 根据题目要求，不考虑行驶延迟
        return f1 + f2 + f3

    def _huber_loss(self, errors, delta=1.0):
        """Huber损失函数（方法2：鲁棒性优化）"""
        abs_errors = np.abs(errors)
        quadratic = np.minimum(abs_errors, delta)
        linear = abs_errors - quadratic
        return 0.5 * quadratic ** 2 + delta * linear

    def _evaluate_model(self, params):
        """评估模型性能（使用加权Huber损失）"""
        predicted = self._calculate_main_flow(self.time_idx, params)
        errors = predicted - self.actual_flow

        # 使用加权Huber损失，对不同时间段赋予不同权重
        # 创建时间段权重，高峰期权重更高
        time_weights = np.ones_like(self.time_idx, dtype=float)
        # 早高峰(7:20-8:00)权重增加
        time_weights[(self.time_idx >= 10) & (self.time_idx <= 30)] = 1.5
        # 晚高峰(8:00-8:40)权重增加
        time_weights[(self.time_idx >= 30) & (self.time_idx <= 50)] = 1.5

        # 计算加权Huber损失
        weighted_errors = [time_weights[i] * self._huber_loss(errors[i], delta=1.5) for i in range(len(errors))]
        error = np.mean(weighted_errors)

        # 约束条件处理
        penalty = self._compute_penalty(params)

        # 添加全局平滑性约束
        predicted_diff = np.abs(np.diff(predicted))
        smoothness_penalty = 50 * np.sum(np.where(predicted_diff > 10, predicted_diff - 10, 0))

        return error + penalty + smoothness_penalty

    def _compute_penalty(self, params):
        """计算约束惩罚项"""
        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, params)
        penalty = 0

        # 非负约束 - 增加惩罚权重
        penalty += 2000 * (np.sum(np.abs(f1[f1 < 0])) +
                           np.sum(np.abs(f2[f2 < 0])) +
                           np.sum(np.abs(f3[f3 < 0])))

        # 支路1流量特征约束 - 确保符合"无车流量→线性增长→稳定→线性减少至无车流量"的趋势
        a1, a2, a3, a4, a5, a6, brk1, brk2, brk3, brk4 = params[:10]
        # 确保a1为正（线性增长）
        penalty += 1000 * max(0, -a1)
        # 确保a3为负（线性减少）
        penalty += 1000 * max(0, a3)
        # 确保a6为负（线性减少至无车流量）
        penalty += 1000 * max(0, a6)

        # 支路2流量特征约束 - 确保在特定时间段内线性增长、稳定和线性减少
        b1, b2, b3, b4, b5 = params[10:15]
        # 确保b1为正（线性增长）
        penalty += 1000 * max(0, -b1)
        # 确保b4为负（线性减少）
        penalty += 1000 * max(0, b4)

        # 支路2二阶差分惩罚（平滑性）- 增加权重
        f2_diff2 = np.abs(np.diff(f2, n=2))
        penalty += 200 * np.sum(f2_diff2)

        # 支路3流量特征约束 - 确保在绿灯时段有合理的流量变化
        c_params = params[15:-1]
        for i in range(7):
            slope, intercept = c_params[2*i], c_params[2*i+1]
            # 确保截距非负
            penalty += 500 * max(0, -intercept)
            # 限制斜率变化范围
            penalty += 100 * max(0, abs(slope) - 2.5)

        # 连续性约束 - 增加权重
        continuity_errors = [
            abs(a1 * (brk2 - brk1) + a2 - a4),  # 支路1第一段与第二段的连续性
            abs(a3 * (brk3 - brk2) + a4 - a5),  # 支路1第二段与第三段的连续性
            abs(a5 - a6 * 0),                   # 支路1第三段与第四段的连续性
            abs(b1 * 17 + b2 - b3),             # 支路2第一段与第二段的连续性
            abs(b3 - b4 * 0 - b5)               # 支路2第二段与第三段的连续性
        ]
        penalty += 1500 * sum(continuity_errors)

        # 转折点顺序约束
        order_errors = [
            max(0, brk1 - brk2),  # 确保 brk1 < brk2
            max(0, brk2 - brk3),  # 确保 brk2 < brk3
            max(0, brk3 - brk4)   # 确保 brk3 < brk4
        ]
        penalty += 2000 * sum(order_errors)

        return penalty

    def optimize_model(self):
        """优化模型参数"""
        # 初始参数设置，添加FIRST_GREEN作为可优化参数
        initial_guess = [
            2.0, 30.0, -1.0, 20.0, 30.0, -1.0, 7.0, 20.0, 30.0, 42.0,
            0.5, 0.0, 40.0, -1.0, 20.0,
            1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0,
            3.0  # FIRST_GREEN的初始值
        ]

        # 扩大参数边界范围，提高搜索精度
        param_bounds = [
            # 支路1参数边界 - 扩大搜索范围
            (0.3, 3.0), (20.0, 45.0), (-6.0, -0.05), (20.0, 40.0), (20.0, 40.0), (-6.0, -0.05),
            (3.0, 12.0), (15.0, 25.0), (20.0, 40.0), (35.0, 50.0),
            # 支路2参数边界 - 扩大搜索范围
            (0.05, 3.0), (0.0, 10.0), (30.0, 50.0), (-3.0, -0.05), (10.0, 30.0),
            # 支路3参数边界 - 调整以适应绿灯周期
            (0.0, 3.0), (0.0, 40.0), (0.0, 3.0), (0.0, 40.0), (0.0, 3.0), (0.0, 40.0),
            (0.0, 3.0), (0.0, 40.0), (0.0, 3.0), (0.0, 40.0), (0.0, 3.0), (0.0, 40.0),
            (0.0, 3.0), (0.0, 40.0),
            # FIRST_GREEN的边界 - 扩大搜索范围
            (0.0, 8.0)
        ]

        # 尝试多次优化，选择最佳结果
        best_error = float('inf')
        best_params = None
        best_result = None

        # 使用不同的初始点和优化方法进行多次优化
        for attempt in range(5):  # 增加尝试次数
            # 在初始猜测附近随机扰动
            perturbed_guess = np.array(initial_guess) * (1 + 0.15 * np.random.randn(len(initial_guess)))

            # 确保扰动后的参数仍在边界内
            for i, (lb, ub) in enumerate(param_bounds):
                perturbed_guess[i] = max(lb, min(ub, perturbed_guess[i]))

            # 选择不同的优化方法
            if attempt % 2 == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 1000, 'gtol': 1e-6}
            else:
                method = 'SLSQP'
                options = {'maxiter': 1000, 'ftol': 1e-6}

            # 执行优化
            optimization_result = minimize(
                self._evaluate_model,
                perturbed_guess,
                method=method,
                bounds=param_bounds,
                options=options
            )

            # 更新最佳结果
            if optimization_result.fun < best_error:
                best_error = optimization_result.fun
                best_params = optimization_result.x
                best_result = optimization_result

        # 使用最佳参数进行一次最终优化
        final_result = minimize(
            self._evaluate_model,
            best_params,
            method='L-BFGS-B',
            bounds=param_bounds,
            options={'maxiter': 2000, 'gtol': 1e-8}  # 增加迭代次数和精度
        )

        if final_result.fun < best_error:
            self.optimal_params = final_result.x
        else:
            self.optimal_params = best_params

        # 计算最终的预测流量和误差
        self.predicted_flow = self._calculate_main_flow(self.time_idx, self.optimal_params)
        self.model_error = np.sqrt(np.mean((self.predicted_flow - self.actual_flow) ** 2))

        # 计算各支路流量
        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, self.optimal_params)
        self.branch_flows = (f1, f2, f3)

    def visualize_results(self):
        """可视化分析结果"""
        self._plot_main_comparison()
        self._plot_branch_flows()

    def _plot_main_comparison(self):
        """绘制主路流量对比图"""
        plt.figure(figsize=(14, 6))
        plt.plot(self.time_idx, self.actual_flow, 'b-', lw=2, label='实测流量(平滑后)')
        plt.plot(self.time_idx, self.predicted_flow, 'r--', lw=2, label='预测流量')
        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('主路流量实测与预测对比')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./P4/主路流量实测与预测对比4.png', dpi=300, bbox_inches='tight')

    def _plot_branch_flows(self):
        """绘制支路流量图"""
        f1, f2, f3 = self.branch_flows
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:15]

        plt.figure(figsize=(14, 8))
        plt.plot(self.time_idx, f1, 'g-', label='支路1')
        plt.plot(self.time_idx, f2, 'm-', label='支路2')
        plt.plot(self.time_idx, f3, 'c-', label='支路3')

        # 标记绿灯时段
        for i in range(7):
            start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
            plt.axvspan(start, start + self.GREEN_DURATION, color='green', alpha=0.1)

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('各支路流量变化情况')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./P4/支路车流量变化4.png', dpi=300, bbox_inches='tight')

    def generate_report(self):
        """生成分析报告"""
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:15]
        c_params = self.optimal_params[15:-1]  # 排除最后一个参数(FIRST_GREEN)

        # 计算特定时间点流量
        t1, t2 = 15, 45  # 7:30和8:30对应的时间索引
        f1_730, f2_730, f3_730 = self._calculate_branch_flows(np.array([t1]), self.optimal_params)
        f1_830, f2_830, f3_830 = self._calculate_branch_flows(np.array([t2]), self.optimal_params)

        # 计算各时间段的平均误差
        time_segments = [(0, 15), (15, 30), (30, 45), (45, 59)]
        segment_errors = []
        for start, end in time_segments:
            segment_idx = (self.time_idx >= start) & (self.time_idx <= end)
            segment_error = np.sqrt(np.mean((self.predicted_flow[segment_idx] - self.actual_flow[segment_idx]) ** 2))
            segment_errors.append(segment_error)

        # 计算主路流量
        main_flow_730 = self._calculate_main_flow(np.array([t1]), self.optimal_params)[0]
        main_flow_830 = self._calculate_main_flow(np.array([t2]), self.optimal_params)[0]

        with open('./P4/交通流量分析报告4.md', 'w', encoding='utf-8') as f:
            f.write("# === 交通流量分析报告 ===\n\n")

            # 模型总体误差评估
            f.write("## 【模型误差评估】\n\n")
            f.write(f"总体RMSE误差: {self.model_error:.4f}\n\n")
            f.write("各时间段误差:\n\n")
            f.write("| 时间段 | RMSE误差 |\n")
            f.write("|--------|----------|\n")
            f.write(f"| 7:00-7:30 | {segment_errors[0]:.4f} |\n")
            f.write(f"| 7:30-8:00 | {segment_errors[1]:.4f} |\n")
            f.write(f"| 8:00-8:30 | {segment_errors[2]:.4f} |\n")
            f.write(f"| 8:30-8:58 | {segment_errors[3]:.4f} |\n\n")

            # 优化后的信号灯参数
            f.write("## 【优化后的系统参数】\n\n")
            f.write(f"第一个绿灯开始时间: {self.optimal_params[-1]:.4f}\n\n")

            f.write("## 【支路1模型参数】\n\n")
            f.write(f"增长阶段斜率: {a_params[0]:.4f}\n")
            f.write(f"初始值: {a_params[1]:.4f}\n")
            f.write(f"下降阶段斜率1: {a_params[2]:.4f}\n")
            f.write(f"稳定值: {a_params[4]:.4f}\n")
            f.write(f"下降阶段斜率2: {a_params[5]:.4f}\n")
            f.write(f"转折点: {a_params[6]:.1f}, {a_params[7]:.1f}, {a_params[8]:.1f}, {a_params[9]:.1f}\n\n")

            f.write("## 【支路2模型参数】\n\n")
            f.write(f"增长斜率: {b_params[0]:.4f}\n")
            f.write(f"截距: {b_params[1]:.4f}\n")
            f.write(f"稳定值: {b_params[2]:.4f}\n")
            f.write(f"下降斜率: {b_params[3]:.4f}\n")
            f.write(f"终值: {b_params[4]:.4f}\n")
            f.write(f"转折点: 35.0, 47.0\n\n")

            f.write("## 【支路3模型参数】\n\n")
            f.write("支路3在红灯时段流量为0，在绿灯时段呈现线性变化\n")
            for i in range(7):
                slope, intercept = c_params[2*i], c_params[2*i+1]
                f.write(f"绿灯周期{i+1}的斜率: {slope:.4f}, 截距: {intercept:.4f}\n")
            f.write("\n")

            # 支路1流量模型表达式
            f.write("## 【支路1流量模型表达式】\n\n")
            f.write(r"$f_1(t) = \begin{cases} ")
            f.write(f"0, & t < {a_params[6]:.1f} \\ ")
            f.write(f"{a_params[0]:.4f} \\cdot (t-{a_params[6]:.1f}) + {a_params[1]:.4f}, & {a_params[6]:.1f} \\leq t < {a_params[7]:.1f} \\\\ ")
            f.write(f"{a_params[2]:.4f} \\cdot (t-{a_params[7]:.1f}) + {a_params[3]:.4f}, & {a_params[7]:.1f} \\leq t < {a_params[8]:.1f} \\\\ ")
            f.write(f"{a_params[4]:.4f}, & {a_params[8]:.1f} \\leq t < {a_params[9]:.1f} \\\\ ")
            f.write(f"{a_params[5]:.4f} \\cdot (t-{a_params[9]:.1f}), & t \\geq {a_params[9]:.1f} ")
            f.write(r"\end{cases}$")
            f.write('\n\n')

            # 支路2流量模型表达式
            f.write("## 【支路2流量模型表达式】\n\n")
            f.write(r"$f_2(t) = \begin{cases} ")
            f.write(f"{b_params[0]:.4f} \\cdot t + {b_params[1]:.4f}, & t \\leq 35.0 \\\\ ")
            f.write(f"{b_params[2]:.4f}, & 35.0 < t \\leq 47.0 \\\\ ")
            f.write(f"{b_params[3]:.4f} \\cdot (t-47.0) + {b_params[4]:.4f}, & t > 47.0 ")
            f.write(r"\end{cases}$")
            f.write('\n\n')

            # 支路3流量模型表达式
            f.write("## 【支路3流量模型表达式】\n\n")
            f.write(r"$f_3(t) = \begin{cases} ")
            for i in range(7):
                start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
                end = start + self.GREEN_DURATION
                slope, intercept = c_params[2*i], c_params[2*i+1]
                f.write(f"{slope:.4f} \\cdot (t-{start}) + {intercept:.4f}, & t \\in [{start}, {end}) \\text{{ 且为绿灯时段}} \\\\ ")
            f.write(r"0, & \text{其他时段（红灯）} \end{cases}$")
            f.write('\n\n')

            # 关键时间点流量
            f.write("## 【关键时间点流量】\n\n")
            f.write("| 时间点 | 支路1 | 支路2 | 支路3 | 主路4(预测) | 主路4(实际) |\n")
            f.write("|--------|-------|-------|-------|------------|------------|\n")
            f.write(f"| 7:30 | {f1_730[0]:5.2f} | {f2_730[0]:5.2f} | {f3_730[0]:5.2f} | {main_flow_730:8.2f} | {self.actual_flow[t1]:8.2f} |\n")
            f.write(f"| 8:30 | {f1_830[0]:5.2f} | {f2_830[0]:5.2f} | {f3_830[0]:5.2f} | {main_flow_830:8.2f} | {self.actual_flow[t2]:8.2f} |\n")


if __name__ == "__main__":
    analyzer = TrafficAnalysisSystem()
    analyzer.optimize_model()
    analyzer.generate_report()
    analyzer.visualize_results()
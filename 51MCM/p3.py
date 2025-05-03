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
        # 根据题目要求，红灯时间为8分钟，绿灯时间为10分钟
        self.RED_DURATION = 4  # 对应8分钟（每个时间单位为2分钟）
        self.GREEN_DURATION = 5  # 对应10分钟（每个时间单位为2分钟）
        self.CYCLE_LENGTH = self.RED_DURATION + self.GREEN_DURATION
        self.FIRST_GREEN = 3  # 第一个绿灯于7:06开始亮起，对应时间索引3
        self.TRAVEL_DELAY = 1  # 支路1和支路2的行驶时间为2分钟，对应时间索引延迟1（因为每个时间单位为2分钟）

        # 根据题目描述，支路2的特定转折点是已知的
        self.BRANCH2_STABILIZE = 35  # 8:10对应的时间索引
        self.BRANCH2_DECREASE = 47   # 8:34对应的时间索引

        # 支路1的最后减少阶段起始点
        self.BRANCH1_FINAL_DECREASE = 53  # 根据数据分析确定的时间点

    def _load_dataset(self):
        """加载交通流量数据"""
        df = pd.read_excel('.\\2025-51MCM-Problem A\附件(Attachment).xlsx',sheet_name='表3 (Table 3)')
        time_points = df['时间 t (Time t)'].values
        flow_data = df['主路4的车流量 (Traffic flow on the Main road 4)'].values

        self.dataset = pd.DataFrame({
            '时间点': time_points,
            '时间索引': range(60),
            '主路流量': flow_data
        })
        self.time_idx = self.dataset['时间索引'].values
        self.actual_flow = self.dataset['主路流量'].values

    def _check_signal_state(self, t_val):
        """检查交通信号状态"""
        # 计算相对于第一个绿灯开始时间的经过时间
        elapsed = t_val - self.FIRST_GREEN

        # 处理第一个绿灯开始前的情况
        if elapsed < 0:
            # 使用周期取模，检查是否在绿灯时段
            adjusted = elapsed % self.CYCLE_LENGTH
            return adjusted >= -self.GREEN_DURATION

        # 对于第一个绿灯开始后的情况，检查是否在绿灯时段
        # 如果 (elapsed % CYCLE_LENGTH) < GREEN_DURATION，则为绿灯
        return (elapsed % self.CYCLE_LENGTH) < self.GREEN_DURATION

    def _get_signal_states(self, t_array):
        """获取信号状态序列"""
        return np.array([self._check_signal_state(t) for t in t_array])

    def _calculate_branch_flows(self, t_array, params):
        """计算各支路流量"""
        # 参数解包
        a_params = params[:7]  # 支路1参数
        b_params = params[7:10]  # 支路2参数
        c_params = params[10:]  # 支路3参数

        # 支路1流量计算
        flow1 = self._compute_flow1(t_array, a_params)

        # 支路2流量计算
        flow2 = self._compute_flow2(t_array, b_params)

        # 支路3流量计算
        flow3 = self._compute_flow3(t_array, c_params)

        return flow1, flow2, flow3

    def _compute_flow1(self, t, params):
        """计算支路1流量 - "无车流量→增长→减少→稳定→减少至无车流量"模式"""
        a1, a2, a3, a4, brk1, brk2, brk3 = params
        result = np.zeros_like(t, dtype=float)

        # 无车流量阶段
        mask = t < brk1
        result[mask] = 0

        # 增长阶段
        mask = (t >= brk1) & (t < brk2)
        result[mask] = a1 * (t[mask] - brk1)  # 从0开始增长

        # 减少阶段
        mask = (t >= brk2) & (t < brk3)
        # 确保连续性
        peak_value = a1 * (brk2 - brk1)
        result[mask] = a2 * (t[mask] - brk2) + peak_value

        # 稳定阶段
        mask = (t >= brk3) & (t < self.BRANCH1_FINAL_DECREASE)
        result[mask] = a3

        # 减少至无车流量阶段
        mask = t >= self.BRANCH1_FINAL_DECREASE
        result[mask] = a4 * (t[mask] - self.BRANCH1_FINAL_DECREASE) + a3

        return np.maximum(result, 0)

    def _compute_flow2(self, t, params):
        """计算支路2流量 - 线性增长、稳定、线性减少模式"""
        b1, b2, b3 = params
        result = np.zeros_like(t, dtype=float)

        # 线性增长阶段 [6:58-8:10]
        mask = t <= self.BRANCH2_STABILIZE
        result[mask] = b1 * t[mask] + b2

        # 稳定阶段 (8:10-8:34)
        mask = (t > self.BRANCH2_STABILIZE) & (t <= self.BRANCH2_DECREASE)
        stable_value = b1 * self.BRANCH2_STABILIZE + b2
        result[mask] = stable_value

        # 线性减少阶段 [8:34-8:58]
        mask = t > self.BRANCH2_DECREASE
        result[mask] = b3 * (t[mask] - self.BRANCH2_DECREASE) + stable_value

        return np.maximum(result, 0)

    def _compute_flow3(self, t, params):
        """计算支路3流量 - 仅在绿灯时有流量"""
        # 支路3只在绿灯时有流量，红灯时为0
        signal_states = self._get_signal_states(t)
        result = np.zeros_like(t, dtype=float)

        # 将参数分组为每个绿灯周期的斜率和截距
        cycle_params = [(params[2 * i], params[2 * i + 1]) for i in range(7)]

        # 计算每个绿灯周期的开始时间
        cycle_starts = [self.FIRST_GREEN + i * self.CYCLE_LENGTH for i in range(7)]

        # 对每个绿灯周期分别计算流量
        for idx, start in enumerate(cycle_starts):
            # 只在绿灯时段内有流量
            mask = (t >= start) & (t < start + self.GREEN_DURATION) & signal_states
            if np.any(mask):
                slope, intercept = cycle_params[idx]
                # 线性模型：斜率*(时间-周期开始时间)+截距
                result[mask] = slope * (t[mask] - start) + intercept

        return np.maximum(result, 0)

    def _calculate_main_flow(self, t_array, params):
        """计算主路流量"""
        f1, f2, f3 = self._calculate_branch_flows(t_array, params)

        f1_delayed = np.zeros_like(t_array)
        f2_delayed = np.zeros_like(t_array)

        valid_idx = t_array >= self.TRAVEL_DELAY
        f1_delayed[valid_idx] = f1[np.where(valid_idx)[0] - self.TRAVEL_DELAY]
        f2_delayed[valid_idx] = f2[np.where(valid_idx)[0] - self.TRAVEL_DELAY]

        f1_delayed[~valid_idx] = f1[0]
        f2_delayed[~valid_idx] = f2[0]

        return f1_delayed + f2_delayed + f3

    def _evaluate_model(self, params):
        """评估模型性能"""
        predicted = self._calculate_main_flow(self.time_idx, params)
        error = np.mean((predicted - self.actual_flow) ** 2)

        # 约束条件处理
        penalty = self._compute_penalty(params)

        return error + penalty

    def _compute_penalty(self, params):
        """计算约束惩罚项"""
        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, params)
        penalty = 0

        # 非负约束 - 增加惩罚权重
        penalty += 2000 * (np.sum(np.abs(f1[f1 < 0])) +
                           np.sum(np.abs(f2[f2 < 0])) +
                           np.sum(np.abs(f3[f3 < 0])))

        # 连续性约束 - 支路1
        a1, a2, a3, a4, brk1, brk2, brk3 = params[:7]

        # 支路1的特定模式要求
        peak_value = a1 * (brk2 - brk1)
        stable_value = a2 * (brk3 - brk2) + peak_value

        continuity_errors = [
            abs(stable_value - a3),  # 确保稳定值正确连接
        ]

        # 增加对转折点顺序的约束
        order_errors = [
            max(0, brk1 - brk2),  # 确保 brk1 < brk2
            max(0, brk2 - brk3),  # 确保 brk2 < brk3
            max(0, brk3 - self.BRANCH1_FINAL_DECREASE)  # 确保 brk3 < 最后减少开始点
        ]

        penalty += 1500 * sum(continuity_errors) + 2000 * sum(order_errors)

        return penalty

    def optimize_model(self):
        """优化模型参数"""
        # 根据新的参数结构调整初始猜测
        initial_guess = [
            2.0, -1.5, 30.0, -2.0, 5.0, 15.0, 25.0,  # 支路1参数 (a1,a2,a3,a4,brk1,brk2,brk3)
            0.8, 8.0, -1.2,                          # 支路2参数 (b1,b2,b3)
            1.0, 20.0, 2.0, 20.0, 2.0, 20.0, 2.0, 25.0, 0.8, 30.0, 1.0, 20.0, 1.0, 20.0 # 支路3参数
        ]

        # 根据新的参数结构调整参数边界
        param_bounds = [
            # 支路1参数边界
            (0.1, 10.0),    # a1: 增长斜率
            (-8.0, -0.1),   # a2: 减少斜率1
            (20.0, 45.0),   # a3: 稳定值
            (-5.0, -0.1),   # a4: 减少斜率2
            (1.0, 12.0),    # brk1: 第一个转折点
            (8.0, 22.0),    # brk2: 第二个转折点
            (18.0, 35.0),   # brk3: 第三个转折点

            # 支路2参数边界 - 转折点已固定
            (0.1, 3.5),     # b1: 增长斜率
            (2.0, 18.0),    # b2: 截距
            (-5.0, -0.1),   # b3: 减少斜率

            # 支路3参数边界
            (0.0, 5.0), (15.0, 50.0),   # 周期1斜率和截距
            (0.0, 5.0), (15.0, 50.0),   # 周期2斜率和截距
            (0.0, 5.0), (15.0, 50.0),   # 周期3斜率和截距
            (0.0, 5.0), (15.0, 50.0),   # 周期4斜率和截距
            (0.0, 5.0), (15.0, 50.0),   # 周期5斜率和截距
            (0.0, 5.0), (10.0, 40.0),   # 周期6斜率和截距
            (0.0, 5.0), (10.0, 40.0)    # 周期7斜率和截距
        ]

        # 创建多个不同的初始猜测点
        initial_guesses = [
            # 原始猜测
            initial_guess,

            # 支路1侧重的初始点
            [3.5, -2.0, 35.0, -1.5, 4.0, 18.0, 28.0] + initial_guess[7:],

            # 支路2侧重的初始点
            initial_guess[:7] + [1.2, 10.0, -2.0] + initial_guess[10:],

            # 支路3侧重的初始点
            initial_guess[:10] + [2.0, 25.0, 2.5, 25.0, 3.0, 25.0, 2.5, 30.0, 1.5, 35.0, 1.0, 25.0, 1.0, 25.0],

            # 基于当前最优解的初始点
            [3.6542, -1.3171, 28.9160, -2.0989, 4.9, 15.3, 22.2,
             0.9182, 8.1058, -1.3577,
             1.2931, 20.6360, 2.2426, 20.8042, 2.4467, 21.1739, 1.9253, 26.5885,
             0.6384, 28.7363, 0.6949, 18.6972, 0.6935, 17.8120]
        ]

        # 尝试多次优化，选择最佳结果
        best_error = float('inf')
        best_params = None

        # 使用多个初始点进行优化
        for base_guess in initial_guesses:
            # 对每个基础初始点尝试多次随机扰动
            for _ in range(3):
                # 在初始猜测附近随机扰动
                perturbed_guess = np.array(base_guess) * (1 + 0.15 * np.random.randn(len(base_guess)))

                # 确保扰动后的参数仍在边界内
                for i, (lb, ub) in enumerate(param_bounds):
                    perturbed_guess[i] = max(lb, min(ub, perturbed_guess[i]))

                # 使用不同的优化器尝试
                for method in ['L-BFGS-B', 'TNC']:
                    optimization_result = minimize(
                        self._evaluate_model,
                        perturbed_guess,
                        method=method,
                        bounds=param_bounds,
                        options={'maxiter': 1500, 'gtol': 1e-7}
                    )

                    if optimization_result.fun < best_error:
                        best_error = optimization_result.fun
                        best_params = optimization_result.x
                        print(f"发现更优解: RMSE={np.sqrt(best_error):.4f}")

        # 使用最佳参数
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
        plt.plot(self.time_idx, self.actual_flow, 'b-', lw=2, label='实测流量')
        plt.plot(self.time_idx, self.predicted_flow, 'r--', lw=2, label='预测流量')
        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('主路流量实测与预测对比')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./P3/主路流量实测与预测对比.png', dpi=300, bbox_inches='tight')

    def _plot_branch_flows(self):
        """绘制支路流量图"""
        f1, f2, f3 = self.branch_flows
        a_params = self.optimal_params[:7]  # 更新为7个参数

        plt.figure(figsize=(14, 8))
        plt.plot(self.time_idx, f1, 'g-', label='支路1')
        plt.plot(self.time_idx, f2, 'm-', label='支路2')
        plt.plot(self.time_idx, f3, 'c-', label='支路3')

        # 标记转折点
        for i, brk in enumerate(a_params[4:7]):
            plt.axvline(x=brk, color='g', ls='--', alpha=0.3)

        # 标记支路1最后减少阶段开始点
        plt.axvline(x=self.BRANCH1_FINAL_DECREASE, color='g', ls='--', alpha=0.3)

        # 标记支路2的固定转折点
        plt.axvline(x=self.BRANCH2_STABILIZE, color='m', ls='--', alpha=0.3)
        plt.axvline(x=self.BRANCH2_DECREASE, color='m', ls='--', alpha=0.3)

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
        plt.savefig('./P3/支路车流量变化.png', dpi=300, bbox_inches='tight')

    def generate_report(self):
        """生成分析报告"""
        a_params = self.optimal_params[:7]   # 更新为7个参数
        b_params = self.optimal_params[7:10] # 更新为3个参数
        c_params = self.optimal_params[10:]  # 支路3参数不变

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

        with open('./P3/交通流量分析报告.md', 'w', encoding='utf-8') as f:
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

            # 支路1模型参数
            f.write("## 【支路1模型参数】\n\n")
            f.write(f"增长阶段斜率: {a_params[0]:.4f}\n")
            f.write(f"减少阶段斜率1: {a_params[1]:.4f}\n")
            f.write(f"稳定值: {a_params[2]:.4f}\n")
            f.write(f"减少阶段斜率2: {a_params[3]:.4f}\n")
            f.write(f"转折点: {a_params[4]:.1f}, {a_params[5]:.1f}, {a_params[6]:.1f}\n")
            f.write(f"最后减少开始点: {self.BRANCH1_FINAL_DECREASE}\n\n")

            # 支路2模型参数
            f.write("## 【支路2模型参数】\n\n")
            f.write(f"增长斜率: {b_params[0]:.4f}\n")
            f.write(f"截距: {b_params[1]:.4f}\n")
            f.write(f"减少斜率: {b_params[2]:.4f}\n")
            f.write(f"转折点(固定): {self.BRANCH2_STABILIZE} (8:10), {self.BRANCH2_DECREASE} (8:34)\n\n")

            # 支路3模型参数
            f.write("## 【支路3模型参数】\n\n")
            for i in range(7):
                slope, intercept = c_params[2*i], c_params[2*i+1]
                f.write(f"绿灯周期{i+1}的斜率: {slope:.4f}, 截距: {intercept:.4f}\n")
            f.write("\n")

            # 支路1流量模型表达式
            f.write("## 【支路1流量模型表达式】\n\n")
            f.write(r"$f_1(t) = \begin{cases} ")
            f.write(f"0, & t < {a_params[4]:.1f} \\\\ ")
            f.write(f"{a_params[0]:.4f} \\cdot (t-{a_params[4]:.1f}), & {a_params[4]:.1f} \\leq t < {a_params[5]:.1f} \\\\ ")

            peak_value = a_params[0] * (a_params[5] - a_params[4])
            f.write(f"{a_params[1]:.4f} \\cdot (t-{a_params[5]:.1f}) + {peak_value:.4f}, & {a_params[5]:.1f} \\leq t < {a_params[6]:.1f} \\\\ ")

            stable_value = a_params[1] * (a_params[6] - a_params[5]) + peak_value
            f.write(f"{a_params[2]:.4f}, & {a_params[6]:.1f} \\leq t < {self.BRANCH1_FINAL_DECREASE} \\\\ ")
            f.write(f"{a_params[3]:.4f} \\cdot (t-{self.BRANCH1_FINAL_DECREASE}) + {a_params[2]:.4f}, & t \\geq {self.BRANCH1_FINAL_DECREASE} ")
            f.write(r"\end{cases}$")
            f.write('\n\n')

            # 支路2流量模型表达式
            f.write("## 【支路2流量模型表达式】\n\n")
            f.write(r"$f_2(t) = \begin{cases} ")
            f.write(f"{b_params[0]:.4f} \\cdot t + {b_params[1]:.4f}, & t \\leq {self.BRANCH2_STABILIZE} \\\\ ")

            stable_value = b_params[0] * self.BRANCH2_STABILIZE + b_params[1]
            f.write(f"{stable_value:.4f}, & {self.BRANCH2_STABILIZE} < t \\leq {self.BRANCH2_DECREASE} \\\\ ")
            f.write(f"{b_params[2]:.4f} \\cdot (t-{self.BRANCH2_DECREASE}) + {stable_value:.4f}, & t > {self.BRANCH2_DECREASE} ")
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
            main_flow_730 = self._calculate_main_flow(np.array([t1]), self.optimal_params)[0]
            main_flow_830 = self._calculate_main_flow(np.array([t2]), self.optimal_params)[0]
            f.write(f"| 7:30 | {f1_730[0]:5.2f} | {f2_730[0]:5.2f} | {f3_730[0]:5.2f} | {main_flow_730:8.2f} | {self.actual_flow[t1]:8.2f} |\n")
            f.write(f"| 8:30 | {f1_830[0]:5.2f} | {f2_830[0]:5.2f} | {f3_830[0]:5.2f} | {main_flow_830:8.2f} | {self.actual_flow[t2]:8.2f} |\n")

if __name__ == "__main__":
    analyzer = TrafficAnalysisSystem()
    analyzer.optimize_model()
    analyzer.generate_report()
    analyzer.visualize_results()
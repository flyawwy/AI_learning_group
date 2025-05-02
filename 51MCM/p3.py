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
        a_params = params[:10]
        b_params = params[10:17]
        c_params = params[17:]

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
        b1, b2, b3, b4, b5, brk5, brk6 = params
        result = np.zeros_like(t, dtype=float)

        mask = t <= brk5
        result[mask] = b1 * t[mask] + b2

        mask = (t > brk5) & (t <= brk6)
        result[mask] = b3

        mask = t > brk6
        result[mask] = b4 * (t[mask] - brk6) + b5

        return np.maximum(result, 0)

    def _compute_flow3(self, t, params):
        """计算支路3流量"""
        # 支路3只在绿灯时有流量，红灯时为0
        signal_states = self._get_signal_states(t)
        result = np.zeros_like(t, dtype=float)

        # 将参数分组为每个绿灯周期的斜率和截距
        cycle_params = [(params[2 * i], params[2 * i + 1]) for i in range(5)]

        # 计算每个绿灯周期的开始时间
        cycle_starts = [self.FIRST_GREEN + i * self.CYCLE_LENGTH for i in range(5)]

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

        # 连续性约束
        a1, a2, a3, a4, a5, a6, brk1, brk2, brk3, brk4 = params[:10]
        b1, b2, b3, b4, b5, brk5, brk6 = params[10:17]

        # 支路1的连续性约束
        continuity_errors = [
            abs(a1 * (brk2 - brk1) + a2 - a4),  # 第一段末尾与第二段开始的连续性
            abs(a3 * (brk3 - brk2) + a4 - a5),  # 第二段末尾与第三段的连续性
            abs(a5 - a6 * 0),                   # 第三段与第四段的连续性
            abs(b1 * brk5 + b2 - b3),           # 支路2第一段与第二段的连续性
            abs(b3 - b5)                         # 支路2第二段与第三段的连续性
        ]

        # 增加对转折点顺序的约束
        order_errors = [
            max(0, brk1 - brk2),  # 确保 brk1 < brk2
            max(0, brk2 - brk3),  # 确保 brk2 < brk3
            max(0, brk3 - brk4),  # 确保 brk3 < brk4
            max(0, brk5 - brk6)   # 确保 brk5 < brk6
        ]

        penalty += 1500 * sum(continuity_errors) + 2000 * sum(order_errors)

        return penalty

    def optimize_model(self):
        """优化模型参数"""
        # 根据问题描述和数据特征调整初始参数
        initial_guess = [
            2.0, 0.0, -1.5, 30.0, 20.0, -0.5, 5.0, 12.0, 16.0, 47.0,  # 支路1参数
            0.8, 8.0, 25.0, -1.2, 30.0, 25.0, 44.0,                  # 支路2参数
            10.0, 5.0, 8.0, 15.0, 12.0, 20.0, 15.0, 25.0, 10.0, 30.0  # 支路3参数
        ]

        # 根据问题描述调整参数边界
        param_bounds = [
            # 支路1参数边界
            (0.5, 8.0), (0.0, 10.0), (-5.0, -0.5), (20.0, 40.0), (15.0, 35.0), (-3.0, -0.1),
            (3.0, 8.0), (7.0, 15.0), (12.0, 20.0), (35.0, 55.0),
            # 支路2参数边界
            (0.3, 2.5), (5.0, 15.0), (20.0, 35.0), (-3.0, -0.5), (25.0, 40.0),
            (20.0, 35.0), (35.0, 50.0),
            # 支路3参数边界 - 调整以适应绿灯周期
            (5.0, 25.0), (0.0, 40.0), (5.0, 25.0), (10.0, 40.0), (5.0, 30.0),
            (10.0, 35.0), (5.0, 30.0), (10.0, 40.0), (0.0, 25.0), (10.0, 40.0)
        ]

        # 尝试多次优化，选择最佳结果
        best_error = float('inf')
        best_params = None

        # 使用不同的初始点进行多次优化
        for _ in range(3):
            # 在初始猜测附近随机扰动
            perturbed_guess = np.array(initial_guess) * (1 + 0.1 * np.random.randn(len(initial_guess)))

            # 确保扰动后的参数仍在边界内
            for i, (lb, ub) in enumerate(param_bounds):
                perturbed_guess[i] = max(lb, min(ub, perturbed_guess[i]))

            optimization_result = minimize(
                self._evaluate_model,
                perturbed_guess,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': 1000, 'gtol': 1e-6}
            )

            if optimization_result.fun < best_error:
                best_error = optimization_result.fun
                best_params = optimization_result.x

        # 使用最佳参数
        self.optimal_params = best_params

        # 计算最终的预测流量和误差
        self.predicted_flow = self._calculate_main_flow(self.time_idx, self.optimal_params)
        self.model_error = np.sqrt(np.mean((self.predicted_flow - self.actual_flow) ** 2))

        # 计算各支路流量
        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, self.optimal_params)
        self.branch_flows = (f1, f2, f3)

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
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:17]

        plt.figure(figsize=(14, 8))
        plt.plot(self.time_idx, f1, 'g-', label='支路1')
        plt.plot(self.time_idx, f2, 'm-', label='支路2')
        plt.plot(self.time_idx, f3, 'c-', label='支路3')

        # 标记转折点
        for i, brk in enumerate(a_params[6:10]):
            plt.axvline(x=brk, color='g', ls='--', alpha=0.3)
        for brk in b_params[5:7]:
            plt.axvline(x=brk, color='m', ls='--', alpha=0.3)

        # 标记绿灯时段
        for i in range(6):
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
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:17]
        c_params = self.optimal_params[17:]

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
            f.write(f"初始值: {a_params[1]:.4f}\n")
            f.write(f"下降阶段斜率1: {a_params[2]:.4f}\n")
            f.write(f"稳定值: {a_params[4]:.4f}\n")
            f.write(f"下降阶段斜率2: {a_params[5]:.4f}\n")
            f.write(f"转折点: {a_params[6]:.1f}, {a_params[7]:.1f}, {a_params[8]:.1f}, {a_params[9]:.1f}\n\n")

            # 支路2模型参数
            f.write("## 【支路2模型参数】\n\n")
            f.write(f"增长斜率: {b_params[0]:.4f}\n")
            f.write(f"截距: {b_params[1]:.4f}\n")
            f.write(f"稳定值: {b_params[2]:.4f}\n")
            f.write(f"下降斜率: {b_params[3]:.4f}\n")
            f.write(f"终值: {b_params[4]:.4f}\n")
            f.write(f"转折点: {b_params[5]:.1f}, {b_params[6]:.1f}\n\n")

            # 支路3模型参数
            f.write("## 【支路3模型参数】\n\n")
            f.write("支路3在红灯时段流量为0，在绿灯时段呈现线性变化\n")
            for i in range(5):
                slope, intercept = c_params[2*i], c_params[2*i+1]
                f.write(f"绿灯周期{i+1}的斜率: {slope:.4f}, 截距: {intercept:.4f}\n")
            f.write("\n")

            # 支路1流量模型表达式
            f.write("## 【支路1流量模型表达式】\n\n")
            f.write(r"$f_1(t) = \begin{cases} ")
            f.write(f"0, & t < {a_params[6]:.1f} \\ ")
            f.write(r"{a_params[0]:.4f} \cdot (t-{a_params[6]:.1f}) + {a_params[1]:.4f}, & {a_params[6]:.1f} \leq t < {a_params[7]:.1f} \\\\ ")
            f.write(r"{a_params[2]:.4f} \cdot (t-{a_params[7]:.1f}) + {a_params[3]:.4f}, & {a_params[7]:.1f} \leq t < {a_params[8]:.1f} \\\\ ")
            f.write(r"{a_params[4]:.4f}, & {a_params[8]:.1f} \leq t < {a_params[9]:.1f} \\\\ ")
            f.write(r"{a_params[5]:.4f} \cdot (t-{a_params[9]:.1f}), & t \geq {a_params[9]:.1f} ")
            f.write(r"\end{cases}$")
            f.write('\n\n')

            # 支路2流量模型表达式
            f.write("## 【支路2流量模型表达式】\n\n")
            f.write(r"$f_2(t) = \begin{cases} ")
            f.write(f"{b_params[0]:.4f} \\cdot t + {b_params[1]:.4f}, & t \\leq {b_params[5]:.1f} \\\\ ")
            f.write(f"{b_params[2]:.4f}, & {b_params[5]:.1f} < t \\leq {b_params[6]:.1f} \\\\ ")
            f.write(f"{b_params[3]:.4f} \\cdot (t-{b_params[6]:.1f}) + {b_params[4]:.4f}, & t > {b_params[6]:.1f} ")
            f.write(r"\end{cases}$")
            f.write('\n\n')

            # 支路3流量模型表达式
            f.write("## 【支路3流量模型表达式】\n\n")
            f.write(r"$f_3(t) = \begin{cases} ")
            for i in range(5):
                start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
                end = start + self.GREEN_DURATION
                slope, intercept = c_params[2*i], c_params[2*i+1]
                f.write(r"{slope:.4f} \cdot (t-{start}) + {intercept:.4f}, & t \in [{start}, {end}) \text{{ 且为绿灯时段}} \\ ")
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
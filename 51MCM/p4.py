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
        self.FIRST_GREEN = 3
        self.TRAVEL_DELAY = 1

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
        a_params = params[:10]
        b_params = params[10:15]
        c_params = params[15:]

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
        brk5 =35
        brk6 = 47

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

        f1_delayed = np.zeros_like(t_array)
        f2_delayed = np.zeros_like(t_array)

        valid_idx = t_array >= self.TRAVEL_DELAY
        f1_delayed[valid_idx] = f1[np.where(valid_idx)[0] - self.TRAVEL_DELAY]
        f2_delayed[valid_idx] = f2[np.where(valid_idx)[0] - self.TRAVEL_DELAY]

        f1_delayed[~valid_idx] = f1[0]
        f2_delayed[~valid_idx] = f2[0]

        return f1_delayed + f2_delayed + f3

    def _huber_loss(self, errors, delta=1.0):
        """Huber损失函数（方法2：鲁棒性优化）"""
        abs_errors = np.abs(errors)
        quadratic = np.minimum(abs_errors, delta)
        linear = abs_errors - quadratic
        return 0.5 * quadratic ** 2 + delta * linear

    def _evaluate_model(self, params):
        """评估模型性能（使用Huber损失）"""
        predicted = self._calculate_main_flow(self.time_idx, params)
        # 方法2：使用Huber损失代替MSE
        error = np.mean(self._huber_loss(predicted - self.actual_flow, delta=1.0))

        # 约束条件处理
        penalty = self._compute_penalty(params)

        return error + penalty

    def _compute_penalty(self, params):
        """计算约束惩罚项"""
        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, params)
        penalty = 0

        # 非负约束
        penalty += 1000 * (np.sum(np.abs(f1[f1 < 0])) +
                           np.sum(np.abs(f2[f2 < 0])) +
                           np.sum(np.abs(f3[f3 < 0])))
        # 支路2二阶差分惩罚（平滑性）
        f2_diff2 = np.abs(np.diff(f2, n=2))
        penalty += 100 * np.sum(f2_diff2)

        # 连续性约束
        a1, a2, a3, a4, a5, a6, brk1, brk2, brk3, brk4 = params[:10]
        b1, b2, b3, b4, b5 = params[10:15]

        continuity_errors = [
            abs(a1 * (brk2 - brk1) + a2 - a4),
            abs(a3 * (brk3 - brk2) + a4 - a5),
            abs(a5 - a6 * 0),
            abs(b1 * 35 + b2 - b3),
            abs(b3 - b5)
        ]

        penalty += 1000 * sum(continuity_errors)
        return penalty

    def optimize_model(self):
        """优化模型参数"""
        initial_guess = [
            2.0, 30.0, -1.0, 20.0, 30.0, -1.0, 7.0, 20.0, 30.0, 42.0,
            0.5, 0.0, 40.0, -1.0, 20.0,
            1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0
        ]

        param_bounds = [
            (0.5, 2.0), (25.0, 40.0), (-5.0, -0.1), (25.0, 35.0), (25.0, 35.0), (-5.0, -0.1),
            (5.0, 10.0), (19.0, 21.0), (25.0, 35.0), (40.0, 45.0),
            (0.1, 2.0), (0.0, 5.0), (35.0, 45.0), (-2.0, -0.1), (15.0, 25.0),
            (0.0, 2.0), (0.0, 30.0), (0.0, 2.0), (0.0, 30.0),(0.0, 2.0), (0.0, 30.0), (0.0, 2.0), (0.0, 30.0),
            (0.0, 2.0), (0.0, 30.0), (0.0, 2.0), (0.0, 30.0),(0.0, 2.0), (0.0, 30.0)
        ]

        optimization_result = minimize(
            self._evaluate_model,
            initial_guess,
            method='L-BFGS-B',
            bounds=param_bounds
        )

        self.optimal_params = optimization_result.x
        self.predicted_flow = self._calculate_main_flow(self.time_idx, self.optimal_params)
        self.model_error = np.sqrt(np.mean((self.predicted_flow - self.actual_flow) ** 2))

        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, self.optimal_params)
        self.branch_flows = (f1, f2, f3)

    def visualize_results(self):
        """可视化分析结果"""
        self._plot_main_comparison()
        self._plot_branch_flows()
        plt.show()

    def _plot_main_comparison(self):
        """绘制主路流量对比图"""
        plt.figure(figsize=(14, 6))
        plt.plot(self.time_idx, self.actual_flow, 'b-', lw=2, label='实测流量(平滑后)')
        plt.plot(self.time_idx, self.predicted_flow, 'r--', lw=2, label='预测流量')
        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('主路流量实测与预测对比(使用Huber损失)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('主路流量实测与预测对比4.png', dpi=300, bbox_inches='tight')

    def _plot_branch_flows(self):
        """绘制支路流量图"""
        f1, f2, f3 = self.branch_flows
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:17]

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
        plt.title('各支路流量变化情况(使用数据平滑和Huber损失)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('支路车流量变化4.png', dpi=300, bbox_inches='tight')

    def generate_report(self):
        """生成分析报告"""
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:17]
        c_params = self.optimal_params[17:]

        # 计算特定时间点流量
        t1, t2 = 15, 45  # 7:30和8:30对应的时间索引
        f1_730, f2_730, f3_730 = self._calculate_branch_flows(np.array([t1]), self.optimal_params)
        f1_830, f2_830, f3_830 = self._calculate_branch_flows(np.array([t2]), self.optimal_params)

        with open('交通流量分析报告4.txt', 'w', encoding='utf-8') as f:
            f.write("=== 交通流量分析报告(使用数据平滑和Huber损失) ===\n\n")
            f.write(f"模型RMSE误差: {self.model_error:.4f}\n\n")
            f.write("注：此版本使用了数据平滑和Huber损失函数，对设备误差更具鲁棒性\n\n")

            f.write("【支路1模型参数】\n")
            f.write(f"增长阶段斜率: {a_params[0]:.4f}\n")
            f.write(f"初始值: {a_params[1]:.4f}\n")
            f.write(f"下降阶段斜率1: {a_params[2]:.4f}\n")
            f.write(f"稳定值: {a_params[4]:.4f}\n")
            f.write(f"下降阶段斜率2: {a_params[5]:.4f}\n")
            f.write(f"转折点: {a_params[6]:.1f}, {a_params[7]:.1f}, {a_params[8]:.1f}, {a_params[9]:.1f}\n\n")

            f.write("【支路2模型参数】\n")
            f.write(f"增长斜率: {b_params[0]:.4f}\n")
            f.write(f"截距: {b_params[1]:.4f}\n")
            f.write(f"稳定值: {b_params[2]:.4f}\n")
            f.write(f"下降斜率: {b_params[3]:.4f}\n")
            f.write(f"终值: {b_params[4]:.4f}\n")
            f.write(f"转折点: {b_params[5]:.1f}, {b_params[6]:.1f}\n\n")

            f.write("【关键时间点流量】\n")
            f.write("时间点 | 支路1 | 支路2 | 支路3\n")
            f.write(f"7:30  | {f1_730[0]:5.2f} | {f2_730[0]:5.2f} | {f3_730[0]:5.2f}\n")
            f.write(f"8:30  | {f1_830[0]:5.2f} | {f2_830[0]:5.2f} | {f3_830[0]:5.2f}\n")


if __name__ == "__main__":
    analyzer = TrafficAnalysisSystem()
    analyzer.optimize_model()
    analyzer.generate_report()
    analyzer.visualize_results()
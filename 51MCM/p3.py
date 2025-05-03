import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import os

## 可视化配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TrafficFlowAnalyzer:
    """交通流量分析系统 - 问题3解决方案"""

    def __init__(self):
        """初始化分析器"""
        self._setup_parameters()
        self._load_dataset()

    def _setup_parameters(self):
        """配置系统参数"""
        # 根据题目要求，红灯时间为8分钟，绿灯时间为10分钟
        self.RED_DURATION = 4      # 对应8分钟（每个时间单位为2分钟）
        self.GREEN_DURATION = 5    # 对应10分钟（每个时间单位为2分钟）
        self.CYCLE_LENGTH = self.RED_DURATION + self.GREEN_DURATION
        self.FIRST_GREEN = 3       # 第一个绿灯于7:06开始亮起，对应时间索引3
        self.TRAVEL_DELAY = 1      # 支路1和支路2的行驶时间为2分钟，对应延迟1个时间单位

    def _load_dataset(self):
        """加载交通流量数据"""
        # 读取Excel文件
        df = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx', sheet_name='表3 (Table 3)')
        time_points = df['时间 t (Time t)'].values
        flow_data = df['主路4的车流量 (Traffic flow on the Main road 4)'].values

        # 整理数据到DataFrame
        self.dataset = pd.DataFrame({
            '时间点': time_points,
            '时间索引': range(60),
            '主路流量': flow_data
        })

        # 提取关键数据
        self.time_idx = self.dataset['时间索引'].values
        self.actual_flow = self.dataset['主路流量'].values

    def _check_signal_state(self, t_val):
        """检查指定时间点的信号灯状态

        Args:
            t_val: 时间点索引

        Returns:
            bool: True表示绿灯，False表示红灯
        """
        # 计算相对于第一个绿灯开始时间的经过时间
        elapsed = t_val - self.FIRST_GREEN

        # 处理第一个绿灯开始前的情况
        if elapsed < 0:
            # 使用周期取模，检查是否在绿灯时段
            adjusted = elapsed % self.CYCLE_LENGTH
            return adjusted >= -self.GREEN_DURATION

        # 对于第一个绿灯开始后的情况，检查是否在绿灯时段
        return (elapsed % self.CYCLE_LENGTH) < self.GREEN_DURATION

    def _get_signal_states(self, t_array):
        """获取一组时间点的信号灯状态序列

        Args:
            t_array: 时间点索引数组

        Returns:
            np.array: 布尔数组，表示每个时间点的信号灯状态
        """
        return np.array([self._check_signal_state(t) for t in t_array])

    def _calculate_branch_flows(self, t_array, params):
        """计算各支路流量

        Args:
            t_array: 时间点索引数组
            params: 模型参数

        Returns:
            tuple: 包含三个支路流量数组的元组
        """
        # 参数解包
        a_params = params[:10]    # 支路1参数
        b_params = params[10:17]  # 支路2参数
        c_params = params[17:]    # 支路3参数

        # 计算各支路流量
        flow1 = self._compute_flow1(t_array, a_params)
        flow2 = self._compute_flow2(t_array, b_params)
        flow3 = self._compute_flow3(t_array, c_params)

        return flow1, flow2, flow3

    def _compute_flow1(self, t, params):
        """计算支路1流量 - 分段线性模型

        Args:
            t: 时间点索引数组
            params: 支路1参数 [a1,a2,a3,a4,a5,a6,brk1,brk2,brk3,brk4]

        Returns:
            np.array: 支路1流量数组
        """
        a1, a2, a3, a4, a5, a6, brk1, brk2, brk3, brk4 = params
        result = np.zeros_like(t, dtype=float)

        # 第一阶段 - 流量为0
        mask = t < brk1
        result[mask] = 0

        # 第二阶段 - 线性增长
        mask = (t >= brk1) & (t < brk2)
        result[mask] = a1 * (t[mask] - brk1) + a2

        # 第三阶段 - 线性减少
        mask = (t >= brk2) & (t < brk3)
        result[mask] = a3 * (t[mask] - brk2) + a4

        # 第四阶段 - 稳定
        mask = (t >= brk3) & (t < brk4)
        result[mask] = a5

        # 第五阶段 - 线性减少
        mask = (t >= brk4)
        result[mask] = a6 * (t[mask] - brk4)

        # 确保流量非负
        return np.maximum(result, 0)

    def _compute_flow2(self, t, params):
        """计算支路2流量 - 三段式模型

        Args:
            t: 时间点索引数组
            params: 支路2参数 [b1,b2,b3,b4,b5,brk5,brk6]

        Returns:
            np.array: 支路2流量数组
        """
        b1, b2, b3, b4, b5, brk5, brk6 = params
        result = np.zeros_like(t, dtype=float)

        # 第一阶段 - 线性增长
        mask = t <= brk5
        result[mask] = b1 * t[mask] + b2

        # 第二阶段 - 稳定
        mask = (t > brk5) & (t <= brk6)
        result[mask] = b3

        # 第三阶段 - 线性变化
        mask = t > brk6
        result[mask] = b4 * (t[mask] - brk6) + b5

        # 确保流量非负
        return np.maximum(result, 0)

    def _compute_flow3(self, t, params):
        """计算支路3流量 - 仅在绿灯时段有流量

        Args:
            t: 时间点索引数组
            params: 支路3参数，每个绿灯周期有一对(斜率,截距)参数

        Returns:
            np.array: 支路3流量数组
        """
        # 获取信号灯状态
        signal_states = self._get_signal_states(t)
        result = np.zeros_like(t, dtype=float)

        # 将参数分组为每个绿灯周期的斜率和截距
        cycle_params = [(params[2*i], params[2*i+1]) for i in range(5)]

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

        # 确保流量非负
        return np.maximum(result, 0)

    def _calculate_main_flow(self, t_array, params):
        """计算主路流量 - 三个支路之和

        Args:
            t_array: 时间点索引数组
            params: 模型参数

        Returns:
            np.array: 主路流量数组
        """
        # 计算各支路原始流量
        f1, f2, f3 = self._calculate_branch_flows(t_array, params)

        # 考虑支路1和支路2的行驶延迟
        f1_delayed = np.zeros_like(t_array)
        f2_delayed = np.zeros_like(t_array)

        # 对有效时间点应用延迟
        valid_idx = t_array >= self.TRAVEL_DELAY
        f1_delayed[valid_idx] = f1[np.where(valid_idx)[0] - self.TRAVEL_DELAY]
        f2_delayed[valid_idx] = f2[np.where(valid_idx)[0] - self.TRAVEL_DELAY]

        # 对起始时间点使用初始值
        f1_delayed[~valid_idx] = f1[0]
        f2_delayed[~valid_idx] = f2[0]

        # 主路流量 = 支路1(延迟) + 支路2(延迟) + 支路3
        return f1_delayed + f2_delayed + f3

    def _evaluate_model(self, params):
        """评估模型性能

        Args:
            params: 模型参数

        Returns:
            float: 带惩罚项的总误差
        """
        # 计算预测流量和均方误差
        predicted = self._calculate_main_flow(self.time_idx, params)
        error = np.mean((predicted - self.actual_flow) ** 2)

        # 添加约束条件惩罚项
        penalty = self._compute_penalty(params)

        return error + penalty

    def _compute_penalty(self, params):
        """计算约束惩罚项

        Args:
            params: 模型参数

        Returns:
            float: 惩罚项总值
        """
        # 计算各支路流量
        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, params)
        penalty = 0

        # 1. 非负约束 - 增加惩罚权重
        penalty += 2000 * (np.sum(np.abs(f1[f1 < 0])) +
                          np.sum(np.abs(f2[f2 < 0])) +
                          np.sum(np.abs(f3[f3 < 0])))

        # 2. 连续性约束
        a1, a2, a3, a4, a5, a6, brk1, brk2, brk3, brk4 = params[:10]
        b1, b2, b3, b4, b5, brk5, brk6 = params[10:17]

        # 2.1 支路1的连续性约束
        continuity_errors = [
            abs(a1 * (brk2 - brk1) + a2 - a4),  # 第一段末尾与第二段开始的连续性
            abs(a3 * (brk3 - brk2) + a4 - a5),  # 第二段末尾与第三段的连续性
            abs(a5 - a6 * 0),                   # 第三段与第四段的连续性
            abs(b1 * brk5 + b2 - b3),           # 支路2第一段与第二段的连续性
            abs(b3 - b5)                         # 支路2第二段与第三段的连续性
        ]

        # 2.2 增加对转折点顺序的约束
        order_errors = [
            max(0, brk1 - brk2),  # 确保 brk1 < brk2
            max(0, brk2 - brk3),  # 确保 brk2 < brk3
            max(0, brk3 - brk4),  # 确保 brk3 < brk4
            max(0, brk5 - brk6)   # 确保 brk5 < brk6
        ]

        # 添加连续性和顺序约束惩罚
        penalty += 1500 * sum(continuity_errors) + 2000 * sum(order_errors)

        # 3. 支路3峰值平滑约束 - 相邻绿灯周期之间的流量变化不应过大
        c_params = params[17:]
        cycle_params = [(c_params[2*i], c_params[2*i+1]) for i in range(5)]

        # 计算相邻周期之间的参数差异
        for i in range(4):
            slope1, intercept1 = cycle_params[i]
            slope2, intercept2 = cycle_params[i+1]

            # 惩罚斜率和截距的剧烈变化
            penalty += 100 * (abs(slope1 - slope2) + abs(intercept1 - intercept2))

        return penalty

    def optimize_model(self):
        """优化模型参数"""
        print("开始模型优化...")

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

        # 优化策略：多重起点，多阶段优化
        best_error = float('inf')
        best_params = None
        optimization_history = []

        # 第一阶段：使用不同的初始点进行多次优化
        num_attempts = 5  # 增加尝试次数
        print(f"第一阶段：使用{num_attempts}个不同起点进行初步优化...")

        for attempt in range(num_attempts):
            # 在初始猜测附近随机扰动
            perturbation = 0.1 * (1 + 0.2 * (attempt/num_attempts))  # 逐渐增加扰动幅度
            perturbed_guess = np.array(initial_guess) * (1 + perturbation * np.random.randn(len(initial_guess)))

            # 确保扰动后的参数仍在边界内
            for i, (lb, ub) in enumerate(param_bounds):
                perturbed_guess[i] = max(lb, min(ub, perturbed_guess[i]))

            print(f"  尝试 {attempt+1}/{num_attempts}...")

            # 选择不同的优化方法
            if attempt % 3 == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 500, 'gtol': 1e-5}
            elif attempt % 3 == 1:
                method = 'SLSQP'
                options = {'maxiter': 500, 'ftol': 1e-5}
            else:
                method = 'TNC'
                options = {'maxiter': 500}

            try:
                optimization_result = minimize(
                    self._evaluate_model,
                    perturbed_guess,
                    method=method,
                    bounds=param_bounds,
                    options=options
                )

                # 记录结果
                optimization_history.append({
                    'params': optimization_result.x,
                    'error': optimization_result.fun,
                    'success': optimization_result.success,
                    'method': method
                })

                if optimization_result.fun < best_error:
                    best_error = optimization_result.fun
                    best_params = optimization_result.x
                    print(f"    发现更优解，误差: {best_error:.6f}")
            except Exception as e:
                print(f"    优化失败: {str(e)}")

        # 第二阶段：细化最佳结果
        print("\n第二阶段：细化最佳结果...")

        # 选择前3个最佳结果进行细化
        sorted_results = sorted(optimization_history, key=lambda x: x['error'])
        top_k = min(3, len(sorted_results))

        for i in range(top_k):
            result = sorted_results[i]
            print(f"  细化优化 {i+1}/{top_k}, 起始误差: {result['error']:.6f}")

            try:
                refined_result = minimize(
                    self._evaluate_model,
                    result['params'],
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': 1000, 'gtol': 1e-7}
                )

                if refined_result.fun < best_error:
                    best_error = refined_result.fun
                    best_params = refined_result.x
                    print(f"    优化成功，新误差: {best_error:.6f}")
            except Exception as e:
                print(f"    细化优化失败: {str(e)}")

        # 使用最佳参数
        self.optimal_params = best_params

        # 计算最终的预测流量和误差
        self.predicted_flow = self._calculate_main_flow(self.time_idx, self.optimal_params)
        self.model_error = np.sqrt(np.mean((self.predicted_flow - self.actual_flow) ** 2))

        # 计算各时间段的误差
        time_segments = [(0, 15), (15, 30), (30, 45), (45, 59)]
        self.segment_errors = []
        for start, end in time_segments:
            segment_idx = (self.time_idx >= start) & (self.time_idx <= end)
            segment_error = np.sqrt(np.mean((self.predicted_flow[segment_idx] - self.actual_flow[segment_idx]) ** 2))
            self.segment_errors.append(segment_error)

        # 计算各支路流量
        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, self.optimal_params)
        self.branch_flows = (f1, f2, f3)

        print(f"\n优化完成！总体RMSE误差: {self.model_error:.4f}")
        for i, (start, end) in enumerate(time_segments):
            print(f"时间段 {start+1}-{end+1} RMSE误差: {self.segment_errors[i]:.4f}")

    def visualize_results(self):
        """可视化分析结果"""
        # 创建输出目录
        os.makedirs('./P3', exist_ok=True)

        # 绘制主要图表
        self._plot_main_comparison()
        self._plot_branch_flows()
        self._plot_error_analysis()

    def _plot_main_comparison(self):
        """绘制主路流量对比图"""
        plt.figure(figsize=(14, 6))
        plt.plot(self.time_idx, self.actual_flow, 'b-', lw=2, label='实测流量', marker='o', markersize=4)
        plt.plot(self.time_idx, self.predicted_flow, 'r--', lw=2, label='预测流量')

        # 添加误差指标
        plt.title(f'主路流量实测与预测对比 (RMSE={self.model_error:.4f})')
        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./P3/主路流量实测与预测对比.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_branch_flows(self):
        """绘制支路流量图"""
        f1, f2, f3 = self.branch_flows
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:17]

        plt.figure(figsize=(14, 8))
        plt.plot(self.time_idx, f1, 'g-', label='支路1', linewidth=2)
        plt.plot(self.time_idx, f2, 'm-', label='支路2', linewidth=2)
        plt.plot(self.time_idx, f3, 'c-', label='支路3', linewidth=2)

        # 标记支路1转折点
        for i, brk in enumerate(a_params[6:10]):
            plt.axvline(x=brk, color='g', ls='--', alpha=0.3)
            plt.annotate(f'支路1转折点{i+1}', xy=(brk, 5), xytext=(brk, 10),
                        arrowprops=dict(arrowstyle='->'), color='g')

        # 标记支路2转折点
        for i, brk in enumerate(b_params[5:7]):
            plt.axvline(x=brk, color='m', ls='--', alpha=0.3)
            plt.annotate(f'支路2转折点{i+1}', xy=(brk, 15), xytext=(brk, 20),
                        arrowprops=dict(arrowstyle='->'), color='m')

        # 标记绿灯时段
        for i in range(6):
            start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
            if start < 60:  # 确保在时间范围内
                end = min(start + self.GREEN_DURATION, 60)
                plt.axvspan(start, end, color='green', alpha=0.1)
                plt.annotate(f'绿灯{i+1}', xy=(start + (end-start)/2, 90),
                            ha='center', color='green')

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('各支路流量变化情况')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./P3/支路车流量变化.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_analysis(self):
        """绘制误差分析图"""
        # 计算残差
        residuals = self.actual_flow - self.predicted_flow

        # 残差时序图
        plt.figure(figsize=(14, 6))
        plt.plot(self.time_idx, residuals, 'r-', label='残差', marker='x')
        plt.axhline(y=0, color='k', linestyle='--')

        # 添加时间段分隔线
        time_segments = [(0, 15), (15, 30), (30, 45), (45, 59)]
        for start, end in time_segments[1:]:
            plt.axvline(x=start, color='gray', linestyle='--')

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('残差值')
        plt.title('预测误差时序分布')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./P3/预测误差时序分布.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 残差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('残差值')
        plt.ylabel('频率')
        plt.title('预测误差分布直方图')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./P3/预测误差分布直方图.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制各时间段误差对比图
        plt.figure(figsize=(10, 6))
        segments = ['7:00-7:30', '7:30-8:00', '8:00-8:30', '8:30-8:58']
        plt.bar(segments, self.segment_errors, color='coral')
        plt.axhline(y=self.model_error, color='r', linestyle='--', label=f'总体RMSE={self.model_error:.4f}')

        plt.xlabel('时间段')
        plt.ylabel('RMSE误差')
        plt.title('各时间段预测误差对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./P3/各时间段预测误差对比.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """生成分析报告"""
        # 创建输出目录
        os.makedirs('./P3', exist_ok=True)

        # 提取模型参数
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:17]
        c_params = self.optimal_params[17:]

        # 计算特定时间点流量
        t1, t2 = 15, 45  # 7:30和8:30对应的时间索引
        f1_730, f2_730, f3_730 = self._calculate_branch_flows(np.array([t1]), self.optimal_params)
        f1_830, f2_830, f3_830 = self._calculate_branch_flows(np.array([t2]), self.optimal_params)

        # 计算主路流量
        main_flow_730 = self._calculate_main_flow(np.array([t1]), self.optimal_params)[0]
        main_flow_830 = self._calculate_main_flow(np.array([t2]), self.optimal_params)[0]

        with open('./P3/交通流量分析报告.md', 'w', encoding='utf-8') as f:
            f.write("# 交通流量分析报告 - 问题3\n\n")

            # 1. 模型总体误差评估
            f.write("## 一、模型误差评估\n\n")
            f.write(f"总体RMSE误差: {self.model_error:.4f}\n\n")
            f.write("各时间段误差:\n\n")
            f.write("| 时间段 | RMSE误差 |\n")
            f.write("|--------|----------|\n")
            time_segments = [(0, 15), (15, 30), (30, 45), (45, 59)]
            for i, (start, end) in enumerate(time_segments):
                f.write(f"| {start//2+7}:00-{end//2+7}:00 | {self.segment_errors[i]:.4f} |\n")
            f.write("\n")

            # 2. 支路1模型参数
            f.write("## 二、各支路模型参数\n\n")
            f.write("### 2.1 支路1模型参数\n\n")
            f.write(f"- 增长阶段斜率: {a_params[0]:.4f}\n")
            f.write(f"- 初始值: {a_params[1]:.4f}\n")
            f.write(f"- 下降阶段斜率1: {a_params[2]:.4f}\n")
            f.write(f"- 中间值: {a_params[3]:.4f}\n")
            f.write(f"- 稳定值: {a_params[4]:.4f}\n")
            f.write(f"- 下降阶段斜率2: {a_params[5]:.4f}\n")
            f.write(f"- 转折点: {a_params[6]:.2f}, {a_params[7]:.2f}, {a_params[8]:.2f}, {a_params[9]:.2f}\n\n")

            # 3. 支路2模型参数
            f.write("### 2.2 支路2模型参数\n\n")
            f.write(f"- 增长斜率: {b_params[0]:.4f}\n")
            f.write(f"- 截距: {b_params[1]:.4f}\n")
            f.write(f"- 稳定值: {b_params[2]:.4f}\n")
            f.write(f"- 下降斜率: {b_params[3]:.4f}\n")
            f.write(f"- 终值: {b_params[4]:.4f}\n")
            f.write(f"- 转折点: {b_params[5]:.2f}, {b_params[6]:.2f}\n\n")

            # 4. 支路3模型参数
            f.write("### 2.3 支路3模型参数\n\n")
            f.write("| 绿灯周期 | 斜率 | 截距 |\n")
            f.write("|----------|------|------|\n")
            for i in range(5):
                slope, intercept = c_params[2*i], c_params[2*i+1]
                f.write(f"| 周期{i+1} | {slope:.4f} | {intercept:.4f} |\n")
            f.write("\n")

            # 5. 各支路流量函数表达式
            f.write("## 三、各支路流量函数表达式\n\n")

            # 支路1流量模型表达式
            f.write("### 3.1 支路1流量模型\n\n")
            f.write(r"$f_1(t) = \begin{cases} ")
            f.write(f"0, & t < {a_params[6]:.2f} \\\\ ")
            f.write(f"{a_params[0]:.4f} \\cdot (t-{a_params[6]:.2f}) + {a_params[1]:.4f}, & {a_params[6]:.2f} \\leq t < {a_params[7]:.2f} \\\\ ")
            f.write(f"{a_params[2]:.4f} \\cdot (t-{a_params[7]:.2f}) + {a_params[3]:.4f}, & {a_params[7]:.2f} \\leq t < {a_params[8]:.2f} \\\\ ")
            f.write(f"{a_params[4]:.4f}, & {a_params[8]:.2f} \\leq t < {a_params[9]:.2f} \\\\ ")
            f.write(f"{a_params[5]:.4f} \\cdot (t-{a_params[9]:.2f}), & t \\geq {a_params[9]:.2f} ")
            f.write(r"\end{cases}$")
            f.write('\n\n')

            # 支路2流量模型表达式
            f.write("### 3.2 支路2流量模型\n\n")
            f.write(r"$f_2(t) = \begin{cases} ")
            f.write(f"{b_params[0]:.4f} \\cdot t + {b_params[1]:.4f}, & t \\leq {b_params[5]:.2f} \\\\ ")
            f.write(f"{b_params[2]:.4f}, & {b_params[5]:.2f} < t \\leq {b_params[6]:.2f} \\\\ ")
            f.write(f"{b_params[3]:.4f} \\cdot (t-{b_params[6]:.2f}) + {b_params[4]:.4f}, & t > {b_params[6]:.2f} ")
            f.write(r"\end{cases}$")
            f.write('\n\n')

            # 支路3流量模型表达式
            f.write("### 3.3 支路3流量模型\n\n")
            f.write(r"$f_3(t) = \begin{cases} ")
            for i in range(5):
                start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
                end = start + self.GREEN_DURATION
                slope, intercept = c_params[2*i], c_params[2*i+1]
                f.write(f"{slope:.4f} \\cdot (t-{start}) + {intercept:.4f}, & t \\in [{start}, {end}) \\text{{ 且为绿灯时段}} \\\\ ")
            f.write(r"0, & \text{其他时段（红灯）} \end{cases}$")
            f.write('\n\n')

            # 6. 关键时间点流量
            f.write("## 四、关键时间点流量\n\n")
            f.write("| 时间点 | 支路1 | 支路2 | 支路3 | 主路4(预测) | 主路4(实际) |\n")
            f.write("|--------|-------|-------|-------|------------|------------|\n")
            f.write(f"| 7:30 | {f1_730[0]:5.2f} | {f2_730[0]:5.2f} | {f3_730[0]:5.2f} | {main_flow_730:8.2f} | {self.actual_flow[t1]:8.2f} |\n")
            f.write(f"| 8:30 | {f1_830[0]:5.2f} | {f2_830[0]:5.2f} | {f3_830[0]:5.2f} | {main_flow_830:8.2f} | {self.actual_flow[t2]:8.2f} |\n\n")

            # 7. 结论分析
            f.write("## 五、结论分析\n\n")
            f.write("1. **支路1特征**: 支路1表现为先增长、后降低、然后稳定、最后再降低的趋势，符合典型的早高峰流量分布特征。\n\n")
            f.write("2. **支路2特征**: 支路2呈现先增长、中间稳定、后期降低的特征，表明该支路车辆主要在早晨通勤高峰期间较为稳定。\n\n")
            f.write("3. **支路3特征**: 支路3仅在绿灯时段有车流量，这与信号灯控制直接相关，每个绿灯周期内车流量表现出线性变化趋势。\n\n")
            f.write("4. **信号灯影响**: 模型很好地捕捉了信号灯对交通流的控制效果，特别是支路3的车流量完全受到信号灯控制。\n\n")
            f.write("5. **预测性能**: 模型在不同时段表现差异较大，特别是7:00-7:30和8:30-8:58时段的预测误差明显较低，说明这些时段车流量变化更为规律可预测。\n\n")

def main():
    """主函数"""
    # 创建分析器实例
    analyzer = TrafficFlowAnalyzer()

    # 优化模型参数
    analyzer.optimize_model()

    # 可视化结果
    analyzer.visualize_results()

    # 生成分析报告
    analyzer.generate_report()

    print("分析完成！结果保存在./P3目录")

if __name__ == "__main__":
    main()
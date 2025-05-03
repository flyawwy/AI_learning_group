import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize
import sys
import os

# 添加父目录到路径，以便导入P2和P3中的函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class KeySamplingPointsAnalyzer:
    """关键采样点分析器 - 用于P5问题求解"""

    def __init__(self, data_source='P2'):
        """初始化分析器

        Args:
            data_source: 'P2'或'P3'，指定使用哪个问题的数据和模型
        """
        self.data_source = data_source
        self._load_data()
        self.sampling_points = []
        self.sampling_indices = []

    def _load_data(self):
        """加载数据"""
        if self.data_source == 'P2':
            # 加载P2数据（表2）
            df = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx',
                              sheet_name='表2 (Table 2)')
            self.times = df['时间 t (Time t)'].values
            self.flow_data = df['主路5的车流量 (Traffic flow on the Main road 5)'].values
        else:
            # 加载P3数据（表3）
            df = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx',
                              sheet_name='表3 (Table 3)')
            self.times = df['时间 t (Time t)'].values
            self.flow_data = df['主路4的车流量 (Traffic flow on the Main road 4)'].values

        self.time_indices = np.arange(len(self.times))

    def identify_key_points(self, method='combined'):
        """识别关键采样点

        Args:
            method: 采样点识别方法，可选值：
                   'derivative' - 基于导数变化
                   'peaks' - 基于峰值检测
                   'breakpoints' - 基于模型转折点
                   'combined' - 结合以上所有方法（默认）
        """
        if method == 'derivative' or method == 'combined':
            self._identify_by_derivative()

        if method == 'peaks' or method == 'combined':
            self._identify_by_peaks()

        if method == 'breakpoints' or method == 'combined':
            self._identify_by_breakpoints()

        # 去重并排序
        self.sampling_indices = sorted(list(set(self.sampling_indices)))
        self.sampling_points = self.times[self.sampling_indices]

        # 确保包含起始和结束点
        if 0 not in self.sampling_indices:
            self.sampling_indices.insert(0, 0)
            self.sampling_points = self.times[self.sampling_indices]

        if len(self.times) - 1 not in self.sampling_indices:
            self.sampling_indices.append(len(self.times) - 1)
            self.sampling_points = self.times[self.sampling_indices]

    def _identify_by_derivative(self, threshold=1.5):
        """基于导数变化识别关键点"""
        # 计算一阶差分（近似导数）
        diff1 = np.diff(self.flow_data)
        # 计算二阶差分（导数变化率）
        diff2 = np.diff(diff1)

        # 找出导数变化显著的点
        mean_abs_diff2 = np.mean(np.abs(diff2))
        significant_changes = np.where(np.abs(diff2) > threshold * mean_abs_diff2)[0] + 1

        # 添加到采样点列表
        self.sampling_indices.extend(significant_changes)

    def _identify_by_peaks(self, prominence=3.0):
        """基于峰值和谷值检测关键点"""
        # 检测峰值
        peaks, _ = find_peaks(self.flow_data, prominence=prominence)

        # 检测谷值（将数据取反再检测峰值）
        valleys, _ = find_peaks(-self.flow_data, prominence=prominence)

        # 添加到采样点列表
        self.sampling_indices.extend(peaks)
        self.sampling_indices.extend(valleys)

    def _identify_by_breakpoints(self):
        """基于模型转折点识别关键点"""
        if self.data_source == 'P2':
            # P2模型的转折点
            # 支路1: 常数
            # 支路2: t=24和t=37是转折点
            # 支路3: 有一个转折点t2
            # 支路4: 周期性变化，无明显转折点
            breakpoints = [24, 37]

            # 找出最接近这些转折点的时间索引
            for bp in breakpoints:
                idx = np.argmin(np.abs(self.times - bp))
                self.sampling_indices.append(idx)

            # 对于支路3的转折点，我们需要估计一个合理值
            # 根据数据观察，大约在t=40左右
            self.sampling_indices.append(np.argmin(np.abs(self.times - 40)))

        else:  # P3模型
            # 支路1: 有4个转折点
            # 支路2: 有2个转折点
            # 支路3: 绿灯周期的开始和结束是关键点

            # 估计支路1的转折点（根据P3模型的初始猜测）
            branch1_breakpoints = [5.0, 12.0, 16.0, 47.0]

            # 估计支路2的转折点
            branch2_breakpoints = [25.0, 44.0]

            # 绿灯周期的开始和结束点
            red_duration = 4  # 对应8分钟
            green_duration = 5  # 对应10分钟
            cycle_length = red_duration + green_duration
            first_green = 3  # 第一个绿灯开始时间

            green_starts = [first_green + i * cycle_length for i in range(5)]
            green_ends = [start + green_duration for start in green_starts]

            # 合并所有转折点
            all_breakpoints = branch1_breakpoints + branch2_breakpoints + green_starts + green_ends

            # 找出最接近这些转折点的时间索引
            for bp in all_breakpoints:
                if 0 <= bp < len(self.times):
                    idx = np.argmin(np.abs(self.times - bp))
                    self.sampling_indices.append(idx)

    def evaluate_sampling_points(self, original_params=None):
        """评估采样点的有效性

        通过比较使用全部数据点和仅使用采样点拟合的模型参数差异

        Args:
            original_params: 使用全部数据点拟合的模型参数，如果为None则重新拟合

        Returns:
            评估结果字典，包含参数差异、拟合误差等
        """
        if self.data_source == 'P2':
            from P2 import total_flow, objective

            # 如果没有提供原始参数，使用默认参数
            if original_params is None:
                # P2默认参数（简化版）
                original_params = [
                    5.0, 0.45, 10.0, 0.2, 0.48, 10.5, 39.4, 5, 25, 5,
                    *[1]*20  # 傅里叶系数
                ]

            # 使用采样点拟合模型
            sampled_times = self.times[self.sampling_indices]
            sampled_flow = self.flow_data[self.sampling_indices]

            # 定义采样点的目标函数
            def sampled_objective(params):
                return objective(params, sampled_times, sampled_flow)

            # 优化采样点模型
            from scipy.optimize import minimize
            bounds = [
                (5, 10), (0.4, 0.5), (10, 20), (0.2, 0.3), (0.4, 0.5), (10, 20), (35, 45),
                (3, 10), (10, 45), (0, 10), *[(-10, 10)]*20
            ]
            result = minimize(sampled_objective, original_params, method='L-BFGS-B', bounds=bounds)
            sampled_params = result.x

            # 计算全部数据点的预测值
            full_predicted = total_flow(self.times, original_params)
            sampled_predicted = total_flow(self.times, sampled_params)

            # 计算误差
            full_error = np.mean((full_predicted - self.flow_data)**2)
            sampled_error = np.mean((sampled_predicted - self.flow_data)**2)

            # 计算参数差异
            param_diff = np.mean(np.abs(np.array(original_params) - np.array(sampled_params)))

            return {
                'full_error': full_error,
                'sampled_error': sampled_error,
                'param_diff': param_diff,
                'full_params': original_params,
                'sampled_params': sampled_params,
                'full_predicted': full_predicted,
                'sampled_predicted': sampled_predicted
            }

        else:  # P3模型
            from p3 import TrafficAnalysisSystem

            # 如果没有提供原始参数，使用默认参数或重新拟合
            if original_params is None:
                # 创建交通分析系统并优化模型
                traffic_system = TrafficAnalysisSystem()
                traffic_system.optimize_model()
                original_params = traffic_system.optimal_params

            # 使用采样点拟合模型
            sampled_times = self.times[self.sampling_indices]
            sampled_flow = self.flow_data[self.sampling_indices]

            # 创建一个新的交通分析系统用于采样点拟合
            sampled_system = TrafficAnalysisSystem()

            # 保存原始数据
            original_time_idx = sampled_system.time_idx
            original_actual_flow = sampled_system.actual_flow

            # 替换为采样点数据
            sampled_system.time_idx = sampled_times
            sampled_system.actual_flow = sampled_flow

            # 优化采样点模型
            sampled_system.optimize_model()
            sampled_params = sampled_system.optimal_params

            # 恢复原始数据用于评估
            sampled_system.time_idx = original_time_idx
            sampled_system.actual_flow = original_actual_flow

            # 计算全部数据点的预测值
            full_predicted = sampled_system._calculate_main_flow(self.times, original_params)
            sampled_predicted = sampled_system._calculate_main_flow(self.times, sampled_params)

            # 计算误差
            full_error = np.mean((full_predicted - self.flow_data)**2)
            sampled_error = np.mean((sampled_predicted - self.flow_data)**2)

            # 计算参数差异
            param_diff = np.mean(np.abs(np.array(original_params) - np.array(sampled_params)))

            # 计算采样点覆盖的数据特征比例
            # 计算数据范围覆盖率
            full_range = np.max(self.flow_data) - np.min(self.flow_data)
            sampled_range = np.max(sampled_flow) - np.min(sampled_flow)
            range_coverage = sampled_range / full_range if full_range > 0 else 1.0

            # 计算变化点覆盖率（使用二阶差分检测变化点）
            diff2 = np.diff(np.diff(self.flow_data))
            significant_changes = np.where(np.abs(diff2) > np.mean(np.abs(diff2)))[0] + 1

            # 计算采样点对变化点的覆盖率
            covered_changes = sum(1 for idx in significant_changes
                                if any(abs(idx - s_idx) <= 1 for s_idx in self.sampling_indices))
            change_coverage = covered_changes / len(significant_changes) if len(significant_changes) > 0 else 1.0

            return {
                'full_error': full_error,
                'sampled_error': sampled_error,
                'param_diff': param_diff,
                'full_params': original_params,
                'sampled_params': sampled_params,
                'full_predicted': full_predicted,
                'sampled_predicted': sampled_predicted,
                'range_coverage': range_coverage,
                'change_coverage': change_coverage,
                'sampling_efficiency': len(self.sampling_indices) / len(self.flow_data)
            }

    def visualize_sampling_points(self, evaluation_results=None):
        """可视化采样点及其评估结果"""
        plt.figure(figsize=(14, 8))

        # 绘制原始数据
        plt.plot(self.times, self.flow_data, 'b-', label='原始流量数据')

        # 标记采样点
        plt.plot(self.times[self.sampling_indices], self.flow_data[self.sampling_indices],
                 'ro', markersize=8, label='关键采样点')

        # 如果有评估结果，绘制拟合曲线
        if evaluation_results and 'sampled_predicted' in evaluation_results:
            plt.plot(self.times, evaluation_results['full_predicted'], 'g--',
                     label='全部数据拟合曲线')
            plt.plot(self.times, evaluation_results['sampled_predicted'], 'm--',
                     label='采样点拟合曲线')

            # 添加评估指标到图例
            plt.title(f'关键采样点分析 (采样效率: {len(self.sampling_indices)/len(self.times):.2f}, '
                      f'参数差异: {evaluation_results["param_diff"]:.4f}, '
                      f'误差比: {evaluation_results["sampled_error"]/evaluation_results["full_error"]:.4f})')
        else:
            plt.title(f'关键采样点分析 (采样点数: {len(self.sampling_indices)}, '
                      f'采样效率: {len(self.sampling_indices)/len(self.times):.2f})')

        plt.xlabel('时间 t')
        plt.ylabel('流量')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 保存图像
        save_dir = f'./P5/{self.data_source}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/关键采样点分析.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, evaluation_results=None):
        """生成分析报告"""
        # 创建保存目录
        save_dir = f'./P5/{self.data_source}'
        os.makedirs(save_dir, exist_ok=True)

        with open(f'{save_dir}/关键采样点分析报告.md', 'w', encoding='utf-8') as f:
            f.write(f'# {self.data_source}问题关键采样点分析报告\n\n')

            f.write('## 关键采样点\n\n')
            f.write('| 序号 | 时间索引 | 时间点 | 流量值 |\n')
            f.write('|------|----------|--------|-----------|\n')

            for i, idx in enumerate(self.sampling_indices):
                f.write(f'| {i+1} | {idx} | {self.times[idx]} | {self.flow_data[idx]:.2f} |\n')

            f.write(f'\n总采样点数: {len(self.sampling_indices)}，采样效率: '
                    f'{len(self.sampling_indices)/len(self.times):.2f}\n\n')

            # 如果有评估结果，添加评估指标
            if evaluation_results:
                f.write('## 评估指标\n\n')

                if 'full_error' in evaluation_results:
                    f.write(f'全部数据拟合误差: {evaluation_results["full_error"]:.4f}\n')
                    f.write(f'采样点拟合误差: {evaluation_results["sampled_error"]:.4f}\n')
                    f.write(f'误差比率: {evaluation_results["sampled_error"]/evaluation_results["full_error"]:.4f}\n')
                    f.write(f'参数差异: {evaluation_results["param_diff"]:.4f}\n\n')

                if 'range_coverage' in evaluation_results:
                    f.write(f'数据范围覆盖率: {evaluation_results["range_coverage"]:.4f}\n')
                    f.write(f'变化点覆盖率: {evaluation_results["change_coverage"]:.4f}\n')
                    f.write(f'采样效率: {evaluation_results["sampling_efficiency"]:.4f}\n\n')

            f.write('## 结论\n\n')
            f.write('通过分析，我们确定了上述关键采样点，这些点能够有效地捕捉流量变化特征，'
                    '使用这些采样点可以推断出整个时间段内各支路的车流量函数表达式。\n\n')

            f.write('关键采样点的选择基于以下原则：\n\n')
            f.write('1. 流量变化显著的点（导数变化大）\n')
            f.write('2. 流量峰值和谷值\n')
            f.write('3. 模型转折点\n')
            f.write('4. 信号灯变化点（对于P3问题）\n\n')

            f.write('这些采样点能够有效地捕捉流量变化特征，使用这些采样点可以推断出整个时间段内各支路的车流量函数表达式。\n')

# 如果直接运行此脚本
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='关键采样点分析工具')
    parser.add_argument('--problem', type=str, default='P2', choices=['P2', 'P3'],
                        help='选择问题（P2或P3）')
    parser.add_argument('--method', type=str, default='combined',
                        choices=['derivative', 'peaks', 'breakpoints', 'combined'],
                        help='采样点识别方法')

    args = parser.parse_args()

    # 创建分析器
    analyzer = KeySamplingPointsAnalyzer(data_source=args.problem)

    # 识别关键采样点
    analyzer.identify_key_points(method=args.method)

    # 评估采样点
    evaluation_results = analyzer.evaluate_sampling_points()

    # 可视化结果
    analyzer.visualize_sampling_points(evaluation_results)

    # 生成报告
    analyzer.generate_report(evaluation_results)

    print(f"分析完成！关键采样点数量：{len(analyzer.sampling_indices)}")
    print(f"采样点时间索引：{analyzer.sampling_indices}")
    print(f"采样点时间值：{analyzer.sampling_points}")
    print(f"结果已保存到 ./P5/{args.problem}/ 目录")
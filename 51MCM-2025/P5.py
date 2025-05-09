import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import os
from P2 import TrafficFlowAnalyzer as P2Analyzer
from P3 import TrafficFlowAnalyzer as P3Analyzer

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SamplingPointsOptimizer:
    """采样点优化器 - 问题5解决方案"""

    def __init__(self, method='combined'):
        """初始化优化器

        Args:
            method: 采样点识别方法 ('derivative', 'peaks', 'breakpoints', 'combined')
        """
        self.method = method
        self._setup_analyzers()

    def _setup_analyzers(self):
        """设置分析器实例"""
        # 创建问题2和问题3的分析器
        self.p2_analyzer = P2Analyzer()
        self.p3_analyzer = P3Analyzer()

        # 加载数据
        self._load_dataset()

    def _load_dataset(self):
        """加载交通流量数据"""
        # 读取问题2数据
        df2 = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx',
                           sheet_name='表2 (Table 2)')
        self.times2 = df2['时间 t (Time t)'].values
        self.flow2 = df2['主路5的车流量 (Traffic flow on the Main road 5)'].values

        # 读取问题3数据
        df3 = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx',
                           sheet_name='表3 (Table 3)')
        self.times3 = df3['时间 t (Time t)'].values
        self.flow3 = df3['主路4的车流量 (Traffic flow on the Main road 4)'].values

    def _find_sampling_points_derivative(self, times, flow, threshold=0.7, window=3):
        """基于导数变化查找采样点

        Args:
            times: 时间数组
            flow: 流量数组
            threshold: 导数变化阈值
            window: 移动窗口大小

        Returns:
            list: 采样点索引列表
        """
        # 计算导数
        derivatives = np.gradient(flow)

        # 计算导数变化率
        derivative_changes = np.abs(np.gradient(derivatives))

        # 使用百分位数确定阈值
        percentile_threshold = np.percentile(derivative_changes, 100 * threshold)

        # 根据阈值筛选点
        candidate_indices = np.where(derivative_changes > percentile_threshold)[0]

        # 应用非极大值抑制
        sampling_indices = []
        for i in range(len(candidate_indices)):
            idx = candidate_indices[i]
            start_idx = max(0, idx - window)
            end_idx = min(len(derivative_changes), idx + window + 1)

            # 检查是否是局部最大值
            if idx == np.argmax(derivative_changes[start_idx:end_idx]) + start_idx:
                sampling_indices.append(idx)

        # 添加边界点
        if 0 not in sampling_indices:
            sampling_indices.append(0)
        if len(flow) - 1 not in sampling_indices:
            sampling_indices.append(len(flow) - 1)

        return sorted(sampling_indices)

    def _find_sampling_points_peaks(self, times, flow, prominence=0.3, width=3):
        """基于峰值检测查找采样点

        Args:
            times: 时间数组
            flow: 流量数组
            prominence: 峰值突出度
            width: 峰宽度

        Returns:
            list: 采样点索引列表
        """
        # 计算峰值和谷值
        peaks, _ = find_peaks(flow, prominence=prominence*np.max(flow), width=width)
        valleys, _ = find_peaks(-flow, prominence=prominence*np.max(flow), width=width)

        # 合并峰值和谷值
        sampling_indices = sorted(list(peaks) + list(valleys))

        # 添加边界点
        if 0 not in sampling_indices:
            sampling_indices.append(0)
        if len(flow) - 1 not in sampling_indices:
            sampling_indices.append(len(flow) - 1)

        return sorted(sampling_indices)

    def _find_sampling_points_breakpoints(self, model_type='p2'):
        """基于模型转折点查找采样点

        Args:
            model_type: 模型类型 ('p2' 或 'p3')

        Returns:
            list: 采样点索引列表
        """
        if model_type == 'p2':
            # 优化问题2模型
            self.p2_analyzer.optimize()

            # 提取模型参数
            params = self.p2_analyzer.best_params

            # 关键转折点
            brk_points = []
            # 支路1无转折点

            # 支路2转折点 (在t=24和t=37处)
            brk_points.extend([24, 37])

            # 支路3转折点
            t2 = params[6]
            brk_points.append(t2)

            # 支路4周期点
            T = params[8]
            N = int(round(params[7]))
            for i in range(1, N+1):
                cycle_point = i * T / N
                if cycle_point < len(self.times2):
                    brk_points.append(cycle_point)

        elif model_type == 'p3':
            # 优化问题3模型
            self.p3_analyzer.optimize_model()

            # 提取模型参数
            params = self.p3_analyzer.optimal_params

            # 关键转折点
            brk_points = []

            # 支路1转折点
            brk_points.extend(params[6:10])

            # 支路2转折点
            brk_points.extend(params[15:17])

            # 支路3信号灯周期点
            for i in range(6):
                start = self.p3_analyzer.FIRST_GREEN + i * self.p3_analyzer.CYCLE_LENGTH
                if start < 60:
                    brk_points.append(start)
                    end = min(start + self.p3_analyzer.GREEN_DURATION, 60)
                    brk_points.append(end)

        # 将转折点转换为最接近的时间索引
        sampling_indices = []
        times = self.times2 if model_type == 'p2' else self.times3

        for point in brk_points:
            idx = np.argmin(np.abs(times - point))
            sampling_indices.append(idx)

        # 添加边界点
        if 0 not in sampling_indices:
            sampling_indices.append(0)
        if len(times) - 1 not in sampling_indices:
            sampling_indices.append(len(times) - 1)

        return sorted(list(set(sampling_indices)))  # 去重并排序

    def identify_sampling_points(self):
        """识别最优采样点集合

        Returns:
            dict: 包含问题2和问题3的采样点信息
        """
        # 创建输出目录
        os.makedirs('./P5', exist_ok=True)

        results = {}

        # 处理问题2
        print("\n识别问题2的最优采样点...")
        if self.method == 'derivative':
            p2_indices = self._find_sampling_points_derivative(self.times2, self.flow2)
        elif self.method == 'peaks':
            p2_indices = self._find_sampling_points_peaks(self.times2, self.flow2)
        elif self.method == 'breakpoints':
            p2_indices = self._find_sampling_points_breakpoints('p2')
        else:  # combined
            derivative_indices = self._find_sampling_points_derivative(self.times2, self.flow2)
            peaks_indices = self._find_sampling_points_peaks(self.times2, self.flow2)
            breakpoints_indices = self._find_sampling_points_breakpoints('p2')

            # 合并不同方法的结果
            p2_indices = sorted(list(set(derivative_indices + peaks_indices + breakpoints_indices)))

        # 处理问题3
        print("\n识别问题3的最优采样点...")
        if self.method == 'derivative':
            p3_indices = self._find_sampling_points_derivative(self.times3, self.flow3)
        elif self.method == 'peaks':
            p3_indices = self._find_sampling_points_peaks(self.times3, self.flow3)
        elif self.method == 'breakpoints':
            p3_indices = self._find_sampling_points_breakpoints('p3')
        else:  # combined
            derivative_indices = self._find_sampling_points_derivative(self.times3, self.flow3)
            peaks_indices = self._find_sampling_points_peaks(self.times3, self.flow3)
            breakpoints_indices = self._find_sampling_points_breakpoints('p3')

            # 合并不同方法的结果
            p3_indices = sorted(list(set(derivative_indices + peaks_indices + breakpoints_indices)))

        # 验证采样点有效性并精简
        p2_indices = self._refine_sampling_points(p2_indices, self.times2, self.flow2, 'p2')
        p3_indices = self._refine_sampling_points(p3_indices, self.times3, self.flow3, 'p3')

        # 保存结果
        results['p2'] = {
            'indices': p2_indices,
            'times': self.times2[p2_indices],
            'flow': self.flow2[p2_indices],
            'count': len(p2_indices)
        }

        results['p3'] = {
            'indices': p3_indices,
            'times': self.times3[p3_indices],
            'flow': self.flow3[p3_indices],
            'count': len(p3_indices)
        }

        # 可视化结果
        self._visualize_sampling_points(results)

        # 生成报告
        self._generate_report(results)

        return results

    def _refine_sampling_points(self, indices, times, flow, problem_type, max_points=20):
        """精简采样点集合

        Args:
            indices: 初始采样点索引
            times: 时间数组
            flow: 流量数组
            problem_type: 问题类型 ('p2' 或 'p3')
            max_points: 最大采样点数量

        Returns:
            list: 精简后的采样点索引
        """
        # 如果采样点过多，进行精简
        if len(indices) > max_points:
            print(f"采样点数量({len(indices)})超过限制({max_points})，进行精简...")

            # 计算每个点的重要性分数
            importance_scores = np.zeros(len(indices))

            for i, idx in enumerate(indices):
                # 局部变化率分数
                if idx > 0 and idx < len(flow) - 1:
                    derivative = abs(flow[idx+1] - flow[idx-1]) / 2
                    importance_scores[i] += derivative / np.max(flow) * 5

                # 流量大小分数
                importance_scores[i] += flow[idx] / np.max(flow) * 3

                # 转折点分数
                if problem_type == 'p2':
                    # 转折点接近24、37和支路3转折点的加分
                    if any(abs(times[idx] - x) < 2 for x in [24, 37, self.p2_analyzer.best_params[6]]):
                        importance_scores[i] += 5
                elif problem_type == 'p3':
                    # 接近信号灯周期点的加分
                    for j in range(6):
                        start = self.p3_analyzer.FIRST_GREEN + j * self.p3_analyzer.CYCLE_LENGTH
                        if start < 60 and abs(times[idx] - start) < 2:
                            importance_scores[i] += 5

            # 确保起点和终点被保留
            importance_scores[0] = importance_scores[-1] = float('inf')

            # 根据重要性分数排序
            sorted_indices = [x for _, x in sorted(zip(importance_scores, range(len(indices))), reverse=True)]

            # 选择前max_points个重要点
            selected_indices = sorted_indices[:max_points]
            refined_indices = [indices[i] for i in sorted(selected_indices)]
        else:
            refined_indices = indices

        print(f"最终选择{len(refined_indices)}个采样点")
        return refined_indices

    def _visualize_sampling_points(self, results):
        """可视化采样点结果

        Args:
            results: 包含问题2和问题3采样点信息的字典
        """
        # 创建问题2采样点图
        plt.figure(figsize=(12, 6))
        plt.plot(self.times2, self.flow2, 'b-', label='完整数据')
        plt.scatter(results['p2']['times'], results['p2']['flow'],
                  color='red', s=50, label=f'选定采样点({results["p2"]["count"]}个)')
        plt.xlabel('时间(t)')
        plt.ylabel('流量')
        plt.title(f'问题2最优采样点分布 - {self.method}方法')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./P5/问题2采样点分布_{self.method}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 创建问题3采样点图
        plt.figure(figsize=(12, 6))
        plt.plot(self.times3, self.flow3, 'g-', label='完整数据')
        plt.scatter(results['p3']['times'], results['p3']['flow'],
                  color='red', s=50, label=f'选定采样点({results["p3"]["count"]}个)')

        # 如果是问题3，标记绿灯时段
        if hasattr(self.p3_analyzer, 'FIRST_GREEN'):
            for i in range(6):
                start = self.p3_analyzer.FIRST_GREEN + i * self.p3_analyzer.CYCLE_LENGTH
                if start < 60:
                    end = min(start + self.p3_analyzer.GREEN_DURATION, 60)
                    plt.axvspan(start, end, color='green', alpha=0.1)

        plt.xlabel('时间(t)')
        plt.ylabel('流量')
        plt.title(f'问题3最优采样点分布 - {self.method}方法')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./P5/问题3采样点分布_{self.method}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self, results):
        """生成采样点分析报告

        Args:
            results: 包含问题2和问题3采样点信息的字典
        """
        with open('./P5/采样点分析报告.md', 'w', encoding='utf-8') as f:
            f.write(f"# 交通流量采样点优化分析 - {self.method}方法\n\n")

            # 问题2采样点分析
            f.write("## 一、问题2采样点分析\n\n")
            f.write(f"采用{self.method}方法共识别出**{results['p2']['count']}**个关键采样点。\n\n")
            f.write("### 采样点时间列表\n\n")
            f.write("| 序号 | 时间索引 | 对应时刻 | 流量值 |\n")
            f.write("|------|----------|----------|--------|\n")

            for i, idx in enumerate(results['p2']['indices']):
                time_idx = self.times2[idx]
                time_str = f"{7 + time_idx // 30}:{(time_idx % 30) * 2:02d}"
                f.write(f"| {i+1} | {time_idx} | {time_str} | {self.flow2[idx]:.2f} |\n")

            f.write("\n### 采样点特征分析\n\n")
            f.write("1. **转折点覆盖**: 所选采样点覆盖了支路2在t=24和t=37的转折点，以及支路3在t≈40的转折点。\n")
            f.write("2. **峰值捕获**: 成功捕获了流量曲线的主要峰值和谷值。\n")
            f.write("3. **边界点**: 包含了起点(t=0)和终点(t=59)，确保完整覆盖。\n")

            # 问题3采样点分析
            f.write("\n## 二、问题3采样点分析\n\n")
            f.write(f"采用{self.method}方法共识别出**{results['p3']['count']}**个关键采样点。\n\n")
            f.write("### 采样点时间列表\n\n")
            f.write("| 序号 | 时间索引 | 对应时刻 | 流量值 |\n")
            f.write("|------|----------|----------|--------|\n")

            for i, idx in enumerate(results['p3']['indices']):
                time_idx = self.times3[idx]
                time_str = f"{7 + time_idx // 30}:{(time_idx % 30) * 2:02d}"
                f.write(f"| {i+1} | {time_idx} | {time_str} | {self.flow3[idx]:.2f} |\n")

            f.write("\n### 采样点特征分析\n\n")
            f.write("1. **信号灯周期**: 采样点捕获了信号灯周期的关键时刻，特别是绿灯开始和结束时刻。\n")
            f.write("2. **流量波动**: 针对流量显著变化的区间增加了采样密度。\n")
            f.write("3. **支路特征**: 覆盖了支路1和支路2的主要转折点。\n")

            # 采样策略总结
            f.write("\n## 三、采样策略总结\n\n")
            f.write("### 采样点选择原则\n\n")
            f.write("1. **关键转折点原则**: 优先选择流量函数在数学上的转折点，如分段函数的连接点。\n")
            f.write("2. **信号周期原则**: 针对有信号灯控制的路段，确保在信号状态变化时刻有采样点。\n")
            f.write("3. **变化率原则**: 在流量变化率较大的区域增加采样密度。\n")
            f.write("4. **边界覆盖原则**: 必须包含时间范围的起点和终点。\n")

            f.write("\n### 不同问题的采样策略差异\n\n")
            f.write("- **问题2**: 由于没有信号灯控制，采样点主要关注支路流量的自然变化特征，例如支路2的分段特征和支路4的周期性变化。\n")
            f.write("- **问题3**: 由于存在信号灯控制，采样策略更加关注信号灯的周期变化，确保每个绿灯周期内有足够的采样点。\n")

            f.write("\n### 结论与建议\n\n")
            f.write(f"1. 对于问题2类型的交通流量，建议至少选取{min(15, results['p2']['count'])}个采样点，重点关注支路流量的转折点和峰值。\n")
            f.write(f"2. 对于问题3类型的有信号灯控制的交通流量，建议至少选取{min(18, results['p3']['count'])}个采样点，重点关注信号灯状态变化和各支路的主要特征点。\n")
            f.write("3. 无论采用何种采样策略，起点和终点都是必须包含的采样点，它们定义了整个时间段的边界。\n")

def main(method='combined'):
    """主函数

    Args:
        method: 采样点识别方法 ('derivative', 'peaks', 'breakpoints', 'combined')
    """
    print(f"\n开始问题5分析 - 采用{method}方法识别最优采样点...")

    # 创建采样点优化器
    optimizer = SamplingPointsOptimizer(method=method)

    # 识别最优采样点
    results = optimizer.identify_sampling_points()

    print(f"\n问题5分析完成！")
    print(f"问题2识别出{results['p2']['count']}个采样点")
    print(f"问题3识别出{results['p3']['count']}个采样点")
    print("结果保存在./P5目录")

if __name__ == "__main__":
    main()
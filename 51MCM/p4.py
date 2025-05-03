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
        """加载并预处理交通流量数据（增强版：多种数据平滑方法结合与异常值检测）"""
        df = pd.read_excel('.\\2025-51MCM-Problem A\附件(Attachment).xlsx',
                           sheet_name='表4 (Table 4)')
        time_points = df['时间 t (Time t)'].values
        flow_data = df['主路4的车流量 (Traffic flow on the Main road 4)'].values

        # 保存原始数据用于对比
        self.raw_flow = flow_data.copy()

        # 步骤1：异常值检测（使用多种方法）
        # 方法1.1：基于相邻点差值的异常值检测
        flow_diff = np.abs(np.diff(flow_data))
        diff_threshold = np.percentile(flow_diff, 90) * 1.5  # 使用90%分位数的1.5倍作为阈值
        is_outlier_diff = np.zeros_like(flow_data, dtype=bool)
        # 优化异常值检测逻辑
        is_outlier_diff[1:-1] = (flow_diff[:-1] > diff_threshold) & (flow_diff[1:] > diff_threshold)

        # 优化数据平滑处理
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        padded_data = np.pad(flow_data, (2, 2), mode='edge')
        smoothed_data = np.convolve(padded_data, weights, mode='valid')

        # 方法1.2：基于移动窗口Z-score的异常值检测
        def rolling_zscore(series, window=5):
            result = np.zeros_like(series, dtype=bool)
            for i in range(len(series)):
                start = max(0, i - window // 2)
                end = min(len(series), i + window // 2 + 1)
                window_data = series[start:end]
                if len(window_data) >= 3:  # 确保有足够的数据计算统计量
                    mean = np.mean(window_data)
                    std = np.std(window_data)
                    if std > 0:  # 避免除以零
                        z_score = abs(series[i] - mean) / std
                        result[i] = z_score > 2.5  # Z-score阈值
            return result

        is_outlier_zscore = rolling_zscore(flow_data)

        # 方法1.3：基于局部密度的异常值检测（使用局部中位数绝对偏差MAD）
        def rolling_mad(series, window=5):
            result = np.zeros_like(series, dtype=bool)
            for i in range(len(series)):
                start = max(0, i - window // 2)
                end = min(len(series), i + window // 2 + 1)
                window_data = series[start:end]
                if len(window_data) >= 3:
                    median = np.median(window_data)
                    mad = np.median(np.abs(window_data - median))
                    if mad > 0:  # 避免除以零
                        score = abs(series[i] - median) / mad
                        result[i] = score > 3.5  # MAD阈值
            return result

        is_outlier_mad = rolling_mad(flow_data)

        # 综合多种异常值检测结果
        is_outlier = is_outlier_diff | is_outlier_zscore | is_outlier_mad

        # 步骤2：数据平滑处理
        # 方法2.1：加权移动平均（中心点权重更高）
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        padded_data = np.pad(flow_data, (2, 2), mode='edge')
        smoothed_data = np.zeros_like(flow_data, dtype=float)

        for i in range(len(flow_data)):
            smoothed_data[i] = np.sum(padded_data[i:i+5] * weights)

        # 方法2.2：自适应加权移动平均（根据数据波动调整窗口大小）
        def adaptive_wma(series, min_window=3, max_window=7):
            result = np.zeros_like(series, dtype=float)
            volatility = np.abs(np.diff(np.pad(series, (1, 1), mode='edge')))

            for i in range(len(series)):
                # 根据局部波动性确定窗口大小
                local_volatility = np.mean(volatility[max(0, i-2):min(len(volatility), i+3)])
                vol_percentile = np.searchsorted(np.sort(volatility), local_volatility) / len(volatility)
                # 波动大用小窗口，波动小用大窗口
                window_size = int(min_window + (max_window - min_window) * (1 - vol_percentile))
                window_size = max(min_window, min(max_window, window_size))
                window_size = window_size + (1 - window_size % 2)  # 确保窗口大小为奇数

                half_window = window_size // 2
                start = max(0, i - half_window)
                end = min(len(series), i + half_window + 1)

                if end - start > 0:
                    # 生成权重，中心点权重最高
                    dist_from_center = np.abs(np.arange(end - start) - (i - start))
                    weights = 1 / (1 + dist_from_center)
                    weights = weights / np.sum(weights)
                    result[i] = np.sum(series[start:end] * weights)
                else:
                    result[i] = series[i]

            return result

        adaptive_smoothed = adaptive_wma(flow_data)

        # 方法2.3：中值滤波（去除异常值）
        from scipy.signal import medfilt
        median_filtered = medfilt(flow_data, kernel_size=5)

        # 方法2.4：指数加权移动平均（对趋势更敏感）
        ewma = pd.Series(flow_data).ewm(span=3, adjust=False).mean().values

        # 方法2.5：Savitzky-Golay滤波（保留峰值特征）
        from scipy.signal import savgol_filter
        try:
            savgol_smoothed = savgol_filter(flow_data, window_length=5, polyorder=2)
        except:
            savgol_smoothed = smoothed_data  # 如果失败，使用普通平滑结果

        # 步骤3：组合多种平滑方法的结果
        # 对于检测到的异常值，使用中值滤波结果
        # 对于其他点，使用多种平滑方法的加权组合
        final_smoothed = np.zeros_like(flow_data, dtype=float)

        # 对异常值使用中值滤波结果
        final_smoothed[is_outlier] = median_filtered[is_outlier]

        # 对非异常值使用多种平滑方法的加权组合
        non_outlier = ~is_outlier
        final_smoothed[non_outlier] = (
            0.3 * smoothed_data[non_outlier] +
            0.3 * adaptive_smoothed[non_outlier] +
            0.2 * ewma[non_outlier] +
            0.2 * savgol_smoothed[non_outlier]
        )

        # 步骤4：保存处理后的数据
        self.dataset = pd.DataFrame({
            '时间点': time_points,
            '时间索引': range(60),
            '主路流量': final_smoothed,
            '原始流量': flow_data,
            '加权平滑': smoothed_data,
            '自适应平滑': adaptive_smoothed,
            '中值滤波': median_filtered,
            '指数平滑': ewma,
            'SG滤波': savgol_smoothed,
            '异常值': is_outlier.astype(int)
        })

        # 保存处理结果
        self.time_idx = self.dataset['时间索引'].values
        self.actual_flow = self.dataset['主路流量'].values

        # 记录异常值信息
        self.outlier_info = {
            'indices': np.where(is_outlier)[0],
            'values': flow_data[is_outlier],
            'corrected': final_smoothed[is_outlier],
            'diff_threshold': diff_threshold,
            'total_count': np.sum(is_outlier)
        }

        print(f"数据预处理完成：检测到{self.outlier_info['total_count']}个异常值并进行了修正")
        if self.outlier_info['total_count'] > 0:
            print(f"异常值位置: {self.outlier_info['indices']}")
            print(f"原始值: {self.outlier_info['values']}")
            print(f"修正值: {self.outlier_info['corrected']}")


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
        """改进的Huber损失函数（自适应参数）"""
        abs_errors = np.abs(errors)
        quadratic = np.minimum(abs_errors, delta)
        linear = abs_errors - quadratic
        # 使用对数项增强对大误差的惩罚，同时保持对小误差的平滑处理
        return 0.5 * quadratic ** 2 + delta * linear + 0.1 * np.log1p(abs_errors)

    def _adaptive_delta(self, errors):
        """自适应计算Huber损失的delta参数"""
        # 根据误差分布动态调整delta值
        # 使用误差的中位数绝对偏差(MAD)作为稳健的尺度估计
        mad = np.median(np.abs(errors - np.median(errors)))
        # 将delta设置为MAD的1.5倍，这是一个常用的异常值检测阈值
        return max(1.0, 1.5 * mad)

    def _evaluate_model(self, params):
        """改进的模型评估函数（动态权重和自适应损失）"""
        predicted = self._calculate_main_flow(self.time_idx, params)
        errors = predicted - self.actual_flow

        # 动态计算Huber损失的delta参数
        adaptive_delta = self._adaptive_delta(errors)

        # 使用动态加权方案，基于数据特征分配权重
        time_weights = np.ones_like(self.time_idx, dtype=float)

        # 基于时间段的权重分配
        # 早高峰(7:20-8:00)权重增加
        time_weights[(self.time_idx >= 10) & (self.time_idx <= 30)] = 1.8
        # 晚高峰(8:00-8:40)权重增加
        time_weights[(self.time_idx >= 30) & (self.time_idx <= 50)] = 1.8

        # 基于数据波动性的权重分配
        # 计算局部波动性（使用滑动窗口的标准差）
        window_size = 5
        padded_flow = np.pad(self.actual_flow, (window_size//2, window_size//2), mode='edge')
        local_std = np.array([np.std(padded_flow[i:i+window_size]) for i in range(len(self.actual_flow))])

        # 归一化局部标准差到[0.8, 1.2]范围，波动大的区域权重较低
        norm_std = 1.2 - 0.4 * (local_std / np.max(local_std))

        # 组合两种权重方案
        final_weights = time_weights * norm_std

        # 计算加权Huber损失
        weighted_errors = [final_weights[i] * self._huber_loss(errors[i], delta=adaptive_delta) for i in range(len(errors))]
        error = np.mean(weighted_errors)

        # 约束条件处理
        penalty = self._compute_penalty(params)

        # 改进的全局平滑性约束
        # 使用二阶差分来惩罚曲线的突变
        predicted_diff1 = np.abs(np.diff(predicted))
        predicted_diff2 = np.abs(np.diff(predicted_diff1))

        # 对一阶和二阶差分都进行惩罚
        smoothness_penalty1 = 40 * np.sum(np.where(predicted_diff1 > 12, predicted_diff1 - 12, 0))
        smoothness_penalty2 = 80 * np.sum(np.where(predicted_diff2 > 8, predicted_diff2 - 8, 0))

        # 保存当前评估的误差和惩罚项（用于调试和分析）
        self._current_error = error
        self._current_penalty = penalty
        self._current_smoothness = smoothness_penalty1 + smoothness_penalty2

        return error + penalty + smoothness_penalty1 + smoothness_penalty2

    def _compute_penalty(self, params):
        """改进的约束惩罚项计算（自适应权重和软硬约束结合）"""
        f1, f2, f3 = self._calculate_branch_flows(self.time_idx, params)
        penalty = 0

        # 1. 非负约束 - 使用指数惩罚增强效果
        neg_penalty = np.sum(np.exp(np.abs(f1[f1 < 0])) - 1) + \
                      np.sum(np.exp(np.abs(f2[f2 < 0])) - 1) + \
                      np.sum(np.exp(np.abs(f3[f3 < 0])) - 1)
        penalty += 1000 * neg_penalty

        # 2. 支路1流量特征约束 - 确保符合"无车流量→线性增长→稳定→线性减少至无车流量"的趋势
        a1, a2, a3, a4, a5, a6, brk1, brk2, brk3, brk4 = params[:10]

        # 2.1 斜率约束 - 使用平方惩罚使约束更平滑
        # 确保a1为正（线性增长）
        penalty += 1200 * max(0, -a1)**2
        # 确保a3为负（线性减少）
        penalty += 1200 * max(0, a3)**2
        # 确保a6为负（线性减少至无车流量）
        penalty += 1200 * max(0, a6)**2

        # 2.2 支路1平滑性约束 - 确保各段之间的过渡平滑
        # 计算支路1的一阶和二阶差分
        f1_diff1 = np.abs(np.diff(f1))
        f1_diff2 = np.abs(np.diff(f1, n=2))
        # 对较大的变化施加惩罚
        penalty += 100 * np.sum(np.where(f1_diff1 > 5, f1_diff1 - 5, 0))
        penalty += 200 * np.sum(f1_diff2)

        # 3. 支路2流量特征约束
        b1, b2, b3, b4, b5 = params[10:15]

        # 3.1 斜率约束
        # 确保b1为正（线性增长）
        penalty += 1200 * max(0, -b1)**2
        # 确保b4为负（线性减少）
        penalty += 1200 * max(0, b4)**2

        # 3.2 支路2平滑性约束
        # 计算支路2的一阶和二阶差分
        f2_diff1 = np.abs(np.diff(f2))
        f2_diff2 = np.abs(np.diff(f2, n=2))
        # 对较大的变化施加惩罚
        penalty += 100 * np.sum(np.where(f2_diff1 > 5, f2_diff1 - 5, 0))
        penalty += 250 * np.sum(f2_diff2)

        # 4. 支路3流量特征约束 - 确保在绿灯时段有合理的流量变化
        c_params = params[15:-1]

        # 4.1 参数约束
        for i in range(7):
            slope, intercept = c_params[2*i], c_params[2*i+1]
            # 确保截距非负
            penalty += 600 * max(0, -intercept)**2
            # 限制斜率变化范围
            penalty += 150 * max(0, abs(slope) - 2.5)**2

        # 4.2 支路3平滑性约束 - 确保相邻绿灯周期的流量变化平滑
        # 计算相邻绿灯周期开始和结束时的流量差异
        cycle_starts = [self.FIRST_GREEN + i * self.CYCLE_LENGTH for i in range(7)]
        cycle_ends = [start + self.GREEN_DURATION for start in cycle_starts]

        # 计算每个绿灯周期开始和结束时的流量
        cycle_start_flows = []
        cycle_end_flows = []

        for i in range(7):
            slope, intercept = c_params[2*i], c_params[2*i+1]
            start_flow = intercept  # t=start时的流量
            end_flow = slope * self.GREEN_DURATION + intercept  # t=end时的流量
            cycle_start_flows.append(start_flow)
            cycle_end_flows.append(end_flow)

        # 计算相邻周期之间的流量差异
        for i in range(6):
            # 当前周期结束与下一周期开始的流量差异
            flow_diff = abs(cycle_end_flows[i] - cycle_start_flows[i+1])
            penalty += 300 * flow_diff

        # 5. 连续性约束 - 使用平方惩罚使约束更平滑
        continuity_errors = [
            (a1 * (brk2 - brk1) + a2 - a4)**2,  # 支路1第一段与第二段的连续性
            (a3 * (brk3 - brk2) + a4 - a5)**2,  # 支路1第二段与第三段的连续性
            (a5 - a6 * 0)**2,                   # 支路1第三段与第四段的连续性
            (b1 * 17 + b2 - b3)**2,             # 支路2第一段与第二段的连续性
            (b3 - b4 * 0 - b5)**2               # 支路2第二段与第三段的连续性
        ]
        penalty += 1800 * sum(continuity_errors)

        # 6. 转折点顺序约束 - 使用指数惩罚增强效果
        order_errors = [
            np.exp(max(0, brk1 - brk2 + 0.5)) - 1,  # 确保 brk1 < brk2
            np.exp(max(0, brk2 - brk3 + 0.5)) - 1,  # 确保 brk2 < brk3
            np.exp(max(0, brk3 - brk4 + 0.5)) - 1   # 确保 brk3 < brk4
        ]
        penalty += 1500 * sum(order_errors)

        # 7. 物理合理性约束 - 确保流量在合理范围内
        # 支路1的最大流量不应过大
        max_f1 = np.max(f1)
        penalty += 500 * max(0, max_f1 - 50)**2

        # 支路2的最大流量不应过大
        max_f2 = np.max(f2)
        penalty += 500 * max(0, max_f2 - 50)**2

        # 支路3的最大流量不应过大
        max_f3 = np.max(f3)
        penalty += 500 * max(0, max_f3 - 40)**2

        # 保存各部分惩罚值用于调试
        self._penalty_components = {
            'neg_penalty': neg_penalty,
            'continuity': sum(continuity_errors),
            'order': sum(order_errors),
            'max_flow': max(0, max_f1 - 50)**2 + max(0, max_f2 - 50)**2 + max(0, max_f3 - 40)**2
        }

        return penalty

    def optimize_model(self):
        """改进的模型参数优化（多阶段优化策略）"""
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

        # 创建多个初始猜测点
        num_initial_points = 8  # 增加初始点数量
        initial_points = []

        # 第一个点使用原始初始猜测
        initial_points.append(np.array(initial_guess))

        # 生成多个随机初始点
        for i in range(1, num_initial_points):
            # 使用不同的随机扰动策略
            if i % 3 == 0:
                # 大幅扰动
                perturbed = np.array(initial_guess) * (1 + 0.3 * np.random.randn(len(initial_guess)))
            elif i % 3 == 1:
                # 中等扰动
                perturbed = np.array(initial_guess) * (1 + 0.15 * np.random.randn(len(initial_guess)))
            else:
                # 小幅扰动 + 偏移
                perturbed = np.array(initial_guess) * (1 + 0.08 * np.random.randn(len(initial_guess)))
                # 对FIRST_GREEN参数进行特殊处理，在整个范围内均匀采样
                perturbed[-1] = np.random.uniform(0.0, 8.0)

            # 确保参数在边界内
            for j, (lb, ub) in enumerate(param_bounds):
                perturbed[j] = max(lb, min(ub, perturbed[j]))

            initial_points.append(perturbed)

        # 多阶段优化策略
        best_error = float('inf')
        best_params = None
        optimization_history = []

        print("开始第一阶段优化：粗略搜索最优区域...")
        # 第一阶段：使用多个初始点进行粗略搜索
        for idx, init_point in enumerate(initial_points):
            print(f"  优化初始点 {idx+1}/{num_initial_points}")

            # 使用不同的优化方法
            if idx % 3 == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 500, 'gtol': 1e-5}
            elif idx % 3 == 1:
                method = 'SLSQP'
                options = {'maxiter': 500, 'ftol': 1e-5}
            else:
                method = 'TNC'
                options = {'maxiter': 500}

            # 执行优化
            try:
                result = minimize(
                    self._evaluate_model,
                    init_point,
                    method=method,
                    bounds=param_bounds,
                    options=options
                )

                # 记录结果
                optimization_history.append({
                    'params': result.x,
                    'error': result.fun,
                    'success': result.success,
                    'method': method
                })

                # 更新最佳结果
                if result.fun < best_error:
                    best_error = result.fun
                    best_params = result.x
                    print(f"    发现更优解，误差: {best_error:.6f}")
            except Exception as e:
                print(f"    优化失败: {str(e)}")

        print("\n开始第二阶段优化：细化最优解...")
        # 第二阶段：从前三个最佳结果开始进行更精细的优化
        # 按误差排序结果
        sorted_results = sorted(optimization_history, key=lambda x: x['error'])
        top_k = min(3, len(sorted_results))  # 取前3个或更少

        for i in range(top_k):
            result = sorted_results[i]
            print(f"  细化优化 {i+1}/{top_k}, 起始误差: {result['error']:.6f}")

            # 使用更精细的优化设置
            try:
                refined_result = minimize(
                    self._evaluate_model,
                    result['params'],
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': 1000, 'gtol': 1e-7}
                )

                # 更新最佳结果
                if refined_result.fun < best_error:
                    best_error = refined_result.fun
                    best_params = refined_result.x
                    print(f"    优化成功，新误差: {best_error:.6f}")
            except Exception as e:
                print(f"    细化优化失败: {str(e)}")

        print("\n开始第三阶段优化：最终精细调整...")
        # 第三阶段：最终精细调整
        try:
            final_result = minimize(
                self._evaluate_model,
                best_params,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': 2000, 'gtol': 1e-8}  # 增加迭代次数和精度
            )

            if final_result.fun < best_error:
                best_error = final_result.fun
                best_params = final_result.x
                print(f"  最终优化成功，最终误差: {best_error:.6f}")
        except Exception as e:
            print(f"  最终优化失败: {str(e)}")

        # 保存最优参数
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
            print(f"时间段 {start}-{end} RMSE误差: {self.segment_errors[i]:.4f}")
        print(f"第一个绿灯开始时间: {self.optimal_params[-1]:.4f}")

        # 保存优化历史记录
        self.optimization_history = optimization_history

    def visualize_results(self):
        """增强的可视化分析结果"""
        # 创建输出目录
        import os
        os.makedirs('./P4', exist_ok=True)

        # 绘制主要图表
        self._plot_main_comparison()
        self._plot_branch_flows()
        self._plot_error_analysis()
        self._plot_data_preprocessing()
        self._plot_signal_timing()
        self._plot_flow_distribution()

    def _plot_main_comparison(self):
        """绘制主路流量对比图（增强版）"""
        plt.figure(figsize=(16, 8))

        # 主图：实测与预测流量对比
        plt.subplot(2, 1, 1)
        plt.plot(self.time_idx, self.actual_flow, 'b-', lw=2, label='实测流量(平滑后)')
        plt.plot(self.time_idx, self.predicted_flow, 'r--', lw=2, label='预测流量')

        # 添加时间标记
        time_labels = ['7:00', '7:30', '8:00', '8:30', '8:58']
        time_ticks = [0, 15, 30, 45, 59]
        plt.xticks(time_ticks, time_labels)

        # 标记绿灯时段
        for i in range(7):
            start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
            plt.axvspan(start, start + self.GREEN_DURATION, color='green', alpha=0.1)

        plt.xlabel('时间')
        plt.ylabel('车流量')
        plt.title('主路流量实测与预测对比')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # 子图：误差分析
        plt.subplot(2, 1, 2)
        errors = self.predicted_flow - self.actual_flow
        plt.bar(self.time_idx, errors, color='gray', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # 添加误差统计信息
        plt.text(0.02, 0.95, f'RMSE: {self.model_error:.4f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        # 标记时间段误差
        time_segments = [(0, 15), (15, 30), (30, 45), (45, 59)]
        segment_colors = ['#8ecfc9', '#ffbe7a', '#fa7f6f', '#82b0d2']
        for i, ((start, end), color) in enumerate(zip(time_segments, segment_colors)):
            plt.axvspan(start, end, color=color, alpha=0.2)
            plt.text((start + end) / 2, max(errors) * 0.8, f'RMSE: {self.segment_errors[i]:.2f}',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('预测误差')
        plt.title('预测误差分布')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./P4/主路流量实测与预测对比4.png', dpi=300, bbox_inches='tight')

    def _plot_branch_flows(self):
        """绘制支路流量图（增强版）"""
        f1, f2, f3 = self.branch_flows
        a_params = self.optimal_params[:10]
        b_params = self.optimal_params[10:15]
        c_params = self.optimal_params[15:-1]

        # 创建一个2x2的子图布局
        fig = plt.figure(figsize=(18, 12))

        # 子图1：所有支路流量
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(self.time_idx, f1, 'g-', lw=2, label='支路1')
        ax1.plot(self.time_idx, f2, 'm-', lw=2, label='支路2')
        ax1.plot(self.time_idx, f3, 'c-', lw=2, label='支路3')
        ax1.plot(self.time_idx, self.predicted_flow, 'r--', lw=1.5, label='主路4(预测)')

        # 标记绿灯时段
        for i in range(7):
            start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
            ax1.axvspan(start, start + self.GREEN_DURATION, color='green', alpha=0.1)

        # 添加时间标记
        time_labels = ['7:00', '7:30', '8:00', '8:30', '8:58']
        time_ticks = [0, 15, 30, 45, 59]
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels(time_labels)

        ax1.set_xlabel('时间')
        ax1.set_ylabel('车流量')
        ax1.set_title('各支路流量变化情况')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # 子图2：支路1流量及其分段
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(self.time_idx, f1, 'g-', lw=2)

        # 标记支路1的各个阶段
        brk1, brk2, brk3, brk4 = a_params[6:10]
        ax2.axvline(x=brk1, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=brk2, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=brk3, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=brk4, color='k', linestyle='--', alpha=0.5)

        # 添加阶段标签
        ax2.text(brk1/2, max(f1)/2, '无车流量', ha='center')
        ax2.text((brk1+brk2)/2, max(f1)/2, '线性增长', ha='center')
        ax2.text((brk2+brk3)/2, max(f1)/2, '线性减少', ha='center')
        ax2.text((brk3+brk4)/2, max(f1)/2, '稳定', ha='center')
        ax2.text((brk4+59)/2, max(f1)/2, '线性减少', ha='center')

        ax2.set_xlabel('时间点(每2分钟)')
        ax2.set_ylabel('车流量')
        ax2.set_title('支路1流量变化及分段')
        ax2.grid(True, alpha=0.3)

        # 子图3：支路2流量及其分段
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(self.time_idx, f2, 'm-', lw=2)

        # 标记支路2的各个阶段
        brk5, brk6 = 17, 47  # 支路2的转折点
        ax3.axvline(x=brk5, color='k', linestyle='--', alpha=0.5)
        ax3.axvline(x=brk6, color='k', linestyle='--', alpha=0.5)

        # 添加阶段标签
        ax3.text(brk5/2, max(f2)/2, '线性增长', ha='center')
        ax3.text((brk5+brk6)/2, max(f2)/2, '稳定', ha='center')
        ax3.text((brk6+59)/2, max(f2)/2, '线性减少', ha='center')

        ax3.set_xlabel('时间点(每2分钟)')
        ax3.set_ylabel('车流量')
        ax3.set_title('支路2流量变化及分段')
        ax3.grid(True, alpha=0.3)

        # 子图4：支路3流量及绿灯周期
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(self.time_idx, f3, 'c-', lw=2)

        # 标记绿灯时段
        for i in range(7):
            start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
            ax4.axvspan(start, start + self.GREEN_DURATION, color='green', alpha=0.2)
            # 添加周期标签
            ax4.text(start + self.GREEN_DURATION/2, max(f3)*0.8, f'周期{i+1}',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        ax4.set_xlabel('时间点(每2分钟)')
        ax4.set_ylabel('车流量')
        ax4.set_title('支路3流量变化及绿灯周期')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./P4/支路车流量变化4.png', dpi=300, bbox_inches='tight')

    def _plot_error_analysis(self):
        """绘制误差分析图"""
        plt.figure(figsize=(16, 12))

        # 子图1：误差直方图
        plt.subplot(2, 2, 1)
        errors = self.predicted_flow - self.actual_flow
        plt.hist(errors, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.axvline(x=np.mean(errors), color='g', linestyle='-', label=f'平均误差: {np.mean(errors):.2f}')
        plt.xlabel('预测误差')
        plt.ylabel('频数')
        plt.title('预测误差分布直方图')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：误差散点图
        plt.subplot(2, 2, 2)
        plt.scatter(self.actual_flow, self.predicted_flow, alpha=0.7, c='blue')
        # 添加对角线（完美预测线）
        min_val = min(min(self.actual_flow), min(self.predicted_flow))
        max_val = max(max(self.actual_flow), max(self.predicted_flow))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('实际流量')
        plt.ylabel('预测流量')
        plt.title('实际流量 vs 预测流量散点图')
        plt.grid(True, alpha=0.3)

        # 子图3：误差随时间变化
        plt.subplot(2, 2, 3)
        plt.plot(self.time_idx, np.abs(errors), 'o-', color='purple', alpha=0.7)

        # 标记时间段
        time_segments = [(0, 15), (15, 30), (30, 45), (45, 59)]
        segment_labels = ['7:00-7:30', '7:30-8:00', '8:00-8:30', '8:30-8:58']
        segment_colors = ['#8ecfc9', '#ffbe7a', '#fa7f6f', '#82b0d2']

        for (start, end), label, color in zip(time_segments, segment_labels, segment_colors):
            plt.axvspan(start, end, color=color, alpha=0.2)
            plt.text((start + end) / 2, max(np.abs(errors)) * 0.9, label,
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('绝对误差')
        plt.title('绝对误差随时间变化')
        plt.grid(True, alpha=0.3)

        # 子图4：各时间段误差箱线图
        plt.subplot(2, 2, 4)
        error_segments = []
        for start, end in time_segments:
            segment_idx = (self.time_idx >= start) & (self.time_idx <= end)
            error_segments.append(errors[segment_idx])

        plt.boxplot(error_segments, labels=segment_labels)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('时间段')
        plt.ylabel('预测误差')
        plt.title('各时间段误差箱线图')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./P4/误差分析4.png', dpi=300, bbox_inches='tight')

    def _plot_data_preprocessing(self):
        """绘制数据预处理效果图"""
        plt.figure(figsize=(16, 8))

        # 子图1：原始数据与平滑后数据对比
        plt.subplot(2, 1, 1)
        plt.plot(self.time_idx, self.raw_flow, 'b-', lw=1.5, alpha=0.7, label='原始数据')
        plt.plot(self.time_idx, self.actual_flow, 'r-', lw=2, label='平滑后数据')

        # 添加时间标记
        time_labels = ['7:00', '7:30', '8:00', '8:30', '8:58']
        time_ticks = [0, 15, 30, 45, 59]
        plt.xticks(time_ticks, time_labels)

        plt.xlabel('时间')
        plt.ylabel('车流量')
        plt.title('原始数据与平滑后数据对比')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 子图2：不同平滑方法对比
        plt.subplot(2, 1, 2)
        plt.plot(self.time_idx, self.raw_flow, 'k-', lw=1, alpha=0.5, label='原始数据')
        plt.plot(self.time_idx, self.dataset['加权平滑'].values, 'g-', lw=1.5, label='加权移动平均')
        plt.plot(self.time_idx, self.dataset['中值滤波'].values, 'b-', lw=1.5, label='中值滤波')
        plt.plot(self.time_idx, self.dataset['指数平滑'].values, 'm-', lw=1.5, label='指数加权平均')
        plt.plot(self.time_idx, self.actual_flow, 'r-', lw=2, label='组合平滑(最终)')

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('不同平滑方法对比')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig('./P4/数据预处理效果4.png', dpi=300, bbox_inches='tight')

    def _plot_signal_timing(self):
        """绘制信号灯时序与流量关系图"""
        plt.figure(figsize=(16, 8))

        # 获取信号灯状态序列
        signal_states = self._get_signal_states(self.time_idx)
        f1, f2, f3 = self.branch_flows

        # 绘制信号灯状态和支路3流量
        plt.subplot(2, 1, 1)
        # 绘制信号灯状态（用颜色带表示）
        for i in range(len(self.time_idx)):
            if signal_states[i]:
                plt.axvspan(self.time_idx[i]-0.4, self.time_idx[i]+0.4, color='green', alpha=0.3)
            else:
                plt.axvspan(self.time_idx[i]-0.4, self.time_idx[i]+0.4, color='red', alpha=0.2)

        # 绘制支路3流量
        plt.plot(self.time_idx, f3, 'b-', lw=2, label='支路3流量')

        # 添加第一个绿灯开始时间标记
        plt.axvline(x=self.FIRST_GREEN, color='g', linestyle='--', lw=2,
                    label=f'第一个绿灯开始时间: {self.FIRST_GREEN:.2f}')

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('信号灯状态与支路3流量关系')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 绘制所有支路流量与信号灯的关系
        plt.subplot(2, 1, 2)
        plt.plot(self.time_idx, f1, 'g-', lw=1.5, alpha=0.7, label='支路1')
        plt.plot(self.time_idx, f2, 'm-', lw=1.5, alpha=0.7, label='支路2')
        plt.plot(self.time_idx, f3, 'c-', lw=2, label='支路3')
        plt.plot(self.time_idx, self.predicted_flow, 'r--', lw=1.5, label='主路4(预测)')

        # 标记绿灯时段
        for i in range(7):
            start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
            plt.axvspan(start, start + self.GREEN_DURATION, color='green', alpha=0.1)
            plt.text(start + self.GREEN_DURATION/2, max(self.predicted_flow)*0.9, f'绿灯{i+1}',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('各支路流量与信号灯周期关系')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig('./P4/信号灯时序分析4.png', dpi=300, bbox_inches='tight')

    def _plot_flow_distribution(self):
        """绘制流量分布与贡献分析图"""
        plt.figure(figsize=(16, 10))

        f1, f2, f3 = self.branch_flows

        # 子图1：各支路流量贡献堆叠图
        plt.subplot(2, 2, 1)
        plt.stackplot(self.time_idx, f1, f2, f3,
                      labels=['支路1', '支路2', '支路3'],
                      colors=['#8ecfc9', '#ffbe7a', '#fa7f6f'],
                      alpha=0.7)
        plt.plot(self.time_idx, self.predicted_flow, 'k--', lw=1.5, label='主路4(预测)')

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('车流量')
        plt.title('各支路流量贡献堆叠图')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # 子图2：各支路流量贡献百分比
        plt.subplot(2, 2, 2)
        total_flow = f1 + f2 + f3
        # 避免除以零
        total_flow = np.where(total_flow == 0, 1e-10, total_flow)

        plt.stackplot(self.time_idx,
                      100 * f1 / total_flow,
                      100 * f2 / total_flow,
                      100 * f3 / total_flow,
                      labels=['支路1', '支路2', '支路3'],
                      colors=['#8ecfc9', '#ffbe7a', '#fa7f6f'],
                      alpha=0.7)

        plt.xlabel('时间点(每2分钟)')
        plt.ylabel('贡献百分比 (%)')
        plt.title('各支路流量贡献百分比')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # 子图3：关键时间点流量分布饼图 (7:30)
        plt.subplot(2, 2, 3)
        t1 = 15  # 7:30对应的时间索引
        f1_t1, f2_t1, f3_t1 = self._calculate_branch_flows(np.array([t1]), self.optimal_params)

        labels = ['支路1', '支路2', '支路3']
        sizes = [f1_t1[0], f2_t1[0], f3_t1[0]]
        colors = ['#8ecfc9', '#ffbe7a', '#fa7f6f']

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=False, wedgeprops={'alpha': 0.7})
        plt.axis('equal')  # 保证饼图是圆形
        plt.title('7:30时刻各支路流量分布')

        # 子图4：关键时间点流量分布饼图 (8:30)
        plt.subplot(2, 2, 4)
        t2 = 45  # 8:30对应的时间索引
        f1_t2, f2_t2, f3_t2 = self._calculate_branch_flows(np.array([t2]), self.optimal_params)

        sizes = [f1_t2[0], f2_t2[0], f3_t2[0]]

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=False, wedgeprops={'alpha': 0.7})
        plt.axis('equal')  # 保证饼图是圆形
        plt.title('8:30时刻各支路流量分布')

        plt.tight_layout()
        plt.savefig('./P4/流量分布分析4.png', dpi=300, bbox_inches='tight')

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
            f.write("支路3在绿灯时段流量为0，在绿灯时段呈现线性变化\n")
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


def main():
    """主函数"""
    # 创建分析系统实例
    analyzer = TrafficAnalysisSystem()

    # 优化模型参数
    analyzer.optimize_model()

    # 可视化结果
    analyzer.visualize_results()

    # 生成分析报告
    analyzer.generate_report()

    print("问题4分析完成！结果保存在./P4目录")

if __name__ == "__main__":
    main()
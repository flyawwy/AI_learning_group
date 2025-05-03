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
        # 根据题目要求的固定参数
        # 红灯时间为8分钟，绿灯时间为10分钟
        self.RED_DURATION = 4      # 对应8分钟（每个时间单位为2分钟）
        self.GREEN_DURATION = 5    # 对应10分钟（每个时间单位为2分钟）
        self.CYCLE_LENGTH = self.RED_DURATION + self.GREEN_DURATION  # 总周期长度为9个时间单位(18分钟)
        self.FIRST_GREEN = 3       # 第一个绿灯于7:06开始亮起，对应时间索引3
        self.TRAVEL_DELAY = 1      # 支路1和支路2的行驶时间为2分钟，对应延迟1个时间单位
        self.NUM_CYCLES = 6        # 总共考虑6个绿灯周期，确保覆盖整个时间范围

        # 添加关键时间点
        self.KEY_TIMES = {
            'start': 0,     # 7:00
            't1': 15,       # 7:30
            't2': 45,       # 8:30
            'end': 59       # 8:58
        }

        # 根据题目明确描述，支路2在特定时间段内有变化
        # 时间点修正为题目准确的时间
        self.ROAD2_LINEAR_GROWTH_END = 35  # 8:10对应时间索引35
        self.ROAD2_STABLE_END = 47         # 8:34对应时间索引47

        # 计算各绿灯周期的开始时间和结束时间
        # 题目明确指出：第一个绿灯于7:06时开始亮起，红灯8分钟，绿灯10分钟
        # 支路3的车流只在绿灯时段存在，红灯时车流为0
        self.GREEN_STARTS = [self.FIRST_GREEN + i * self.CYCLE_LENGTH for i in range(self.NUM_CYCLES)]
        self.GREEN_ENDS = [start + self.GREEN_DURATION for start in self.GREEN_STARTS]

        # 根据计算，绿灯周期为：
        # 第一个周期：7:06-7:16 (时间索引3-8)
        # 第二个周期：7:24-7:34 (时间索引12-17)
        # 第三个周期：7:42-7:52 (时间索引21-26)
        # 第四个周期：8:00-8:10 (时间索引30-35)
        # 第五个周期：8:18-8:28 (时间索引39-44)
        # 第六个周期：8:36-8:46 (时间索引48-53)

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
        b_params = params[10:15]  # 支路2参数
        c_params = params[15:]    # 支路3参数 - 现在可以有12个参数(6个周期)

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

        # 根据题目描述：支路1的车流量呈现"无车流量→增长→减少→稳定→减少至无车流量"的趋势

        # 第一阶段 - 无车流量
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

        # 第五阶段 - 线性减少至无车流量
        mask = t >= brk4
        # 计算在brk4到达终点的斜率，确保在终点(59)处车流量约为0
        end_distance = max(1, 59 - brk4)  # 避免除以零
        decay_rate = a5 / end_distance     # 计算线性衰减率
        result[mask] = a5 - decay_rate * (t[mask] - brk4)

        # 将过小的值设为0
        result[result < 0.1] = 0

        return result

    def _compute_flow2(self, t, params):
        """计算支路2流量 - 三段式模型

        Args:
            t: 时间点索引数组
            params: 支路2参数 [b1,b2,b3,b4,b5] - 移除了不需要优化的转折点参数

        Returns:
            np.array: 支路2流量数组
        """
        b1, b2, b3, b4, b5 = params
        result = np.zeros_like(t, dtype=float)

        # 根据题目描述：支路2在[6:58,8:10]和[8:34,8:58]时间段内线性增长和线性减少
        # 在(8:10,8:34)时间段内稳定，使用固定的时间点参数

        # 第一阶段 - 线性增长 [6:58,8:10]
        mask = t <= self.ROAD2_LINEAR_GROWTH_END
        result[mask] = b1 * t[mask] + b2

        # 计算转折点处的连接值，确保连续性
        value_at_growth_end = b1 * self.ROAD2_LINEAR_GROWTH_END + b2

        # 第二阶段 - 稳定 (8:10,8:34)
        mask = (t > self.ROAD2_LINEAR_GROWTH_END) & (t <= self.ROAD2_STABLE_END)
        # 使用增长结束时的值作为稳定值，确保连续性
        result[mask] = value_at_growth_end

        # 第三阶段 - 线性减少 [8:34,8:58]
        mask = t > self.ROAD2_STABLE_END
        # 使用b4作为减少斜率，从稳定值开始减少
        result[mask] = b4 * (t[mask] - self.ROAD2_STABLE_END) + value_at_growth_end

        # 确保流量非负
        return np.maximum(result, 0)

    def _compute_flow3(self, t, params):
        """计算支路3流量 - 仅在绿灯时段有流量

        Args:
            t: 时间点索引数组
            params: 支路3参数，每个绿灯周期的(斜率,截距)参数对

        Returns:
            np.array: 支路3流量数组
        """
        # 获取信号灯状态 - 使用预先定义的信号灯周期计算
        # 题目明确指出：当信号灯C显示红灯时，支路3的车流量视为0
        # 当C显示绿灯时，支路3的车流量可能稳定或呈现线性变化趋势
        signal_states = self._get_signal_states(t)
        result = np.zeros_like(t, dtype=float)

        # 将参数分组为每个绿灯周期的斜率和截距
        # 题目中明确了信号灯周期是固定的：
        # - 红灯8分钟(4个单位)，绿灯10分钟(5个单位)
        # - 第一个绿灯周期从7:06开始(时间索引3)
        # - 每个周期长度为18分钟(9个时间单位)
        # 因此绿灯周期的开始时间分别为时间索引：3, 12, 21, 30, 39, 48
        cycle_params = [(params[2*i], params[2*i+1]) for i in range(min(len(params)//2, self.NUM_CYCLES))]

        # 确保每个周期都有参数，不足的用默认值补充
        while len(cycle_params) < self.NUM_CYCLES:
            # 如果周期参数不足，使用最后一个周期的略微降低的值
            last_slope, last_intercept = cycle_params[-1]
            cycle_params.append((last_slope * 0.9, last_intercept * 0.9))

        # 对每个绿灯周期分别计算流量
        for idx, (start, end) in enumerate(zip(self.GREEN_STARTS, self.GREEN_ENDS)):
            # 确保不超出时间范围
            if start >= len(t) or idx >= len(cycle_params):
                continue

            # 只在绿灯时段内有流量，红灯时段流量为0
            # 我们还需要检查signal_states以确保仅在绿灯时段计算流量
            mask = (t >= start) & (t < end) & signal_states

            if np.any(mask):
                slope, intercept = cycle_params[idx]
                # 线性模型：斜率*(时间-周期开始时间)+截距
                # 为每个绿灯周期使用独立的参数对
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

        # 参数解包
        a_params = params[:10]
        b_params = params[10:15]
        c_params = params[15:]  # 现在包含12个参数，6个绿灯周期

        a1, a2, a3, a4, a5, a6, brk1, brk2, brk3, brk4 = a_params
        b1, b2, b3, b4, b5 = b_params

        # 1. 非负约束 - 确保所有流量为非负值
        negative_flow_penalty = 8000 * (
            np.sum(np.abs(f1[f1 < 0])) +
            np.sum(np.abs(f2[f2 < 0])) +
            np.sum(np.abs(f3[f3 < 0]))
        )
        penalty += negative_flow_penalty

        # 2. 连续性约束 - 确保各段函数在转折点处连续
        # 支路1连续性
        flow1_at_brk2 = a1 * (brk2 - brk1) + a2
        flow1_at_brk3 = a3 * (brk3 - brk2) + a4

        # 支路2连续性
        flow2_at_brk5 = b1 * self.ROAD2_LINEAR_GROWTH_END + b2
        flow2_at_brk6 = flow2_at_brk5 # 使用计算值而非参数

        continuity_errors = [
            # 支路1的连续性约束
            abs(flow1_at_brk2 - (a3 * 0 + a4)),  # 第一段末尾与第二段开始的连续性
            abs(flow1_at_brk3 - a5),  # 第二段末尾与第三段的连续性
            abs(a5 - (a5 - a5/(max(1, 59-brk4)) * 0)),  # 第三段与第四段的连续性

            # 支路2的连续性约束
            abs(flow2_at_brk5 - b3),  # 第一段与第二段的连续性
            abs(flow2_at_brk6 - b5)  # 第二段与第三段的连续性
        ]
        penalty += 5000 * sum(continuity_errors)

        # 3. 转折点顺序约束 - 确保转折点按时间顺序排列
        order_errors = [
            max(0, brk1 - brk2) * 30,  # 确保 brk1 < brk2
            max(0, brk2 - brk3) * 30,  # 确保 brk2 < brk3
            max(0, brk3 - brk4) * 30,  # 确保 brk3 < brk4
            max(0, self.ROAD2_LINEAR_GROWTH_END - self.ROAD2_STABLE_END) * 30   # 确保 brk5 < brk6
        ]
        penalty += 8000 * sum(order_errors)

        # 4. 支路1特定时间点约束 - 根据题目"无车流量→增长→减少→稳定→减少至无车流量"
        # 确保在开始和结束时流量接近于0
        start_end_errors = [
            np.sum(f1[:int(brk1)]),  # 开始时为0
            np.sum(f1[int(58):])  # 结束时接近于0
        ]
        penalty += 1000 * sum(start_end_errors)

        # 5. 支路2特定时间段约束 - 根据题目描述
        # 在(8:10,8:34)时间段内稳定
        stable_variation = np.std(f2[(self.time_idx > self.ROAD2_LINEAR_GROWTH_END) & (self.time_idx <= self.ROAD2_STABLE_END)])
        penalty += 2000 * stable_variation

        # 确保支路2的线性增长和线性减少特性
        growth_linearity = np.std(np.diff(f2[self.time_idx <= self.ROAD2_LINEAR_GROWTH_END]))
        decrease_linearity = np.std(np.diff(f2[self.time_idx > self.ROAD2_STABLE_END]))
        penalty += 1000 * (growth_linearity + decrease_linearity)

        # 6. 支路3绿灯周期流量约束
        cycle_params = [(c_params[2*i], c_params[2*i+1]) for i in range(5)]
        # 添加第六个周期参数
        if len(c_params) >= 12:
            cycle_params.append((c_params[10], c_params[11]))

        # 6.1 确保绿灯时有流量，红灯时无流量
        red_light_flow = np.sum(f3[~self._get_signal_states(self.time_idx)])
        penalty += 10000 * red_light_flow

        # 6.2 相邻绿灯周期间流量变化平滑
        # 现在处理6个周期
        for i in range(len(cycle_params) - 1):
            slope1, intercept1 = cycle_params[i]
            slope2, intercept2 = cycle_params[i+1]

            # 惩罚斜率和截距的剧烈变化
            penalty += 400 * (abs(slope1 - slope2) + 0.5 * abs(intercept1 - intercept2))

            # 如果相邻周期流量变化方向相反，额外惩罚
            if slope1 * slope2 < 0 and abs(slope1) > 0.1 and abs(slope2) > 0.1:
                penalty += 1000

        # 7. 确保总流量与实测数据的模式一致
        predicted_flow = self._calculate_main_flow(self.time_idx, params)
        trend_mismatch = 0

        # 计算相邻点的趋势一致性
        for i in range(1, len(self.time_idx)):
            pred_direction = np.sign(predicted_flow[i] - predicted_flow[i-1])
            actual_direction = np.sign(self.actual_flow[i] - self.actual_flow[i-1])

            # 如果趋势不一致且变化明显，添加惩罚
            if pred_direction * actual_direction < 0:
                change_magnitude = abs(self.actual_flow[i] - self.actual_flow[i-1])
                if change_magnitude > 5:  # 仅惩罚明显的趋势不一致
                    trend_mismatch += change_magnitude

        penalty += 300 * trend_mismatch

        # 8. 惩罚预测流量与实际流量的峰值差异
        # 找出实际流量的峰值位置
        peaks = []
        for i in range(1, len(self.time_idx)-1):
            if (self.actual_flow[i] > self.actual_flow[i-1] and
                self.actual_flow[i] > self.actual_flow[i+1] and
                self.actual_flow[i] > 90):  # 只考虑主要峰值
                peaks.append(i)

        # 在峰值位置惩罚预测误差，加大权重
        peak_error = 0
        for peak in peaks:
            peak_error += abs(predicted_flow[peak] - self.actual_flow[peak])

        penalty += 200 * peak_error

        # 新增：识别和惩罚关键时间点的预测误差
        key_points = [15, 22, 32, 45]  # 7:30, 7:44, 8:04, 8:30的索引
        key_point_error = 0
        for point in key_points:
            key_point_error += abs(predicted_flow[point] - self.actual_flow[point])

        penalty += 150 * key_point_error

        # 9. 流量平滑度约束 - 惩罚预测流量的过度波动
        predicted_gradient = np.diff(predicted_flow)
        gradient_penalty = np.sum(np.abs(predicted_gradient[abs(predicted_gradient) > 15])) - 15 * np.sum(abs(predicted_gradient) > 15)
        penalty += 80 * gradient_penalty

        # 10. 特定时间区间约束 - 在红绿灯转换处额外惩罚流量剧烈变化
        signal_changes = []
        prev_state = self._check_signal_state(0)
        for i in range(1, len(self.time_idx)):
            curr_state = self._check_signal_state(i)
            if curr_state != prev_state:
                signal_changes.append(i)
                prev_state = curr_state

        # 在信号灯变化前后各1个时间点内惩罚流量剧变
        change_window = 1
        for change_idx in signal_changes:
            start_idx = max(0, change_idx - change_window)
            end_idx = min(len(self.time_idx) - 1, change_idx + change_window)
            window_gradient = np.diff(predicted_flow[start_idx:end_idx+1])
            gradient_sum = np.sum(np.abs(window_gradient))
            penalty += 400 * gradient_sum

        return penalty

    def optimize_model(self):
        """优化模型参数"""
        print("开始模型优化...")

        # 根据问题描述和数据特征调整初始参数
        # 支路1参数 [a1,a2,a3,a4,a5,a6,brk1,brk2,brk3,brk4] - 10个参数
        # 支路2参数 [b1,b2,b3,b4,b5] - 5个参数，移除了固定的转折点时间
        # 支路3参数 - 每个绿灯周期的斜率和截距 - 12个参数(6个周期)
        initial_guess = [
            # 支路1参数 - 根据图表观察优化初始值
            5.0, 0.0, -2.2, 34.0, 18.0, -0.8, 3.5, 10.0, 15.0, 46.0,

            # 支路2参数 - 根据图表调整参数，移除了固定的转折点
            0.6, 4.0, 29.0, -1.0, 30.0,

            # 支路3参数 - 根据绿灯周期和明显的峰值调整
            # 每个周期两个参数：(斜率,截距)
            10.0, 6.0,    # 第一个绿灯周期 (7:06-7:16)
            10.0, 16.0,   # 第二个绿灯周期 (7:24-7:34)
            10.0, 22.0,   # 第三个绿灯周期 (7:42-7:52)
            10.0, 22.0,   # 第四个绿灯周期 (8:00-8:10)
            5.0, 15.0,    # 第五个绿灯周期 (8:18-8:28)
            5.0, 12.0     # 第六个绿灯周期 (8:36-8:46)
        ]

        # 根据图表观察调整参数边界
        param_bounds = [
            # 支路1参数边界
            (3.0, 20.0),    # 增长斜率 - 增加上限以适应更陡峭的增长
            (0.0, 2.0),    # 初始值 - 确保初始值接近0
            (-4.0, -1.0),  # 减少斜率1 - 加大下限以适应更快的下降
            (28.0, 60.0),  # 峰值 - 根据图表调整范围
            (16.0, 30.0),  # 稳定值 - 降低上限以更好匹配稳定值
            (-2.0, -0.01),  # 减少斜率2
            (1.0, 10.0),    # brk1 - 调整起始增长时间
            (8.0, 11.0),   # brk2 - 根据图表调整第一个转折点
            (14.0, 16.0),  # brk3 - 细化稳定阶段开始时间
            (45.0, 50.0),  # brk4 - 延后最终减少时间点

            # 支路2参数边界 - 根据图表调整，移除了固定的转折点边界
            (0.4, 1.0),    # 增长斜率 - 降低上限以平缓增长
            (3.0, 8.0),    # 截距
            (25.0, 32.0),  # 稳定值 - 根据图表观察调整上限
            (-1.5, -0.5),  # 减少斜率 - 调整为更合适的范围
            (15.0, 35.0),  # 终值

            # 支路3参数边界 - 绿灯周期固定，仅优化各周期的斜率和截距
            (5.0, 15.0), (5.0, 15.0),    # 第一个绿灯周期
            (5.0, 15.0), (10.0, 25.0),   # 第二个绿灯周期
            (5.0, 15.0), (15.0, 30.0),   # 第三个绿灯周期
            (5.0, 15.0), (15.0, 30.0),   # 第四个绿灯周期
            (2.0, 10.0), (10.0, 20.0),   # 第五个绿灯周期
            (2.0, 10.0), (8.0, 20.0)     # 第六个绿灯周期
        ]

        # 优化策略：多重起点，多阶段优化
        best_error = float('inf')
        best_params = None
        optimization_history = []

        # 第一阶段：使用不同的初始点进行多次优化
        num_attempts = 12  # 增加尝试次数以提高找到好解的几率
        print(f"第一阶段：使用{num_attempts}个不同起点进行初步优化...")

        for attempt in range(num_attempts):
            # 使用不同的扰动策略
            if attempt < 4:
                # 在初始猜测附近小幅扰动
                perturbation = 0.1 * (1 + 0.15 * attempt)
                perturbed_guess = np.array(initial_guess) * (1 + perturbation * np.random.randn(len(initial_guess)))
            elif attempt < 8:
                # 在初始猜测附近中等幅度扰动，同时对某些关键参数额外调整
                perturbation = 0.2 * (1 + 0.1 * (attempt-4))
                perturbed_guess = np.array(initial_guess) * (1 + perturbation * np.random.randn(len(initial_guess)))

                # 特别调整支路1的转折点
                key_indices = [6, 7, 8, 9]  # 支路1转折点索引
                for idx in key_indices:
                    perturbed_guess[idx] = initial_guess[idx] + 1.5 * np.random.randn()
            else:
                # 使用更大的扰动
                perturbation = 0.3 * (1 + 0.1 * (attempt-8))
                perturbed_guess = np.array(initial_guess) * (1 + perturbation * np.random.randn(len(initial_guess)))

                # 对支路2和支路3参数进行额外调整
                s2_indices = list(range(10, 15))  # 支路2参数索引(5个参数)
                s3_indices = list(range(15, 27))  # 支路3参数索引(12个参数)

                for idx in s2_indices:
                    perturbed_guess[idx] = initial_guess[idx] * (1 + 0.25 * np.random.randn())

                for idx in s3_indices:
                    perturbed_guess[idx] = initial_guess[idx] * (1 + 0.3 * np.random.randn())

            # 确保扰动后的参数仍在边界内
            for i, (lb, ub) in enumerate(param_bounds):
                perturbed_guess[i] = max(lb, min(ub, perturbed_guess[i]))

            print(f"  尝试 {attempt+1}/{num_attempts}...")

            # 选择不同的优化方法
            methods = ['L-BFGS-B', 'SLSQP', 'TNC', 'Powell']
            method = methods[attempt % len(methods)]

            if method == 'L-BFGS-B':
                options = {'maxiter': 800, 'gtol': 1e-5}
            elif method == 'SLSQP':
                options = {'maxiter': 800, 'ftol': 1e-6}
            elif method == 'TNC':
                options = {'maxiter': 800, 'xtol': 1e-6}
            else:  # Powell
                options = {'maxiter': 800, 'xtol': 1e-6}

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

        # 选择前6个最佳结果进行细化
        sorted_results = sorted(optimization_history, key=lambda x: x['error'])
        top_k = min(6, len(sorted_results))

        for i in range(top_k):
            result = sorted_results[i]
            print(f"  细化优化 {i+1}/{top_k}, 起始误差: {result['error']:.6f}")

            try:
                refined_result = minimize(
                    self._evaluate_model,
                    result['params'],
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': 2000, 'gtol': 1e-8}
                )

                if refined_result.fun < best_error:
                    best_error = refined_result.fun
                    best_params = refined_result.x
                    print(f"    优化成功，新误差: {best_error:.6f}")
            except Exception as e:
                print(f"    细化优化失败: {str(e)}")

        # 第三阶段：使用最佳结果进行最终优化
        print("\n第三阶段：最终优化...")
        try:
            # 先使用SLSQP方法
            final_result1 = minimize(
                self._evaluate_model,
                best_params,
                method='SLSQP',
                bounds=param_bounds,
                options={'maxiter': 2500, 'ftol': 1e-10}
            )

            if final_result1.fun < best_error:
                best_error = final_result1.fun
                best_params = final_result1.x
                print(f"  SLSQP优化成功，误差: {best_error:.6f}")

            # 再使用L-BFGS-B方法进一步优化
            final_result2 = minimize(
                self._evaluate_model,
                best_params,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': 2500, 'gtol': 1e-10}
            )

            if final_result2.fun < best_error:
                best_error = final_result2.fun
                best_params = final_result2.x
                print(f"  L-BFGS-B优化成功，最终误差: {best_error:.6f}")

        except Exception as e:
            print(f"  最终优化失败: {str(e)}")

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

        # 计算特定时刻的流量
        t1 = self.KEY_TIMES['t1']  # 7:30
        t2 = self.KEY_TIMES['t2']  # 8:30
        f1_t1, f2_t1, f3_t1 = self._calculate_branch_flows(np.array([t1]), self.optimal_params)
        f1_t2, f2_t2, f3_t2 = self._calculate_branch_flows(np.array([t2]), self.optimal_params)

        print("\n关键时刻流量预测:")
        print(f"7:30 - 支路1: {f1_t1[0]:.2f}, 支路2: {f2_t1[0]:.2f}, 支路3: {f3_t1[0]:.2f}")
        print(f"8:30 - 支路1: {f1_t2[0]:.2f}, 支路2: {f2_t2[0]:.2f}, 支路3: {f3_t2[0]:.2f}")

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
        b_params = self.optimal_params[10:15]

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
        road2_breakpoints = [self.ROAD2_LINEAR_GROWTH_END, self.ROAD2_STABLE_END]
        for i, brk in enumerate(road2_breakpoints):
            plt.axvline(x=brk, color='m', ls='--', alpha=0.3)
            brk_name = "线性增长结束" if i == 0 else "稳定段结束"
            plt.annotate(f'支路2:{brk_name}', xy=(brk, 15), xytext=(brk, 20),
                        arrowprops=dict(arrowstyle='->'), color='m')

        # 标记绿灯时段
        for i, (start, end) in enumerate(zip(self.GREEN_STARTS, self.GREEN_ENDS)):
            if start < 60:  # 确保在时间范围内
                end = min(end, 60)
                plt.axvspan(start, end, color='green', alpha=0.1)
                # 只标注在图表范围内的绿灯
                if start < 60 and i < self.NUM_CYCLES:
                    plt.annotate(f'绿灯{i+1}', xy=(start + (end-start)/2, max(90, 75+i*5)),
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
        b_params = self.optimal_params[10:15]
        c_params = self.optimal_params[15:]

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
            f.write("\n")

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
            # 计算第一阶段结束点的值（用于确保连续性）
            value_at_growth_end = b_params[0] * self.ROAD2_LINEAR_GROWTH_END + b_params[1]
            f.write(f"{b_params[0]:.4f} \\cdot t + {b_params[1]:.4f}, & t \\leq {self.ROAD2_LINEAR_GROWTH_END} \\\\ ")
            f.write(f"{value_at_growth_end:.4f}, & {self.ROAD2_LINEAR_GROWTH_END} < t \\leq {self.ROAD2_STABLE_END} \\\\ ")
            f.write(f"{b_params[3]:.4f} \\cdot (t-{self.ROAD2_STABLE_END}) + {value_at_growth_end:.4f}, & t > {self.ROAD2_STABLE_END} ")
            f.write(r"\end{cases}$")
            f.write('\n\n')

            # 支路3流量模型表达式
            f.write("### 3.3 支路3流量模型\n\n")
            f.write(r"$f_3(t) = \begin{cases} ")
            # 添加所有6个绿灯周期的参数
            for i in range(self.NUM_CYCLES):
                start = self.FIRST_GREEN + i * self.CYCLE_LENGTH
                end = start + self.GREEN_DURATION
                if i < len(c_params) // 2:
                    slope, intercept = c_params[2*i], c_params[2*i+1]
                else:
                    # 使用最后一个已知周期的参数
                    last_idx = (len(c_params) // 2) - 1
                    slope, intercept = c_params[2*last_idx] * 0.9, c_params[2*last_idx+1] * 0.9

                # 添加时间区间描述
                time_start = (start * 2) % 60
                time_start_hour = 7 + time_start // 60
                time_start_min = time_start % 60

                time_end = (end * 2) % 60
                time_end_hour = 7 + time_end // 60
                time_end_min = time_end % 60

                time_desc = f"({time_start_hour}:{time_start_min:02d}-{time_end_hour}:{time_end_min:02d})"

                f.write(f"{slope:.4f} \\cdot (t-{start}) + {intercept:.4f}, & t \\in [{start}, {end}) \\text{{ 且为绿灯时段 {time_desc}}} \\\\ ")

            f.write(r"0, & \text{其他时段（红灯）} \end{cases}$")
            f.write('\n\n')

            # 6. 关键时间点流量
            f.write("## 四、关键时间点流量\n\n")
            f.write("| 时间点 | 支路1 | 支路2 | 支路3 | 主路4(预测) | 主路4(实际) |\n")
            f.write("|--------|-------|-------|-------|------------|------------|\n")
            f.write(f"| 7:30 | {f1_730[0]:5.2f} | {f2_730[0]:5.2f} | {f3_730[0]:5.2f} | {main_flow_730:8.2f} | {self.actual_flow[t1]:8.2f} |\n")
            f.write(f"| 8:30 | {f1_830[0]:5.2f} | {f2_830[0]:5.2f} | {f3_830[0]:5.2f} | {main_flow_830:8.2f} | {self.actual_flow[t2]:8.2f} |\n\n")

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
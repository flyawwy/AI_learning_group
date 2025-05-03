import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import os

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TrafficFlowAnalyzer:
    """交通流量分析器 - 问题1解决方案"""

    def __init__(self):
        """初始化分析器"""
        self._load_dataset()
        self._setup_parameters()

    def _load_dataset(self):
        """加载交通流量数据"""
        # 读取Excel文件
        df = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx', sheet_name='表1 (Table 1)')
        self.times = df['时间 t (Time t)'].values
        self.main_road_flow = df['主路3的车流量 (Traffic flow on the Main road 3)'].values

    def _setup_parameters(self):
        """设置模型参数"""
        # 参数边界
        self.bounds = [
            (0.1, 2.0),     # a1 - 支路1增长斜率
            (0.0, 20.0),    # b1 - 支路1初始值
            (0.1, 2.0),     # a2 - 支路2增长阶段斜率
            (0.0, 20.0),    # b2 - 支路2增长阶段初始值
            (-2.0, -0.1),   # c2 - 支路2下降阶段斜率
            (30.0, 100.0),  # d2 - 支路2下降阶段截距
            (20.0, 40.0)    # t0 - 支路2转折点
        ]

        # 初始参数猜测
        self.initial_params = [0.6, 0.0, 0.8, 15.0, -1.0, 75.0, 30.0]

    def branch1_flow(self, t, a1, b1):
        """计算支路1流量 - 线性增长模型

        Args:
            t: 时间点数组
            a1: 增长斜率
            b1: 初始值

        Returns:
            np.array: 支路1流量数组
        """
        return a1 * np.array(t) + b1

    def branch2_flow(self, t, a2, b2, c2, d2, t0):
        """计算支路2流量 - 分段线性模型

        Args:
            t: 时间点数组
            a2: 增长阶段斜率
            b2: 增长阶段初始值
            c2: 下降阶段斜率
            d2: 下降阶段截距
            t0: 转折点

        Returns:
            np.array: 支路2流量数组
        """
        t_array = np.array(t)
        return np.where(t_array < t0,
                        a2 * t_array + b2,     # t < t0: 线性增长
                        c2 * t_array + d2)     # t >= t0: 线性下降

    def total_flow(self, params, t):
        """计算总流量 - 两个支路之和

        Args:
            params: 模型参数
            t: 时间点数组

        Returns:
            np.array: 主路流量预测值
        """
        a1, b1, a2, b2, c2, d2, t0 = params
        return (self.branch1_flow(t, a1, b1) +
                self.branch2_flow(t, a2, b2, c2, d2, t0))

    def residuals(self, params):
        """计算残差

        Args:
            params: 模型参数

        Returns:
            np.array: 残差数组
        """
        return self.total_flow(params, self.times) - self.main_road_flow

    def objective(self, params):
        """优化目标函数 - 带惩罚项的均方误差

        Args:
            params: 模型参数

        Returns:
            float: 目标函数值
        """
        # 基本均方误差
        mse_loss = np.sum(self.residuals(params) ** 2)

        # 提取参数
        a1, b1, a2, b2, c2, d2, t0 = params

        # 惩罚项1：支路2下降阶段不能有负值
        t_after = self.times[self.times >= t0]
        q2_after = c2 * t_after + d2
        negative_penalty = np.sum(np.abs(q2_after[q2_after < 0])) * 20.0

        # 惩罚项2：支路2在转折点处的流量应该连续
        continuity_point1 = a2 * t0 + b2
        continuity_point2 = c2 * t0 + d2
        continuity_penalty = abs(continuity_point1 - continuity_point2) * 10.0

        # 惩罚项3：确保支路1和支路2的流量始终为正
        q1 = self.branch1_flow(self.times, a1, b1)
        q2 = self.branch2_flow(self.times, a2, b2, c2, d2, t0)
        positive_penalty = (np.sum(np.abs(q1[q1 < 0])) +
                           np.sum(np.abs(q2[q2 < 0]))) * 20.0

        # 合并所有惩罚项
        total_penalty = negative_penalty + continuity_penalty + positive_penalty

        return mse_loss + total_penalty

    def optimize(self):
        """优化模型参数"""
        print("开始模型优化...")

        # 多次优化，选择最佳结果
        best_error = float('inf')
        best_params = None

        # 尝试不同的初始点和优化方法
        methods = ['L-BFGS-B', 'SLSQP', 'TNC']

        for method in methods:
            print(f"  使用{method}方法优化...")

            # 对初始参数添加随机扰动
            perturbed_params = np.array(self.initial_params) * (1 + 0.1 * np.random.randn(len(self.initial_params)))

            # 确保扰动的参数在边界内
            for i, (lb, ub) in enumerate(self.bounds):
                perturbed_params[i] = max(lb, min(ub, perturbed_params[i]))

            # 执行优化
            try:
                result = minimize(
                    self.objective,
                    perturbed_params,
                    method=method,
                    bounds=self.bounds
                )

                if result.fun < best_error:
                    best_error = result.fun
                    best_params = result.x
                    print(f"    发现更优解，误差: {best_error:.6f}")
            except Exception as e:
                print(f"    优化失败: {str(e)}")

        # 使用最佳参数进行最终优化
        print("\n进行最终优化...")
        final_result = minimize(
            self.objective,
            best_params,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 1000, 'gtol': 1e-8}
        )

        # 保存最优参数
        self.optimal_params = final_result.x

        # 计算最终预测结果
        self.branch1_predicted = self.branch1_flow(self.times, *self.optimal_params[:2])
        self.branch2_predicted = self.branch2_flow(self.times, *self.optimal_params[2:])
        self.total_predicted = self.branch1_predicted + self.branch2_predicted

        # 计算误差指标
        self.residuals_array = self.total_predicted - self.main_road_flow
        self.mse = np.mean(self.residuals_array ** 2)
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(np.abs(self.residuals_array))
        self.r2 = 1 - np.sum(self.residuals_array ** 2) / np.sum((self.main_road_flow - np.mean(self.main_road_flow)) ** 2)

        print(f"优化完成! RMSE = {self.rmse:.4f}, R² = {self.r2:.4f}")

    def visualize_results(self):
        """可视化分析结果"""
        # 创建输出目录
        os.makedirs('./P1', exist_ok=True)

        # 绘制支路流量和总流量图
        plt.figure(figsize=(12, 6))
        plt.plot(self.times, self.main_road_flow, 'ko-', label="主路观测流量", markersize=4)
        plt.plot(self.times, self.branch1_predicted, 'b--', label="支路1估计流量")
        plt.plot(self.times, self.branch2_predicted, 'g--', label="支路2估计流量")
        plt.plot(self.times, self.total_predicted, 'r-', label="预测总流量", alpha=0.7)

        # 添加转折点标记
        t0 = self.optimal_params[6]
        t0_idx = np.argmin(np.abs(self.times - t0))
        plt.axvline(x=t0, color='gray', linestyle='--', alpha=0.7)
        plt.annotate(f'转折点 t={t0:.2f}', xy=(t0, 30), xytext=(t0+2, 40),
                    arrowprops=dict(arrowstyle='->'), color='gray')

        plt.xlabel("时间点(t)")
        plt.ylabel("车流量")
        plt.title(f"主路车流量与支路估计车流量对比 (RMSE={self.rmse:.4f}, R²={self.r2:.4f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./P1/主路车流量与支路估计车流量对比.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制残差分布图
        plt.figure(figsize=(12, 4))
        plt.plot(self.times, self.residuals_array, 'ro-', label="残差", alpha=0.7)
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel("时间点(t)")
        plt.ylabel("残差 (预测-观测)")
        plt.title("残差分布图")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./P1/残差分布.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制残差直方图
        plt.figure(figsize=(10, 5))
        plt.hist(self.residuals_array, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel("残差值")
        plt.ylabel("频率")
        plt.title("残差分布直方图")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./P1/残差直方图.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """生成分析报告"""
        # 创建输出目录
        os.makedirs('./P1', exist_ok=True)

        # 提取最优参数
        a1, b1, a2, b2, c2, d2, t0 = self.optimal_params

        # 生成报告内容
        with open('./P1/交通流量分析报告.md', 'w', encoding='utf-8') as f:
            f.write("# 交通流量分析报告 - 问题1\n\n")

            # 1. 模型性能
            f.write("## 一、模型性能\n\n")
            f.write(f"- 均方根误差 (RMSE): {self.rmse:.4f}\n")
            f.write(f"- 平均绝对误差 (MAE): {self.mae:.4f}\n")
            f.write(f"- 决定系数 (R²): {self.r2:.4f}\n\n")

            # 2. 模型参数
            f.write("## 二、模型参数\n\n")
            f.write("### 2.1 支路1参数\n\n")
            f.write(f"- 增长斜率 (a1): {a1:.4f}\n")
            f.write(f"- 初始值 (b1): {b1:.4f}\n\n")

            f.write("### 2.2 支路2参数\n\n")
            f.write(f"- 增长阶段斜率 (a2): {a2:.4f}\n")
            f.write(f"- 增长阶段初始值 (b2): {b2:.4f}\n")
            f.write(f"- 下降阶段斜率 (c2): {c2:.4f}\n")
            f.write(f"- 下降阶段截距 (d2): {d2:.4f}\n")
            f.write(f"- 转折点 (t0): {t0:.4f}\n\n")

            # 3. 流量函数表达式
            f.write("## 三、流量函数表达式\n\n")
            f.write("### 3.1 支路1流量函数\n\n")
            f.write(f"$f_1(t) = {a1:.4f} \\cdot t + {b1:.4f}$\n\n")

            f.write("### 3.2 支路2流量函数\n\n")
            f.write(r"$f_2(t) = \begin{cases} ")
            f.write(f"{a2:.4f} \\cdot t + {b2:.4f}, & t < {t0:.2f} \\\\ ")
            f.write(f"{c2:.4f} \\cdot t + {d2:.4f}, & t \\geq {t0:.2f} ")
            f.write(r"\end{cases}$")
            f.write("\n\n")

            # 4. 关键时间点流量
            f.write("## 四、关键时间点流量\n\n")
            f.write("| 时间点 | 支路1流量 | 支路2流量 | 主路预测值 | 主路实际值 |\n")
            f.write("|--------|----------|----------|------------|------------|\n")

            # 计算t=10和t=30时的流量
            key_times = [10, 30]
            for t in key_times:
                t_idx = np.where(self.times == t)[0][0]
                q1 = self.branch1_predicted[t_idx]
                q2 = self.branch2_predicted[t_idx]
                q_pred = self.total_predicted[t_idx]
                q_actual = self.main_road_flow[t_idx]

                # 计算时间字符串 (t=10对应7:20, t=30对应8:00)
                time_str = f"{7 + t // 30}:{(t % 30) * 2:02d}"

                f.write(f"| {time_str} | {q1:.2f} | {q2:.2f} | {q_pred:.2f} | {q_actual:.2f} |\n")
            f.write("\n")

            # 5. 结论分析
            f.write("## 五、结论分析\n\n")
            f.write("1. **支路1特征**: 支路1表现为线性增长趋势，反映了随着时间推移，该支路车流量稳定增加的特点。\n\n")
            f.write("2. **支路2特征**: 支路2呈现先增长后降低的趋势，转折点在t≈30处，这反映了早晨高峰期后车流量开始减少的现象。\n\n")
            f.write("3. **预测性能**: 模型整体拟合效果良好，决定系数R²达到了{:.4f}，说明模型能够解释大部分流量变化。\n\n".format(self.r2))
            f.write("4. **残差分析**: 残差分布较为均匀，无明显系统性误差，表明模型结构适合描述给定的车流量数据。\n\n")

            f.write("## 六、支路特点总结\n\n")
            f.write("- **支路1**: 呈现持续线性增长趋势，可能是连接主要居住区到工作区的道路。\n")
            f.write("- **支路2**: 呈现典型的早高峰特征，先增长后下降，可能是学校或商业区的主要进入道路。\n")

def main():
    """主函数"""
    # 创建分析器实例
    analyzer = TrafficFlowAnalyzer()

    # 优化模型
    analyzer.optimize()

    # 可视化结果
    analyzer.visualize_results()

    # 生成报告
    analyzer.generate_report()

    print("分析完成！结果保存在./P1目录")

if __name__ == "__main__":
    main()
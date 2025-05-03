import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pyswarm import pso
import matplotlib.pyplot as plt
import os

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TrafficFlowAnalyzer:
    """交通流量分析器 - 问题2求解"""

    def __init__(self):
        self._load_dataset()
        self._setup_parameters()

    def _load_dataset(self):
        """加载交通流量数据"""
        df = pd.read_excel('.\\2025-51MCM-Problem A\\附件(Attachment).xlsx', sheet_name='表2 (Table 2)')
        self.times = df['时间 t (Time t)'].values
        self.actual_flow = df['主路5的车流量 (Traffic flow on the Main road 5)'].values

    def _setup_parameters(self):
        """设置模型参数"""
        # 参数边界
        self.bounds = [
            (5, 10),       # C1 - 支路1常数流量
            (0.4, 0.5),    # a1 - 支路2增长斜率
            (10, 20),      # b1 - 支路2初始值
            (0.2, 0.3),    # a2 - 支路2下降斜率
            (0.4, 0.5),    # a3 - 支路3增长斜率
            (10, 20),      # b3 - 支路3初始值
            (35, 45),      # t2 - 支路3转折点
            (3, 10),       # N - 傅里叶级数项数
            (10, 45),      # T - 傅里叶周期
            (0, 10),       # A0 - 傅里叶常数项
            *[(-10, 10)]*20  # 傅里叶系数
        ]

        # 初始参数猜测
        self.initial_params = [
            7.5,           # C1 - 支路1常数流量
            0.45,          # a1 - 支路2增长斜率
            12.0,          # b1 - 支路2初始值
            0.25,          # a2 - 支路2下降斜率
            0.45,          # a3 - 支路3增长斜率
            15.0,          # b3 - 支路3初始值
            40.0,          # t2 - 支路3转折点
            5,             # N - 傅里叶级数项数
            25,            # T - 傅里叶周期
            5,             # A0 - 傅里叶常数项
            *[1]*20        # 傅里叶系数初始值
        ]

    def branch1(self, t, C1):
        """支路1流量计算 - 稳定常数"""
        return C1 if np.isscalar(t) else np.full_like(t, C1)

    def branch2(self, t, a1, b1, a2):
        """支路2流量计算 - 分段线性"""
        # 计算稳定值
        C2 = a1 * 24 + b1
        result = np.zeros_like(t, dtype=float)

        # 三段式：线性增长 -> 稳定 -> 线性下降
        mask_growth = (t <= 24)
        mask_stable = (t > 24) & (t <= 37)
        mask_decrease = (t > 37)

        result[mask_growth] = a1 * t[mask_growth] + b1
        result[mask_stable] = C2
        result[mask_decrease] = a2 * (t[mask_decrease] - 37) + C2

        return result

    def branch3(self, t, a3, b3, t2):
        """支路3流量计算 - 先线性增长后稳定"""
        result = np.zeros_like(t, dtype=float)

        # 计算稳定值
        C3 = a3 * t2 + b3

        # 两段式：线性增长 -> 稳定
        linear_growth_mask = t < t2
        stable_mask = t >= t2

        result[linear_growth_mask] = a3 * t[linear_growth_mask] + b3
        result[stable_mask] = C3

        return result

    def branch4(self, t, params_fourier):
        """支路4流量计算 - 周期性规律（傅里叶级数）"""
        N = int(round(params_fourier[0]))
        T = params_fourier[1]
        A0 = params_fourier[2]
        A = params_fourier[3:3+N]
        B = params_fourier[3+N:3+2*N]

        # 基础值
        result = np.full_like(t, A0, dtype=float)

        # 累加傅里叶级数项
        for n in range(1, N+1):
            result += A[n-1] * np.cos(2 * np.pi * n * t / T) + B[n-1] * np.sin(2 * np.pi * n * t / T)

        return result

    def total_flow(self, t, params):
        """计算总流量 - 四个支路之和"""
        C1, a1, b1, a2, a3, b3, t2 = params[:7]
        N = int(round(params[7]))
        T = params[8]
        A0 = params[9]
        A = params[10:10+N]
        B = params[10+N:10+2*N]

        params_fourier = [N, T, A0] + list(A) + list(B)

        # 支路1和支路2有2分钟的延迟（对应时间索引延迟1）
        t_delayed = np.maximum(0, t - 1)

        return (
            self.branch1(t_delayed, C1) +
            self.branch2(t_delayed, a1, b1, a2) +
            self.branch3(t, a3, b3, t2) +
            self.branch4(t, params_fourier)
        )

    def huber_loss(self, y_true, y_pred, delta=10.0):
        """Huber损失函数 - 对异常值更鲁棒"""
        error = y_true - y_pred
        return np.where(np.abs(error) <= delta,
                        0.5 * error**2,
                        delta * (np.abs(error) - 0.5 * delta))

    def objective(self, params):
        """优化目标函数"""
        predicted = self.total_flow(self.times, params)

        # 计算基本损失
        loss = self.huber_loss(self.actual_flow, predicted, delta=10.0).sum()

        # 添加自适应惩罚项
        # 支路3转折点后应为常数
        t2 = params[6]
        mask_after_t2 = self.times >= t2
        if np.any(mask_after_t2):
            a3, b3 = params[4], params[5]
            stable_value = a3 * t2 + b3
            branch3_values = self.branch3(self.times[mask_after_t2], a3, b3, t2)
            stability_penalty = 20 * np.sum((branch3_values - stable_value)**2)
            loss += stability_penalty

        # 确保傅里叶参数在合理范围内
        fourier_params = params[10:]
        fourier_penalty = 0.1 * np.sum(fourier_params**2)
        loss += fourier_penalty

        return loss

    def compute_branch_flows(self, t):
        """计算各支路流量"""
        t_array = np.array([t]) if np.isscalar(t) else np.array(t)

        # 提取各支路参数
        C1 = self.best_params[0]
        a1, b1, a2 = self.best_params[1:4]
        a3, b3, t2 = self.best_params[4:7]

        N = int(round(self.best_params[7]))
        T = self.best_params[8]
        A0 = self.best_params[9]
        A = self.best_params[10:10+N]
        B = self.best_params[10+N:10+2*N]

        params_fourier = [N, T, A0] + list(A) + list(B)

        # 支路1和支路2有2分钟的延迟（对应时间索引延迟1）
        t_delayed = np.maximum(0, t_array - 1)

        flows = {
            "支路1": self.branch1(t_delayed, C1),
            "支路2": self.branch2(t_delayed, a1, b1, a2),
            "支路3": self.branch3(t_array, a3, b3, t2),
            "支路4": self.branch4(t_array, params_fourier)
        }

        return {k: v[0] if np.isscalar(t) else v for k, v in flows.items()}

    def format_expressions(self):
        """格式化各支路流量表达式"""
        C1 = self.best_params[0]
        a1, b1, a2 = self.best_params[1:4]
        a3, b3, t2 = self.best_params[4:7]

        N = int(round(self.best_params[7]))
        T = self.best_params[8]
        A0 = self.best_params[9]
        A = self.best_params[10:10+N]
        B = self.best_params[10+N:10+2*N]

        # 计算中间值
        C2 = a1 * 24 + b1
        C3 = a3 * t2 + b3

        # 格式化表达式
        expr1 = f"f_1(t) = {C1:.2f}"

        expr2 = (f"f_2(t) = \\begin{{cases}} "
                f"{a1:.4f}t + {b1:.2f}, & t < 24 \\\\ "
                f"{C2:.2f}, & 24 \\leq t < 37 \\\\ "
                f"{a2:.4f}(t-37) + {C2:.2f}, & t \\geq 37 "
                "\\end{cases}")

        expr3 = (f"f_3(t) = \\begin{{cases}} "
                f"{a3:.4f}t + {b3:.2f}, & t < {t2:.1f} \\\\ "
                f"{C3:.2f}, & t \\geq {t2:.1f} "
                "\\end{cases}")

        # 傅里叶级数表达式
        fourier_terms = [f"{A0:.4f}"]
        for n in range(1, N+1):
            if abs(A[n-1]) > 1e-4:
                sign = "+" if A[n-1] > 0 else "-"
                fourier_terms.append(f"{sign} {abs(A[n-1]):.4f}\\cos\\left(\\frac{{2\\pi {n} t}}{{{T:.2f}}}\\right)")
            if abs(B[n-1]) > 1e-4:
                sign = "+" if B[n-1] > 0 else "-"
                fourier_terms.append(f"{sign} {abs(B[n-1]):.4f}\\sin\\left(\\frac{{2\\pi {n} t}}{{{T:.2f}}}\\right)")

        expr4 = f"f_4(t) = " + " ".join(fourier_terms)

        return {
            "支路1": expr1,
            "支路2": expr2,
            "支路3": expr3,
            "支路4": expr4
        }

    def optimize(self):
        """优化模型参数"""
        print("开始模型优化...")

        # 使用PSO进行全局优化
        print("执行PSO全局优化...")
        xopt_pso, _ = pso(
            lambda p: self.objective(p),
            lb=[b[0] for b in self.bounds],
            ub=[b[1] for b in self.bounds],
            swarmsize=200,
            omega=0.5,
            phip=0.5,
            phig=0.5,
            maxiter=1000,
            minfunc=1e-8
        )

        # 使用L-BFGS-B进行局部优化
        print("执行L-BFGS-B局部优化...")
        result = minimize(
            self.objective,
            xopt_pso,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
        )

        self.best_params = result.x
        self.best_error = result.fun

        # 计算关键时刻的流量
        self.flow_7_30 = self.compute_branch_flows(15)  # t=15对应7:30
        self.flow_8_30 = self.compute_branch_flows(45)  # t=45对应8:30

        # 计算最终预测值和误差指标
        self.predicted_flow = self.total_flow(self.times, self.best_params)
        self.mse = np.mean((self.predicted_flow - self.actual_flow) ** 2)
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(np.abs(self.predicted_flow - self.actual_flow))
        self.r2 = 1 - np.sum((self.actual_flow - self.predicted_flow) ** 2) / np.sum((self.actual_flow - np.mean(self.actual_flow)) ** 2)

        print(f"优化完成! RMSE = {self.rmse:.4f}, R² = {self.r2:.4f}")

    def plot_results(self):
        """可视化分析结果"""
        output_dir = './P2'
        os.makedirs(output_dir, exist_ok=True)

        # 主图：流量拟合结果
        plt.figure(figsize=(12, 7))
        plt.plot(self.times, self.actual_flow, label='实际主路流量', color='blue', marker='o', markersize=4)
        plt.plot(self.times, self.predicted_flow, label='拟合主路流量', linestyle='--', color='green')

        # 添加各支路流量
        flows = self.compute_branch_flows(self.times)
        colors = ['red', 'purple', 'orange', 'brown']
        for i, (name, flow) in enumerate(flows.items()):
            plt.plot(self.times, flow, label=name, color=colors[i], alpha=0.7)

        plt.xlabel('时间（分钟）')
        plt.ylabel('流量（辆/2分钟）')
        plt.title(f'主路及各支路流量拟合结果 (RMSE={self.rmse:.4f}, R²={self.r2:.4f})')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/主路及各支路流量拟合结果.png', dpi=300)
        plt.close()

        # 残差分析图
        residuals = self.actual_flow - self.predicted_flow

        plt.figure(figsize=(12, 6))
        plt.plot(self.times, residuals, label='残差', color='red', marker='x')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('时间（分钟）')
        plt.ylabel('残差')
        plt.title('残差分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/残差分布.png', dpi=300)
        plt.close()

        # 残差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel('残差')
        plt.ylabel('频率')
        plt.title('残差分布直方图')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/残差分布直方图.png', dpi=300)
        plt.close()

    def generate_report(self):
        """生成分析报告"""
        output_dir = './P2'
        os.makedirs(output_dir, exist_ok=True)

        # 获取格式化表达式
        expressions = self.format_expressions()

        # 生成报告内容
        report_lines = [
            '# 交通流量分析报告 - 问题2',
            '',
            '## 模型性能',
            f'- 均方根误差(RMSE): {self.rmse:.4f}',
            f'- 平均绝对误差(MAE): {self.mae:.4f}',
            f'- 决定系数(R²): {self.r2:.4f}',
            '',
            '## 各支路流量函数表达式',
            '',
            f'$${expressions["支路1"]}$$',
            '',
            f'$${expressions["支路2"]}$$',
            '',
            f'$${expressions["支路3"]}$$',
            '',
            f'$${expressions["支路4"]}$$',
            '',
            '## 关键时间点流量值',
            '',
            '### 7:30 (t=15) 各支路流量',
            ''
        ]

        # 添加7:30流量数据
        for name, value in self.flow_7_30.items():
            report_lines.append(f'- {name}: {value:.2f}')

        report_lines.extend([
            '',
            '### 8:30 (t=45) 各支路流量',
            ''
        ])

        # 添加8:30流量数据
        for name, value in self.flow_8_30.items():
            report_lines.append(f'- {name}: {value:.2f}')

        # 保存报告
        with open(f'{output_dir}/结果分析报告.md', 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + '\n')

        print(f"分析报告已保存至 {output_dir}/结果分析报告.md")

def main():
    """主函数"""
    # 创建分析器实例
    analyzer = TrafficFlowAnalyzer()

    # 优化模型
    analyzer.optimize()

    # 可视化结果
    analyzer.plot_results()

    # 生成报告
    analyzer.generate_report()

    # 输出关键信息
    print("\n关键时间点流量值:")
    print("7:30各支路流量:", analyzer.flow_7_30)
    print("8:30各支路流量:", analyzer.flow_8_30)

if __name__ == "__main__":
    main()
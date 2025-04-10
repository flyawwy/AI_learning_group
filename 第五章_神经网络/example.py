import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 生成数据
X = torch.linspace(-5, 5, 100).reshape(-1, 1)
y = torch.sin(X) + 0.2 * torch.randn(X.size())

# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # 输入层→隐藏层
        self.fc2 = nn.Linear(10, 1)  # 隐藏层→输出层
        self.activation = nn.ReLU()   # 激活函数

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
epochs = 1000

# 创建一个图形，包含两个子图
plt.figure(figsize=(12, 5))

# 训练循环
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    # 每100个epoch绘制一次预测结果
    if epoch % 100 == 0 or epoch == epochs-1:
        plt.clf()  # 清除当前图形
        
        # 绘制损失函数
        plt.subplot(1, 2, 1)
        plt.plot(losses, 'b-', label='训练损失')
        plt.title('训练过程中的损失变化')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
        
        # 绘制预测结果
        plt.subplot(1, 2, 2)
        with torch.no_grad():
            predicted = model(X)
        plt.scatter(X.numpy(), y.numpy(), c='b', label='真实数据', alpha=0.5)
        plt.plot(X.numpy(), predicted.numpy(), 'r-', label='模型预测', linewidth=2)
        plt.title(f'预测结果 (Epoch {epoch+1})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.pause(0.5)  # 暂停一小段时间以显示动画效果

plt.show()  # 显示最终结果
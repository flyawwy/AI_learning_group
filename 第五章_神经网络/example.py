import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 可视化
plt.plot(losses, label='Training Loss')
plt.legend()
plt.show()
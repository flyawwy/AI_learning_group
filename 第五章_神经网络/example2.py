import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 生成更复杂的数据
def generate_complex_data(n_samples=1000):
    X = torch.linspace(-10, 10, n_samples).reshape(-1, 1)
    # 生成一个更复杂的函数：sin(x) + 0.3*cos(2x) + 0.2*sin(3x)
    y = torch.sin(X) + 0.3*torch.cos(2*X) + 0.2*torch.sin(3*X) + 0.1*torch.randn(X.size())
    return X, y

# 生成训练集和测试集
X_train, y_train = generate_complex_data(1000)
X_test, y_test = generate_complex_data(200)

# 定义更复杂的网络
class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# 训练设置
model = ComplexNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)

# 训练参数
epochs = 2000
train_losses = []
test_losses = []
best_loss = float('inf')

# 创建图形
plt.figure(figsize=(15, 5))

# 训练循环
for epoch in range(epochs):
    # 训练模式
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    train_loss = criterion(outputs, y_train)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
    
    # 更新学习率
    scheduler.step(test_loss)
    
    # 保存最佳模型
    if test_loss < best_loss:
        best_loss = test_loss
        best_model_state = model.state_dict().copy()
    
    # 每50个epoch更新一次图像
    if epoch % 50 == 0 or epoch == epochs-1:
        plt.clf()
        
        # 绘制损失函数
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, 'b-', label='训练损失', alpha=0.7)
        plt.plot(test_losses, 'r-', label='测试损失', alpha=0.7)
        plt.title('损失函数变化')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # 使用对数尺度
        
        # 绘制训练集预测结果
        plt.subplot(1, 3, 2)
        with torch.no_grad():
            train_predicted = model(X_train)
        plt.scatter(X_train.numpy(), y_train.numpy(), c='b', label='训练数据', alpha=0.3, s=10)
        plt.plot(X_train.numpy(), train_predicted.numpy(), 'r-', label='模型预测', linewidth=2)
        plt.title(f'训练集预测 (Epoch {epoch+1})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        
        # 绘制测试集预测结果
        plt.subplot(1, 3, 3)
        with torch.no_grad():
            test_predicted = model(X_test)
        plt.scatter(X_test.numpy(), y_test.numpy(), c='g', label='测试数据', alpha=0.3, s=10)
        plt.plot(X_test.numpy(), test_predicted.numpy(), 'r-', label='模型预测', linewidth=2)
        plt.title(f'测试集预测 (Epoch {epoch+1})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.pause(0.1)

# 加载最佳模型
model.load_state_dict(best_model_state)
plt.show()

# 打印最终损失
print(f'最终训练损失: {train_losses[-1]:.6f}')
print(f'最终测试损失: {test_losses[-1]:.6f}')
print(f'最佳测试损失: {best_loss:.6f}')
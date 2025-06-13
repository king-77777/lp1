import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

# 设置 TensorBoard 输出路径
writer = SummaryWriter(log_dir="runs/mlp_experiment")

# 读取数据
data = pd.read_csv("life_expectancy_data.csv")
data = data.drop(columns=["Country", "Year"])
data["Status"] = data["Status"].map({"Developing": 0, "Developed": 1})
data.fillna(data.median(numeric_only=True), inplace=True)

# 特征和目标
X = data.drop(columns=["Life expectancy "])
y = data["Life expectancy "]

# 特征选择（相关性 > 0.3）
corr = data.corr()["Life expectancy "].abs().sort_values(ascending=False)
selected_features = corr[corr > 0.3].index.tolist()
selected_features.remove("Life expectancy ")
X = X[selected_features]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

# 转为 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MLP(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型并记录 TensorBoard 日志
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # 验证集 loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

    # TensorBoard 记录
    writer.add_scalars("Loss", {"Train": loss.item(), "Test": val_loss.item()}, epoch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {val_loss.item():.4f}")

# 输出评估指标
y_pred = model(X_test_tensor).detach().numpy()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ 最终评估结果：")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# 权重直方图（每一层）
for name, param in model.named_parameters():
    if 'weight' in name:
        writer.add_histogram(name, param, epoch)

# 保存模型结构图
writer.add_graph(model, X_train_tensor)

writer.close()

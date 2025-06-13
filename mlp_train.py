import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import yaml
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# ✅ 1. 加载模型参数（来自 YAML 文件）
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["mlp"]

# ✅ 2. 读取并预处理数据
df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})  # 状态转为 0 / 1
X = df.drop(columns=["Life expectancy "]).values
y = df["Life expectancy "].values.reshape(-1, 1)

# ✅ 3. 特征标准化处理
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ✅ 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ 5. 定义多层感知器模型（MLP）
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))  # 输出层
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ✅ 6. 初始化模型、损失函数、优化器
input_dim = X.shape[1]
hidden_layers = params["hidden_layer_sizes"]
model = MLP(input_dim, hidden_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 7. 初始化 TensorBoard（保留历史日志 + 自定义 run 名）
run_comment = f"{len(hidden_layers)}layer_{'_'.join(map(str, hidden_layers))}_lr0.001"
log_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_comment}"
writer = SummaryWriter(log_dir=log_dir)

EPOCHS = params.get("max_iter", 500)
train_loss_curve = []
val_loss_curve = []

# ✅ 8. 开始训练
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_curve.append(avg_train_loss)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)

    # 验证阶段
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(test_loader)
    val_loss_curve.append(avg_val_loss)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)

    # ✅ 写入每层的权重分布直方图
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    print(f"[{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# ✅ 9. 模型评估（还原原始标签尺度）
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_inv, y_pred)
r2 = r2_score(y_test_inv, y_pred)
print(f"✅ 模型评估完成: MSE = {mse:.3f}, R² = {r2:.3f}")

# ✅ 10. 绘制并保存损失曲线图
plt.figure(figsize=(8, 5))
plt.plot(train_loss_curve, label="Train Loss")
plt.plot(val_loss_curve, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Neural Network Learning Curve (PyTorch)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_loss_curve.png")
plt.show()

# ✅ 11. 保存性能指标
with open("mlp_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"mse": mse, "r2": r2}, f, indent=2)

# ✅ 12. 关闭 TensorBoard 写入器
writer.close()

# ✅ 输出提示
print("✅ 损失曲线图保存为：mlp_loss_curve.png")
print("✅ 模型指标写入：mlp_metrics.json")
print(f"✅ TensorBoard 权重与损失日志路径：{log_dir}")
print("➡ 请运行命令：tensorboard --logdir=runs 查看多训练曲线对比")

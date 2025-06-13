import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# 🟢 1. 指定模型文件名
model_file = "linear_model.pkl"  # 可替换为 tree_model.pkl、mlp_model.pkl 等

# 🟢 2. 加载模型
if not os.path.exists(model_file):
    raise FileNotFoundError(f"找不到模型文件：{model_file}")

model = joblib.load(model_file)
print(f"✅ 成功加载模型文件: {model_file}")
print(f"模型类型: {type(model).__name__}")

# 🟢 3. 加载数据并做预处理
df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})

X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

# 🟢 4. 模型类型判断与分析展示
if isinstance(model, LinearRegression):
    print("\n📊 线性回归系数（特征权重）:")
    for name, coef in zip(X.columns, model.coef_):
        print(f"{name}: {coef:.3f}")
    print(f"\n截距: {model.intercept_:.3f}")

elif isinstance(model, DecisionTreeRegressor):
    print("\n🌳 决策树结构展示（前两层）:")
    plt.figure(figsize=(14, 6))
    plot_tree(model, max_depth=2, feature_names=X.columns, filled=True)
    plt.title("决策树前两层")
    plt.show()

elif isinstance(model, MLPRegressor):
    print("\n🧠 神经网络结构:")
    for i, w in enumerate(model.coefs_):
        print(f"第{i+1}层权重矩阵形状: {w.shape}")
        print(w)

else:
    print("🟡 暂不支持此模型的结构打印，但你仍可调用 model.predict() 进行预测")

# 🟢 5. 示例预测
print("\n🔍 使用模型预测前 5 条样本:")
predictions = model.predict(X.head())
for i, p in enumerate(predictions):
    print(f"样本 {i+1} 的预测预期寿命: {p:.2f}")

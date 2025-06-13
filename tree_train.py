import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import yaml
import json

# 读取超参数
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["tree"]

# 数据预处理
df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})

X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

# 划分训练集和测试集（40% 用于评估）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 创建模型并训练
model = DecisionTreeRegressor(
    max_depth=params["max_depth"],
    random_state=params.get("random_state", 42)
)
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "tree_model.pkl")
print("✅ 决策树模型已保存: tree_model.pkl")

# 模型评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 写入指标文件供 DVC 使用
with open("tree_metrics.json", "w") as f:
    json.dump({"MSE": mse, "R2": r2}, f, indent=4)

print("✅ 指标已写入 tree_metrics.json")

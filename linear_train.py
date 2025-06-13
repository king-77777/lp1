import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import yaml
import json

# 读取超参数
with open("params.yaml") as f:
    params = yaml.safe_load(f)["linear"]

# 读取和预处理数据
df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"])
df = df.dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})

X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=42)

# 模型训练
model = LinearRegression(fit_intercept=params["fit_intercept"])
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "linear_model.pkl")
print("✅ 模型已训练并保存: linear_model.pkl")

# ✅ 模型评估并写入 metrics 文件
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

with open("linear_metrics.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f, indent=2)

print("✅ 评估指标已写入 linear_metrics.json")

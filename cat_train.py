import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import yaml
import json

# 读取超参数
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["catboost"]

# 数据加载和预处理
df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})

X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=42)

# 模型训练
model = CatBoostRegressor(**params)
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "catboost_model.pkl")
print("✅ CatBoost 模型已保存: catboost_model.pkl")

# 模型评估
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# 保存指标
metrics = {
    "mse": round(mse, 2),
    "r2": round(r2, 4)
}
with open("catboost_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("✅ 指标已写入 catboost_metrics.json")

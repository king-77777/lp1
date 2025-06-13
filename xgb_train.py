import pandas as pd
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import matplotlib.pyplot as plt

# 📂 数据加载与预处理
df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

# 🤖 模型训练
model = XGBRegressor(verbosity=0, random_state=42)
model.fit(X, y)

# 💾 模型保存
joblib.dump(model, "xgb_full_model.pkl")
print("✅ XGBoost 模型已使用全部数据训练并保存为 xgb_full_model.pkl")

# 📊 模型评估
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 📝 保存评估指标
with open("xgb_full_metrics.json", "w") as f:
    json.dump({"mse": round(mse, 3), "r2": round(r2, 3)}, f, indent=2)
print("📄 评估指标已保存为 xgb_full_metrics.json")

# 🔍 特征重要性提取并保存为 CSV
importance = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

importance_df.to_csv("xgb_feature_importance.csv", index=False)
print("📊 特征重要性已保存为 xgb_feature_importance.csv")

# 📈 可视化并保存为图像
plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"], color='lightgreen')
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png")
print("📈 图像已保存为 xgb_feature_importance.png")

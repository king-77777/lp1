
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

df = pd.read_csv("selected_features.csv")
X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

model = joblib.load("model.pkl")
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

metrics = {
    "mse": round(mse, 2),
    "r2": round(r2, 2)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 模型评估完成，已输出 metrics.json")

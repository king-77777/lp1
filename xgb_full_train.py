import pandas as pd
from xgboost import XGBRegressor
import joblib

df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

model = XGBRegressor(verbosity=0, random_state=42)
model.fit(X, y)

joblib.dump(model, "xgb_full_model.pkl")
print("✅ XGBoost 模型已使用全部数据训练")

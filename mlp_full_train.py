import pandas as pd
from sklearn.neural_network import MLPRegressor
import joblib

df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
model.fit(X, y)

joblib.dump(model, "mlp_full_model.pkl")
print("✅ 神经网络已使用全部数据训练完成")

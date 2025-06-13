import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"])
df = df.dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})

X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "linear_full_model.pkl")
print("✅ 使用全部数据训练完成，已保存为 linear_full_model.pkl")

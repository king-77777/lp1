import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib

df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X, y)

joblib.dump(model, "tree_full_model.pkl")
print("✅ 决策树模型已使用全部数据训练并保存")

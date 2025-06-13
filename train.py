
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("selected_features.csv")
X = df.drop(columns=['Life expectancy '])
y = df['Life expectancy ']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ 模型训练完成并已保存为 model.pkl")

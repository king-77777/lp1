import pandas as pd, joblib, json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("life_expectancy_data.csv").drop(columns=["Country", "Year"]).dropna()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
X = df.drop(columns=["Life expectancy "])
y = df["Life expectancy "]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = joblib.load("cat_model.pkl")
y_pred = model.predict(X_test)

metrics = {"mse": round(mean_squared_error(y_test, y_pred), 3),
           "r2": round(r2_score(y_test, y_pred), 3)}

with open("cat_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ CatBoost 评估完成: cat_metrics.json")


import pandas as pd

df = pd.read_csv("life_expectancy_data.csv")
df.fillna(df.median(numeric_only=True), inplace=True)
df.to_csv("processed_data.csv", index=False)
print("✅ 数据预处理完成")

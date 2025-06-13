import pandas as pd

df = pd.read_csv("processed_data.csv")

# 删除非数值列（只保留 float/int 类型）
df_numeric = df.select_dtypes(include='number')

# 相关性分析并筛选特征
correlation = df_numeric.corr()['Life expectancy '].abs().sort_values(ascending=False)
selected_features = correlation[correlation > 0.3].index.tolist()
selected_features.remove('Life expectancy ')

# 保存选中的特征列 + 目标列
df_numeric[selected_features + ['Life expectancy ']].to_csv("selected_features.csv", index=False)

print("✅ 特征选择完成")

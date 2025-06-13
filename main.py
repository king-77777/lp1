# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

# ==============================
# 第 1 部分：数据集描述与特征选择 (EDA)
# ==============================

# 加载数据集
data = pd.read_csv('life_expectancy_data.csv')

# 数据集基本信息
print("数据集基本信息：")
print(data.info())

# 查看前几行数据
print("\n数据集预览：")
print(data.head())

# 检查缺失值
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("\n缺失值统计：\n", missing_values)

# 填充缺失值（使用中位数）
data.fillna(data.median(numeric_only=True), inplace=True)

# 编码分类变量
data['Status'] = data['Status'].map({'Developing': 0, 'Developed': 1})

# 描述性统计
print("\n描述性统计：")
print(data.describe())

# 目标变量分布
plt.figure(figsize=(10, 6))
sns.histplot(data['Life expectancy '], bins=30, kde=True, color='skyblue')
plt.title('预期寿命分布')
plt.xlabel('预期寿命')
plt.ylabel('频率')
plt.show()

# 删除非数值列
data_numeric = data.drop(columns=['Country', 'Year'])

# 查看特征与目标变量的相关性
correlation_matrix = data_numeric.corr()
correlation_with_target = correlation_matrix['Life expectancy '].sort_values(ascending=False)
print("\n与预期寿命相关性：\n", correlation_with_target)

# 可视化相关性矩阵
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('特征相关性矩阵')
plt.show()

# 根据相关性筛选特征
selected_features = correlation_with_target[abs(correlation_with_target) > 0.3].index.tolist()
selected_features.remove('Life expectancy ')

print("\n选择的特征：", selected_features)

# ==============================
# 第 2 部分：根据 EDA 结果构建流水线 (DVC)
# ==============================
# DVC 步骤：
# 1. 初始化 DVC：
#    dvc init
#
# 2. 创建 DVC 阶段：
# - 数据预处理阶段：
#    dvc run -n preprocess -d life_expectancy_data.csv -o processed_data.csv python preprocess.py
#
# - 特征选择阶段：
#    dvc run -n feature_selection -d processed_data.csv -o selected_features.csv python feature_selection.py
#
# - 模型训练阶段：
#    dvc run -n train_model -d selected_features.csv -o model.pkl python train_model.py
#
# - 模型评估阶段：
#    dvc run -n evaluate -d model.pkl -o metrics.json python evaluate.py
#
# 3. 查看 DVC 流水线图：
#    dvc dag

# ==============================
# 第 3 部分：线性回归模型实现
# ==============================

# 数据预处理：选择相关特征和目标变量
X = data[selected_features]
y = data['Life expectancy ']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建线性回归模型
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 预测
y_pred = linear_model.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n线性回归模型 - 均方误差 (MSE): {mse}")
print(f"线性回归模型 - R² 得分: {r2}")

# 查看权重（系数）
coefficients = pd.Series(linear_model.coef_, index=selected_features).sort_values(ascending=False)
print("\n线性回归模型 - 权重（系数）：\n", coefficients)

# 绘制预测值和真实值的对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("真实值 (预期寿命)")
plt.ylabel("预测值 (预期寿命)")
plt.title("线性回归模型预测结果")
plt.show()

# 显示特征的重要性
plt.figure(figsize=(12, 8))
coefficients.plot(kind='bar')
plt.title('线性回归模型 - 特征重要性 (系数)')
plt.show()

# ==============================
# 第 4 部分：决策树模型（最大深度 = 4）+ 实际深度显示 + 图层限制
# ==============================

tree_model_4 = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model_4.fit(X_train, y_train)

# 模型评估
y_pred_tree = tree_model_4.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"\n决策树模型(max_depth=4) - MSE: {mse_tree:.2f}  R²: {r2_tree:.2f}")
print("实际使用的树深度：", tree_model_4.get_depth())

# 绘图
plt.figure(figsize=(20, 10))
plot_tree(tree_model_4, 
          feature_names=selected_features, 
          filled=True, 
          rounded=True, 
          fontsize=10, 
          max_depth=3)  # 可选：控制展示层数
plt.title("决策树结构图(显示前3层)")
plt.show()

# ==============================
# 完成
# ==============================

import numpy as np
from scipy.optimize import linprog
import pandas as pd

# 读取Excel文件中的数据
data = pd.read_excel('成本加成定价.xlsx')

# 提取所需数据
category_names = data['分类名称'].values
category_cost = data['总成本'].values
category_sales_price = data['品类销售单价'].values
category_sales_volume = data['品类销售量'].values
category_transport_loss = data['平均损耗率'].values
budget_limit = 1000000  # 商超的预算限制

# 假设每天的销售需求存储在一个长度为7的列表daily_demand中
# 您需要根据您的数据来设置daily_demand
daily_demand = [1000, 1200, 900, 1100, 1300, 950, 1050]

# 合并相同品类的数据
unique_categories = np.unique(category_names)
merged_data = []

for category in unique_categories:
    indices = np.where(category_names == category)[0]
    merged_entry = {
        '分类名称': category,
        '成本': np.mean(category_cost[indices]),
        '品类销售单价': np.mean(category_sales_price[indices]),
        '品类销售量': np.sum(category_sales_volume[indices]),
        '运输损耗': np.mean(category_transport_loss[indices])
    }
    merged_data.append(merged_entry)

merged_df = pd.DataFrame(merged_data)

# 构建线性规划目标函数的系数向量
# 目标函数是要最大化的数量，即总收益
# 在这里，我们根据题目要求构建目标函数
num_categories = len(unique_categories)
f = np.zeros(2 * num_categories)
for i in range(num_categories):
    f[i] = -(category_sales_price[i] - category_cost[i]) * daily_demand[i % 7]
    f[i + num_categories] = -category_transport_loss[i] * daily_demand[i % 7]

# 构建线性规划约束矩阵A和右侧约束向量b
# 这里需要根据新的约束条件来构建
# 在这个示例中，我们只添加了一个库存约束
# 您需要根据具体约束条件进一步构建A和b
A = np.zeros((num_categories, 2 * num_categories))
for i in range(num_categories):
    A[i, i] = -1
    A[i, i + num_categories] = -1

# 设置库存约束，假设每个品类的库存不超过最大库存容量
max_inventory_capacity = 500
b = np.array([-max_inventory_capacity] * num_categories)

# 构建线性规划约束边界
bounds = [(0, None)] * (2 * num_categories)  # 所有变量的下界为0，上界为无穷大

# 使用线性规划求解器求解
result = linprog(f, A_ub=A, b_ub=b, bounds=bounds, method='simplex')

# 提取最优补货总量和最优定价策略
optimal_replenishment = result.x[:num_categories]
optimal_pricing_strategy = result.x[num_categories:]

# 打印最优策略
for i, category in enumerate(unique_categories):
    print(f"{category} 最优补货量: {optimal_replenishment[i]}")
    print(f"{category} 最优定价策略: {optimal_pricing_strategy[i]}")

# 打印最大总收益
print("最大总收益:", -result.fun)

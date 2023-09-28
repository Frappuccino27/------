from pymoo.core.problem import Problem
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize

import numpy as np
import pandas as pd

# 读取Excel文件中的数据
data = pd.read_excel('~/桌面/数模国一/提交版/2023国赛C题提交版/C题/Python代码//问题二 支撑材料/第二题第二问/成本加成定价.xlsx')  # 替换为您的数据文件

# 提取所需数据
category_names = data['分类名称'].values
category_sales_price = data['品类销售单价'].values
category_wholesale_price = data['品类批发价格'].values
category_cost_addition = data['成本加成定价'].values
category_loss_rate = data['平均损耗率'].values

# 预测的每天的销售需求数据
daily_demand_predictions = np.array([
    [147.043753, 47.62095, 69.723208, 19.459019, 19.171694, 11.455291],
    [176.312842, 67.591781, 109.254956, 32.313881, 20.696354, 14.72123],
    [203.907935, 67.21362, 114.117458, 40.607864, 23.101579, 18.621244],
    [156.591079, 59.545905, 81.948173, 24.301677, 17.390321, 11.506333],
    [140.303653, 48.474784, 70.726631, 22.438577, 17.769178, 10.279744],
    [132.478564, 55.796775, 80.212048, 21.570322, 16.0298, 17.345301],
    [146.66664, 54.764696, 83.518169, 26.263543, 15.046251, 23.874639]
])

# 合并相同品类的数据
unique_categories = np.unique(category_names)
merged_data = []

for category in unique_categories:
    indices = np.where(category_names == category)[0]
    merged_entry = {
        '分类名称': category,
        '品类销售单价': np.mean(category_sales_price[indices]),
        '品类批发价格': np.mean(category_wholesale_price[indices]),
        '成本加成定价': np.mean(category_cost_addition[indices]),
        '损耗率(%)': np.mean(category_loss_rate[indices])
    }
    merged_data.append(merged_entry)

merged_df = pd.DataFrame(merged_data)

# 定义多目标优化问题
class MultiObjectiveProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2 * len(unique_categories),
                         n_obj=2,
                         n_constr=0,
                         xl=np.zeros(2 * len(unique_categories)),
                         xu=np.ones(2 * len(unique_categories)))

    def _evaluate(self, X, out, *args, **kwargs):
        replenishments = X[:, :len(unique_categories)]
        pricing_strategies = X[:, len(unique_categories):]

        objectives = np.zeros((X.shape[0], 2))

        for i in range(X.shape[0]):
            total_profit = 0
            total_cost = 0

            for day in range(7):
                for j, category in enumerate(unique_categories):
                    category_index = np.where(merged_df['分类名称'].values == category)[0][0]

                    # 补货量受约束
                    max_replenishment = 50  # 根据您的需求修改
                    replenishment = replenishments[i][j] * max_replenishment

                    # 计算销售量和成本
                    purchase_quantity = replenishment + merged_df[merged_df['分类名称'] == category]['损耗率(%)'].values[0] / 100
                    cost = category_wholesale_price[category_index] * purchase_quantity
                    revenue = pricing_strategies[i][j] * daily_demand_predictions[day][category_index]
                    profit = revenue - cost

                    total_profit += profit
                    total_cost += cost

            objectives[i, 0] = -total_profit  # 最大化总收益
            objectives[i, 1] = total_cost  # 最小化总成本

        out["F"] = objectives

# 初始化问题
problem = MultiObjectiveProblem()

# 初始化算法和其他参数
algorithm = NSGA2(pop_size=100,
                   n_offsprings=20,
                   eliminate_duplicates=True)

# 运行多目标优化
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               save_history=True,
               verbose=True)

# 获取 Pareto 前沿中的解
pareto_front = res.F

# 打印 Pareto 前沿中的解
print("Pareto Front:")
print(pareto_front)

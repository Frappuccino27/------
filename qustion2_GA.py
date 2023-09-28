import pandas as pd
from scipy.optimize import minimize

# 读取数据
historical_df = pd.read_excel('/home/hyk/桌面/成本加成定价.xlsx')
avg_sales_price = historical_df.groupby('分类名称')['品类销售单价'].mean()
# 提取2023年7月1日到2023年7月7日的销售预测
forecast_df = pd.DataFrame({
    '日期': ['2023/7/1', '2023/7/2', '2023/7/3', '2023/7/4', '2023/7/5', '2023/7/6', '2023/7/7'],
    '花叶类': [147.043753, 176.312842, 203.907935, 156.591079, 140.303653, 132.478564, 146.66664],
    '食用菌': [47.62095, 67.591781, 67.21362, 59.545905, 48.474784, 55.796775, 54.764696],
    # ... 其他蔬菜品类
})
forecast_df['日期'] = pd.to_datetime(forecast_df['日期'])

# 假设我们有以下的参数
max_stock_capacity = 1000
budget = 10000

# 定义目标函数
def objective(x, category, date):
    sales_forecast = forecast_df[forecast_df['日期'] == date][category].iloc[0]
    loss_rate = historical_df[historical_df['分类名称'] == category]['平均损耗率'].mean() / 100
    revenue = x[1] * sales_forecast
    cost = x[0] * historical_df[historical_df['分类名称'] == category]['品类批发价格'].mean()
    transport_loss = x[0] * loss_rate
    profit = revenue - cost
    total_loss_value = transport_loss * x[1]  # 运输损耗的总价值

    # 这里我们让利润最大化，而运输损耗最小化。你可以根据需要调整这两个值的权重。
    return -(3*profit - total_loss_value)

# 约束函数定义
def sales_demand_constraint(x, category, date):
    return forecast_df[forecast_df['日期'] == date][category].iloc[0] - x[0]

def stock_constraint(x):
    return max_stock_capacity - x[0]

def cost_constraint(x, category):
    return budget - x[0] * historical_df[historical_df['分类名称'] == category]['品类批发价格'].mean()

# 循环通过每个日期和品类进行优化
results = {}
categories = historical_df['分类名称'].unique()
dates = forecast_df['日期'].unique()
categories = [col for col in forecast_df.columns if col != "日期"]
for date in dates:
    results[date] = {}
    for category in categories:
        initial_estimate = [100, avg_sales_price[category]]
        constraints = [
            {'type': 'ineq', 'fun': sales_demand_constraint, 'args': (category, date)},
            {'type': 'ineq', 'fun': stock_constraint},
            {'type': 'ineq', 'fun': cost_constraint, 'args': (category,)}
        ]
        res = minimize(objective, x0=[100, 10], args=(category, date), constraints=constraints)
        results[date][category] = {
            '补货量': res.x[0],
            '销售单价': res.x[1]
        }

print(results)
import numpy as np

# 定义遗传算法的参数
DNA_SIZE = 2
POP_SIZE = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
N_GENERATIONS = 50

def initial_population():
    # 随机初始化种群
    return np.random.rand(POP_SIZE, DNA_SIZE)

def get_fitness(pop, category, date):
    # 计算每个个体的适应度，根据利润最大化定义适应度函数
    fitness = []
    for i in range(POP_SIZE):
        x = pop[i, 0]  # 补货量
        y = pop[i, 1]  # 销售单价
        sales_forecast = forecast_df[forecast_df['日期'] == date][category].iloc[0]
        loss_rate = historical_df[historical_df['分类名称'] == category]['平均损耗率'].mean() / 100
        revenue = x * sales_forecast
        cost = x * historical_df[historical_df['分类名称'] == category]['品类批发价格'].mean()
        transport_loss = x * loss_rate
        profit = revenue - cost
        total_loss_value = transport_loss * sales_forecast
        fitness.append(3 * profit - total_loss_value)  # 最大化利润
    return np.array(fitness)

def crossover(parent, pop):
    if np.random.rand() < CROSSOVER_RATE:
        # 随机选择另一个父母
        mate = pop[np.random.randint(POP_SIZE)]
        # 随机选择交叉点
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(bool)
        # 交叉操作
        parent[cross_points] = mate[cross_points]
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            # 随机变异
            child[point] = np.random.rand()
    return child

# 主循环
results = {}
for date in dates:
    results[date] = {}
    for category in categories:
        pop = initial_population()
        for _ in range(N_GENERATIONS):
            fitness = get_fitness(pop, category, date)
            pop = pop[np.argsort(-fitness)]  # 按适应度降序排序
            pop = pop[:POP_SIZE]  # 选择前面的个体作为新的种群
            for i in range(POP_SIZE):
                parent = pop[i]
                child = crossover(parent.copy(), pop)
                child = mutate(child)
                pop[i] = child
        results[date][category] = {
            '补货量': pop[0, 0],
            '销售单价': pop[0, 1]
        }

print(results)

total_max_profit = 0  # 用于存储七天的收益最大总和

for date in dates:
    max_profit = float("-inf")  # 初始化为负无穷大
    for category in categories:
        pop = initial_population()
        for _ in range(N_GENERATIONS):
            fitness = get_fitness(pop, category, date)
            max_fitness = -fitness.max()  # 利润最大化
            if max_fitness > max_profit:
                max_profit = max_fitness
            pop = pop[np.argsort(-fitness)]
            pop = pop[:POP_SIZE]
            for i in range(POP_SIZE):
                parent = pop[i]
                child = crossover(parent.copy(), pop)
                child = mutate(child)
                pop[i] = child
    total_max_profit += max_profit

print("七天内的收益最大总和：", total_max_profit)

# import matplotlib.pyplot as plt

# # 利润变化图
# profit_history = []  # 存储每次迭代的最大利润
# for date in dates:
#     for category in categories:
#         pop = initial_population()
#         profit_iter = []  # 存储每次迭代的利润
#         for _ in range(N_GENERATIONS):
#             fitness = get_fitness(pop, category, date)
#             max_profit = -fitness.max()  # 利润最大化
#             profit_iter.append(max_profit)
#             pop = pop[np.argsort(-fitness)]
#             pop = pop[:POP_SIZE]
#             for i in range(POP_SIZE):
#                 parent = pop[i]
#                 child = crossover(parent.copy(), pop)
#                 child = mutate(child)
#                 pop[i] = child
#         profit_history.append(profit_iter)

# # 绘制利润变化图
# for i, date in enumerate(dates):
#     for j, category in enumerate(categories):
#         plt.subplot(len(dates), len(categories), i * len(categories) + j + 1)
#         plt.plot(range(N_GENERATIONS), profit_history[i * len(categories) + j])
#         plt.title(f'{category} ({date})')
#         plt.xlabel('迭代次数')
#         plt.ylabel('利润')
# plt.tight_layout()
# plt.show()

# # 补货量分布图
# for date in dates:
#     for category in categories:
#         pop = initial_population()
#         for _ in range(N_GENERATIONS):
#             fitness = get_fitness(pop, category, date)
#             pop = pop[np.argsort(-fitness)]
#             pop = pop[:POP_SIZE]
#             for i in range(POP_SIZE):
#                 parent = pop[i]
#                 child = crossover(parent.copy(), pop)
#                 child = mutate(child)
#                 pop[i] = child
#         replenishment = pop[0, 0]  # 最优补货量
#         plt.hist(pop[:, 0], bins=20, alpha=0.5)
#         plt.axvline(replenishment, color='red', linestyle='dashed', linewidth=2, label='最优补货量')
#         plt.title(f'{category} ({date})')
#         plt.xlabel('补货量')
#         plt.ylabel('频数')
#         plt.legend()
#         plt.show()



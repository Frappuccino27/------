import pandas as pd
import numpy as np

# 读取Excel文件中的数据
data = pd.read_excel('成本加成定价.xlsx')

# 提取所需数据
category_names = data['分类名称'].values
category_sales_price = data['品类销售单价'].values
category_wholesale_price = data['品类批发价格'].values
category_sales_volume = data['品类销售量'].values
category_cost_addition = data['成本加成定价'].values

# 假设每天的销售需求存储在一个长度为7的列表daily_demand中
# 您需要根据您的数据来设置daily_demand
daily_demand = [100, 1200, 900, 1100, 1300, 950, 1050]  # 举例，根据实际数据修改

# 合并相同品类的数据
unique_categories = np.unique(category_names)
merged_data = []

for category in unique_categories:
    indices = np.where(category_names == category)[0]
    merged_entry = {
        '分类名称': category,
        '品类销售单价': np.mean(category_sales_price[indices]),
        '品类批发价格': np.mean(category_wholesale_price[indices]),
        '品类销售量': np.sum(category_sales_volume[indices]),
        '成本加成定价': np.mean(category_cost_addition[indices])
    }
    merged_data.append(merged_entry)

merged_df = pd.DataFrame(merged_data)

# 初始化每天的最优策略和总收益列表
optimal_strategies = []
total_profits = []

# 模拟每天的销售需求，假设销售需求是随机的
np.random.seed(0)  # 设置随机种子以确保可重复性
daily_demand_predictions = np.random.randint(80, 150, size=(7, len(unique_categories)))

# 遍历每一天的销售需求
for day in range(7):
    # 初始化最优策略字典
    optimal_strategy = {}

    # 遍历每个品类
    for category_name in unique_categories:
        # 检查品类是否存在于数据中
        if category_name in merged_df['分类名称'].values:
            # 获取品类在数据中的索引
            category_index = np.where(merged_df['分类名称'].values == category_name)[0][0]

            # 计算最佳补货量，确保不为负数
            replenishment = max(daily_demand_predictions[day, category_index] - category_sales_volume[category_index], 0)

            # 随机生成最佳定价策略（这里假设在一定范围内生成）
            min_price = category_wholesale_price[category_index]
            max_price = category_sales_price[category_index]
            pricing_strategy = np.random.uniform(min_price, max_price)

            # 将最佳策略存储到字典中
            optimal_strategy[category_name] = {
                '最优补货量': replenishment,
                '最优定价策略': pricing_strategy
            }

    # 计算当天的总收益
    total_profit = np.sum(
        [(strategy['最优定价策略'] - category_wholesale_price[category_index]) * strategy['最优补货量'] for category_name, strategy in optimal_strategy.items()]
    )

    # 如果总收益为负数，则将总收益设为0
    total_profit = max(total_profit, 0)

    # 将最优策略和总收益存储到列表中
    optimal_strategies.append(optimal_strategy)
    total_profits.append(total_profit)

# 创建一个列表来存储每天的数据
daily_data = []

# 遍历每一天
for day, (strategy, profit) in enumerate(zip(optimal_strategies, total_profits)):
    # 遍历每个品类的策略
    for category_name, data in strategy.items():
        daily_data.append({
            '日期': f"未来第{day + 1}天",
            '分类名称': category_name,
            '最优补货量': data['最优补货量'],
            '最优定价策略': data['最优定价策略'],
            '总收益': profit
        })

# 创建DataFrame
daily_df = pd.DataFrame(daily_data)

# 将DataFrame写入Excel文件
daily_df.to_excel('每天最优策略和收益.xlsx', index=False)

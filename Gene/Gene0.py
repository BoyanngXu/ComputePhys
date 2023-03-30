# 找函数极小值点

import random
import numpy as np

# 创建一个随机个体
def create_individual():
    return [random.uniform(-3, 3), random.uniform(-3, 3)]   # 个体：x,y坐标表示的点  x,y坐标被称为个体的基因

# 定义适应度函数，计算个体的适应度值
def fitness(individual):
    x, y = individual
    return 50 - (x ** 2) - (y ** 2)   # 越小越适应，越易被选取

# 变异操作，对个体进行随机变异
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:   # 触发变异几率
        index = random.randint(0, 1)   # 随机选取某个坐标
        individual[index] += random.gauss(0, 1)   # 加减一个随机变化
    return individual   # 变异后的新个体

# 交叉操作，对两个父代个体进行交叉产生两个子代个体
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:   # 触发交叉几率
        child1 = [parent1[0], parent2[1]]   # 坐标互换
        child2 = [parent2[0], parent1[1]]
    else:
        child1 = parent1[:]
        child2 = parent2[:]
    return child1, child2

# 选择操作，根据适应度值选择一个个体
def selection(population):
    fitnesses = [fitness(individual) for individual in population]   # 计算每个个体对应的函数值
    total_fitness = sum(fitnesses)
    p = [fitness / total_fitness for fitness in fitnesses]
    try:
        population[np.random.choice(len(population), p=p)]
    except(ValueError):
        print(population, "\n", fitnesses, "\n", p)
    return population[np.random.choice(len(population), p=p)]   # 按概率选中一个个体

# 定义遗传算法函数
def genetic_algorithm(population_size=1000, num_generations=100, crossover_rate=0.8, mutation_rate=0.01):
    population = [create_individual() for _ in range(population_size)]  # 种群：population_size个体，即 population_size个坐标数组 构成的数组

    for generation in range(num_generations):  # 迭代指定代数
        new_population = []
        for _ in range(population_size // 2):  # 保持新种群数量不变
            parent1 = selection(population)  # 选择两个父代个体
            parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2, crossover_rate)  # 进行交叉操作产生两个子代个体
            new_population.append(mutate(child1, mutation_rate))  # 对子代个体进行变异操作并加入新种群
            new_population.append(mutate(child2, mutation_rate))
        population = new_population  # 更新种群

    best_individual = max(population, key=fitness)  # 找到最优个体
    return best_individual

result = genetic_algorithm()  # 运行遗传算法求解问题
print(f"The minimum value of the function f=(x^2)+(y^2) is {50 - fitness(result)} at point {result}")
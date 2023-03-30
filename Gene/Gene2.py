# 汤普森问题 - 命令能量单调递减

import random
import numpy as np
import matplotlib.pyplot as plt

# 创建一个随机个体
def create_individual(N):
    n_individuals = []
    for i in range(N):
        th = random.uniform(0, 2 * np.pi)
        ph = random.uniform(0, np.pi)
        individual = [th, ph]
        n_individuals.append(individual)
    return n_individuals   # 个体：N个theta, phi坐标表示的点组成的列表

# 定义适应度函数，计算个体的适应度值
def fitness(n_individuals):
    fitnesses = 0
    for j in range(len(n_individuals)):
        th_0, ph_0 = n_individuals[j]  # 逐个提取
        x_0, y_0, z_0 = np.cos(th_0) * np.sin(ph_0), np.sin(th_0) * np.sin(ph_0), np.cos(ph_0)
        for i in range(len(n_individuals)):
            th, ph = n_individuals[i]
            x, y, z = np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)
            r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2 + (z - z_0) ** 2)
            # if r != 0:
            #     energy += 1/r
            # else: continue
            fitnesses += r
    return fitnesses


# 变异操作，对个体进行随机变异
def mutate(n_individuals, mutation_rate):
    if random.random() < mutation_rate:   # 触发变异几率
        num = random.randint(0, len(n_individuals)-1)   # 随机选取某个点
        index = random.randint(0, 1)   # 随机选取某个坐标
        n_individuals[num][index] += random.gauss(0, np.sin(np.pi/2))   # 加减一个随机变化
    return n_individuals   # 变异后的新个体


# 交叉操作，对两个父代个体进行交叉产生两个子代个体
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:   # 触发交叉几率
        index = random.randint(0, len(parent1)-1)
        child1 = parent1[:]
        child2 = parent2[:]
        child1[index] = parent2[index]
        child2[index] = parent1[index]
    else:
        child1 = parent1[:]
        child2 = parent2[:]
    return child1, child2

# 选择操作，根据适应度值选择一个个体
def selection(population):
    fitnesses = [fitness(n_individuals) for n_individuals in population]   # 计算每个个体对应的函数值
    total_fitness = sum(fitnesses)
    p = [fitness / total_fitness for fitness in fitnesses]
    # print(fitnesses,"\n",p)
    return population[np.random.choice(len(population), p=p)]   # 按概率选中一个个体

# 定义遗传算法函数
def genetic_algorithm(population_size=100, num_generations=100, crossover_rate=0.9, mutation_rate=0.3):
    population = [create_individual(N = 6) for _ in range(population_size)]  # 种群：population_sizeg个个体，即 population_size个坐标数组 构成的三维列表

    for generation in range(num_generations + 1):  # 迭代指定代数
        best_individual = max(population, key=fitness)
        plot(best_individual, generation)
        print(f'第{generation}代分布：{best_individual}')
        if generation == num_generations + 1: break

        new_population = []
        new_best_individual = best_individual
        while fitness(new_best_individual) <= fitness(best_individual):
            for _ in range(population_size // 2):  # 保持新种群大小不变
                parent1 = selection(population)  # 选择两个父代个体
                parent2 = selection(population)
                child1, child2 = crossover(parent1, parent2, crossover_rate)  # 进行交叉操作 产生两个子代个体
                new_population.append(mutate(child1, mutation_rate))  # 对子代个体进行变异操作 并加入新种群
                new_population.append(mutate(child2, mutation_rate))
                new_best_individual = max(new_population, key=fitness)
            print(1000 / fitness(new_best_individual))

        population = new_population  # 更新种群

    return population

#定义绘图函数
def plot(individual, generation):
    fig = plt.figure()  # 绘图
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f'{generation}-generation(s) charges distribution (N={len(individual)})')

    # 绘制球体
    u, v = np.mgrid[0:2 * np.pi:40j, 0:2 * np.pi:40j]
    x1 = np.cos(u) * np.sin(v)
    y1 = np.sin(u) * np.sin(v)
    z1 = np.cos(v)
    ax.plot_wireframe(x1, y1, z1, color="0.5", linewidth=0.1)

    # 绘制散点图
    th = [loc[0] for loc in individual]
    ph = [loc[1] for loc in individual]
    x, y, z = [], [], []
    for i in range(len(individual)):
        x.append(np.cos(th[i]) * np.sin(ph[i]))
        y.append(np.sin(th[i]) * np.sin(ph[i]))
        z.append(np.cos(ph[i]))
    print(x, y, z)
    ax.scatter(x, y, z, c='r')
    ax.text(0.5, 0, 2.3, f'Energy = {(100000 / fitness(individual)):.6f}')

    plt.show()

genetic_algorithm()  # 运行遗传算法求解问题



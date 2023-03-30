import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import time

# Rosenbrock函数
def f(x):
    res = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return res

# 数值求梯度
def grad(f,x,steps):
    grad = np.array([(f(x+[steps,0])-f(x))/steps,(f(x+[0,steps])-f(x))/steps])
    return grad

# 梯度下降找极小值
def process_point(p0,n):
    iter = 1000  # 迭代步数
    k = 0.02  # 学习率
    tol = 10 ** (-5)  # 允差
    x1 = np.zeros(iter)
    y1 = np.zeros(iter)
    x = p0[n]
    y = 9999
    x1[0] = x[0]; y1[0] = x[1]  # 各点迭代历程备份

    for i in range(iter):
        if abs(f(x)-y) < tol: break  # 迭代达标——退出循环
        if (f(x) > y):  # 步长过长——状态回滚
            x = np.array([x1[i-1],y1[i-1]])
            k = k/2
        y = f(x)  # y坐标迭代更新
        x1[i] = x[0]; y1[i] = x[1]  # 迭代历程存档
        gr = grad(f,x,10**(-9))
        norm = np.linalg.norm(gr)  #梯度归一化
        print("test:{}, iter:{}, loc={}, grad={}".format(n+1, i, [f"{i:.4f}" for i in x], [f"{i:.4f}" for i in gr/norm]))  # 迭代历程展示
        x = x - k * y * (gr/norm)  #x坐标迭代更新
    return x

# 最小二乘拟合
def fitting(p1):
    xp = [sublist[0] for sublist in p1]  # 提取结果x坐标
    yp = [sublist[1] for sublist in p1]  # 提取结果y坐标
    coe = np.polyfit(xp, yp, 2)
    return np.poly1d(coe)  # 曲线拟合

if __name__ == '__main__':
    # 初始化
    start_time = time.time()
    test = 20  # 选取点数
    p0 = [np.array([random.uniform(-5, 5), random.uniform(-1, 5)]) for n in range(test)]  # 随机生成初始点
    p1 = [np.array([0,0]) for n in range(test)]  # 预备存放结果

    # 寻找极小值
    pool = mp.Pool()   # 创建进程池

    results = []
    for n in range(test):
        results.append(pool.apply_async(process_point,args=(p0,n)))   # 函数异步执行
    for n, res in enumerate(results):   # 存储结果
        x = res.get()   # 58、60-62 等价于 x = process_point(p0, n)
        p1[n] = x  #存放各点迭代结果
        plt.plot(p1[n][0], p1[n][1],'o',color='red')

    pool.close()   # 关闭进程池
    pool.join()

    # 拟合与绘图
    poly = fitting(p1)
    print("收敛曲线方程:",poly)
    # for n in range(test):   #箭头绘制
    #     plt.arrow(p0[n][0], p0[n][1], p1[n][0] - p0[n][0], p1[n][1] - p0[n][1], head_width=0.3, head_length=0.3,
    #                   length_includes_head=True)

    xl = np.linspace(-2, 2, 100)
    yl = poly(xl)
    plt.plot(xl, yl, "r-",color='blue')   #曲线绘制
    plt.show()

    # 计时
    end_time = time.time()
    print(f"总耗时:{end_time-start_time:.6f}s")
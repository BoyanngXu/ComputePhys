import multiprocessing as mp

import numpy
import numpy as np  #改用numpy计算

def f(x):
    return 4/(1+x**2)

def parallel_integrate(steps, processes):
    pool = mp.Pool(processes)
    x = np.linspace(0, 1, steps+1)  #确定x采样取值
    y = pool.map(f, x)
    pool.close()  #关闭进程池
    pool.join()  #回收进程资源
    return np.sum(y[:-1] + y[1:]) * 0.5 / steps  #梯形法则

if __name__ == '__main__':
    N = [2, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    for n in N:
        result = parallel_integrate(n, 4)  #固定进程数
        print(f"Number of Steps: {n}, Result: {result}, Error: {abs((numpy.pi - result)/numpy.pi)*100:.6f}%")
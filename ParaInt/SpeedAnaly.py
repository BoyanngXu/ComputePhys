import time
import numpy as np
import multiprocessing as mp

def f(x):
    return 4/(1+x**2)

def parallel_integrate(steps, processes):
    pool = mp.Pool(processes)  #进程池数目
    dx = 1 / steps  #步长
    x = np.arange(0, 1, dx)  #确定x采样取值
    y = pool.map(f, x)
    pool.close()  #关闭进程池
    return sum(y) * dx  #梯形法则

steps = 1000000   #步长
processes_list = [2, 4, 8, 16]   #测试的进程数

if __name__ == '__main__':
    if mp.get_start_method() != 'spawn':
        mp.set_start_method('spawn')   #使用spawn模块启动方法

    # 单进程测试
    single_start_time = time.time()   #记录计算开始时间
    single_result = parallel_integrate(steps, 1)
    single_end_time = time.time()   #记录计算完成时间

    print(f"Single process: {single_result} in {single_end_time - single_start_time:.6f} seconds")

    # 多进程测试
    for processes in processes_list:
        multi_start_time = time.time()  #记录计算开始时间
        multi_result = parallel_integrate(steps, processes)
        multi_end_time = time.time()   #记录计算完成时间

        print(f"{processes} processes: {multi_result} in {multi_end_time - multi_start_time:.6f} seconds")
        print(f"Speedup: {single_end_time - single_start_time:.6f} / {multi_end_time - multi_start_time:.6f} = {(single_end_time - single_start_time)/(multi_end_time - multi_start_time):.2f}x")
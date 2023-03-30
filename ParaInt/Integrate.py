import multiprocessing as mp

def f(x):
    return 4/(1+x**2)

def int(a, b, N, pool):
    dx = (b - a) / N  #确定微元大小
    x = [a + i * dx for i in range(N + 1)]  #确定x采样取值
    y = pool.map(f, x) #各个x所对应的y值
    result = (y[0] + 2 * sum(y[1:N]) + y[N]) * dx / 2  #梯形法则
    return result

if __name__ == '__main__':  #确保并行处理在主程序运行
    N = 1000000
    num_processes = mp.cpu_count()
    pool = mp.Pool(num_processes)
    a = 0
    b = 1
    integral = int(a, b, N, pool)
    print(integral)
    pool.close() #关闭进程池
    pool.join() #回收进程资源
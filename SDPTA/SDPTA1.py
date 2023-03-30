import numpy as np
import matplotlib.pyplot as plt

loc = np.array([(4, 3), (2, 3), (1, 4), (4, 2), (3, 1), (1, 1)])
sig = np.array([1, 1, 1, -1, -1, -1])
w = np.zeros(2)  # 初始化权重向量
b = 0  # 初始化偏置值
lr = 0.1  # 学习率
epoch = 100  # 迭代次数

for i in range(epoch):
    err = 0
    for j in range(len(loc)):
        x = loc[j]
        y = sig[j]
        a = np.dot(w, x) + b  # 计算激活值
        if np.sign(a) != np.sign(y):  # 判断是否分错类
            w += lr * y * x  # 更新权重向量
            b += lr * y  # 更新偏置值
            err += 1  # 错误次数加 1
    if err == 0:  # 分类正确，提前结束迭代
        break
print(i,err)

# 绘制分类结果
fig, ax = plt.subplots()
for i in range(len(loc)):
    if sig[i] == 1:
        ax.scatter(loc[i][0], loc[i][1], marker='o', color='red', s=100)
    else:
        ax.scatter(loc[i][0], loc[i][1], marker='o', color='blue', s=100)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
x = np.linspace(xmin, xmax)
y = -(w[0] * x + b) / w[1]  # 计算分类直线
ax.plot(x, y)
plt.show()

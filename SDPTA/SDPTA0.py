import numpy as np
import matplotlib.pyplot as plt

def est(list):
    list1 = [0]*len(list)
    for i in range(len(list)):
        list1[i] = float('%.3g'%list[i])
    return list1

loc = [(4,3),(2,3),(1,4),(4,2),(3,1),(1,1)]    # 点坐标
sig = [1,1,1,-1,-1,-1]     # 标志
l = [1,-1,0]  # ax+by+c=0一般式参数
k = 0.1     # 学习速率
n = 10   # 迭代步数

for j in range(n):
    r = 0  # 误差
    for i in range (len(loc)):     # 误差计算
        w = loc[i][0]*l[0]+loc[i][1]*l[1]+l[2]
        if w > 0: s = -1
        elif w == 0: s = 0
        else: s = 1
        r = r + abs(sig[i]-s)

        if sig[i]-s > 0:    # 直线移位
            l = [l[0]-k*loc[i][0],l[1]-k*loc[i][1],l[2]-k]
        if sig[i]-s < 0:
            l = [l[0]+k*loc[i][0],l[1]+k*loc[i][1],l[2]+k]
        print(est(l),r)

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
y = -(l[0]/l[1])*x-(l[2]/l[1])  # 计算分类直线
ax.plot(x, y)
plt.show()


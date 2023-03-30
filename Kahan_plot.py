import random

def est(num):
    num = float('%.6g'%num)
    return num

def trad_sum(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
        sum = est(sum)
    return sum

def kahan_sum(arr):
    sum = 0
    c = 0
    for i in range(len(arr)):
        y = arr[i] - c
        y = est(y)
        t = sum + y
        t = est(t)
        c = (t - sum) - y
        c = est(c)
        sum = t
    return est(sum)

arr = [random.getrandbits(20)/10,random.getrandbits(20)/100,random.getrandbits(20)/1000,random.getrandbits(20)/10000,random.getrandbits(20)/100000]
print("电脑精度：6，求和数组：", arr)
print('正确结果：',est(sum(arr)))
print("传统求和结果：", trad_sum(arr))
print("传统求和结果与正确结果之差：", est(abs(trad_sum(arr) - est(sum(arr)))))

kahan_sum = kahan_sum(arr)
print("Kahan算法求和结果：", kahan_sum)
print("Kahan算法求和结果与正确结果之差：", est(abs(kahan_sum - est(sum(arr)))))

if(est(abs(kahan_sum - est(sum(arr))))) < est(abs(trad_sum(arr) - est(sum(arr)))):
    print("Kahan算法性能好")
elif(est(abs(kahan_sum - est(sum(arr))))) == est(abs(trad_sum(arr) - est(sum(arr)))):
    print("两算法性能一样好")
else:
    print("传统算法性能好")
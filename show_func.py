import matplotlib.pyplot as plt
import numpy as np
import math


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


x = np.linspace(0, 10, 400)
y1 = []
y2 = []
y3 = []

for i in x:
    y1.append(softmax([0,i,1])[0])
    y2.append(softmax([0,i,1])[1])
    y3.append(softmax([0,i,1])[2])

plt.plot(x, y1, label="0")
plt.plot(x, y2, label="x")
plt.plot(x, y3, label="1")

'''
plt.plot(x-np.max(x), y1, label="0")
plt.plot(x-np.max(x), y2, label="x")
plt.plot(x-np.max(x), y3, label="1")
'''

'''
def corss(z,a):
    sum=0
    for i in range(10):
        sum -= z[i]*math.log(a[i])
    return sum

y=[]
x=[]
z = [0,0,0,0,0,0,0,0,0,1]
a = [0,0,0,0,0,0,0,0,0,1]
for i in range(1,100):
    x.append(1-0.01*i)
    a[9]=1-0.01*i
    for j in range(9):
        a[j]=0.01*i/9

    y.append(corss(z,a))

plt.plot(x, y)
'''


plt.title("Softmax")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)  # 显示网格
plt.legend()
plt.show()
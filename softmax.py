import numpy as np
import matplotlib.pyplot as plt

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


x = np.linspace(-10, 10, 200)
y = softmax(x)
plt.plot(x,y)
plt.show()
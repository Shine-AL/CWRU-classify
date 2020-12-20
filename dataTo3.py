import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from scipy.io import loadmat
import numpy as np


# 读取 120000 原始数据
def read_org_data(dir, ID):
    data = loadmat(dir)[ID + '_DE_time'][:120000]
    data = data.reshape(-1, 400)
    return data


# 打标签
def make_label(dir, ID, fault):
    data = read_org_data(dir, ID)
    label = np.full((300, 1), fault)
    return data, label
# def make_label(dir, ID, fault):
#     data = read_org_data(dir, ID)
#     label = np.zeros((300,400))
#     for i in range(label.shape[0]):
#         label[i][fault] = 1
#
#     return data, label


# 读取数据并打上标签
def make_dataset():
    data1, label1 = make_label('./dataset/Normal.mat', 'X097', 0)
    data2, label2 = make_label('./dataset/0.007-Ball.mat', 'X118', 1)
    data3, label3 = make_label('./dataset/0.007-InnerRace.mat', 'X105', 2)
    data4, label4 = make_label('./dataset/0.007-OuterRace6.mat', 'X130', 3)
    data5, label5 = make_label('./dataset/0.014-Ball.mat', 'X185', 4)
    data6, label6 = make_label('./dataset/0.014-InnerRace.mat', 'X169', 5)
    data7, label7 = make_label('./dataset/0.014-OuterRace6.mat', 'X197', 6)
    data8, label8 = make_label('./dataset/0.021-Ball.mat', 'X222', 7)
    data9, label9 = make_label('./dataset/0.021-InnerRace.mat', 'X209', 8)
    data10, label10 = make_label('./dataset/0.021-OuterRace6.mat', 'X234', 9)
    # 合并数据
    x = np.vstack((data1, data2, data3, data4, data5, data6, data7, data8, data9, data10))
    y = np.vstack((label1, label2, label3, label4, label5, label6, label7, label8, label9, label10)).reshape(-1)

    # 对数据集打乱
    per = np.random.permutation(x.shape[0])
    new_x = x[per, :]
    new_y = y[per]

    train_x = new_x[:2700].reshape(-1, 400, 1)
    train_y = new_y[:2700].reshape(-1, 1)
    test_x = new_x[2700:3000].reshape(-1, 400, 1)
    test_y = new_y[2700:3000].reshape(-1, 1)

    return train_x, train_y, test_x, test_y

train_x,train_y,test_x,test_y = make_dataset()

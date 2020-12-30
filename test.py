from scipy.io import loadmat
import numpy as np


def make_data_val(dir, ID, fault):
    data = loadmat(dir)[ID + '_DE_time'][:120000]
    data = data.reshape(-1, 400)
    label = np.zeros((300, 10))

    for i in range(label.shape[0]):
        label[i][fault] = 1

    return data, label

def read_data():
    data0, label0 = make_data_val('./dataset/Normal.mat', 'X097', 0)
    data1, label1 = make_data_val('./dataset/0.007-Ball.mat', 'X118', 1)
    data2, label2 = make_data_val('./dataset/0.007-InnerRace.mat', 'X105', 2)
    data3, label3 = make_data_val('./dataset/0.007-OuterRace6.mat', 'X130', 3)
    data4, label4 = make_data_val('./dataset/0.014-Ball.mat', 'X185', 4)
    data5, label5 = make_data_val('./dataset/0.014-InnerRace.mat', 'X169', 5)
    data6, label6 = make_data_val('./dataset/0.014-OuterRace6.mat', 'X197', 6)
    data7, label7 = make_data_val('./dataset/0.021-Ball.mat', 'X222', 7)
    data8, label8 = make_data_val('./dataset/0.021-InnerRace.mat', 'X209', 8)
    data9, label9 = make_data_val('./dataset/0.021-OuterRace6.mat', 'X234', 9)

    data0 = data0.reshape(-1,400,1)
    data1 = data1.reshape(-1,400,1)
    data2 = data2.reshape(-1,400,1)
    data3 = data3.reshape(-1,400,1)
    data4 = data4.reshape(-1,400,1)
    data5 = data5.reshape(-1,400,1)
    data6 = data6.reshape(-1,400,1)
    data7 = data7.reshape(-1,400,1)
    data8 = data8.reshape(-1,400,1)
    data9 = data9.reshape(-1,400,1)

    return data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,label0,label1,label2,label3,label4,label5,label6,label7,label8,label9



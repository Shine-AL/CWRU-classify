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


# 读取数据并打上标签
data1, label1 = make_label('./dataset/Normal.mat', 'X097', 0)
data2, label2 = make_label('./dataset/0.007-Ball.mat', 'X118', 1)
data3, label3 = make_label('./dataset/0.007-InnerRace.mat', 'X105', 2)
data4, label4 = make_label('./dataset/0.007-OuterRace6.mat', 'X130', 3)

# 合并数据
x = np.vstack((data1, data2, data3, data4))
y = np.vstack((label1, label2, label3, label4)).reshape(-1)

# 对数据集打乱
per = np.random.permutation(x.shape[0])
new_x = x[per, :]
new_y = y[per]

train_x = new_x[:1000].reshape(-1, 400, 1)
train_y = new_y[:1000].reshape(-1, 1)
test_x = new_x[1000:1200].reshape(-1, 400, 1)
test_y = new_y[1000:1200].reshape(-1, 1)


def create_model():
    model = keras.Sequential()
    model.add(layers.LSTM(32, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(), loss='mae', metrics=['accuracy'])
    return model


model = create_model()
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)
history = model.fit(train_x, train_y,
                    epochs=1000,
                    batch_size=100,
                    validation_data=(test_x, test_y),
                    callbacks=[learning_rate_reduction]
                    )  # 训练模型并进行测试

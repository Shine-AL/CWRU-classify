import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from scipy.io import loadmat
import numpy as np
import dataTo3_H

train_x, train_y, test_x, test_y = dataTo3_H.make_dataset()


def create_model():
    model = keras.Sequential()
    # 输入数据的shape为(n_samples, timestamps, features)
    # 隐藏层设置为20, input_shape元组第二个参数1意指features为1
    model.add(layers.LSTM(units=20, input_shape=(train_x.shape[1], train_x.shape[2])))
    # model.add(Dropout(0.2))
    # 后接全连接层，直接输出单个值，故units为10
    model.add(layers.Dense(units=10))
    model.add(layers.Activation('softmax'))  # 选用非线性激活函数
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])  # 损失函数为平均均方误差，优化器为Adam，学习率为0.001
    return model


model = create_model()
history = model.fit(train_x, train_y, epochs=2500, batch_size=100, validation_data=(test_x, test_y))  # 训练模型并进行测试

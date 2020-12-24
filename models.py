from tensorflow import keras
from tensorflow.keras import layers


def LSTM():
    model = keras.Sequential()
    # 输入数据的shape为(n_samples, timestamps, features)
    # 隐藏层设置为20, input_shape元组第二个参数1意指features为1
    model.add(layers.LSTM(units=20, input_shape=(400, 1)))
    # model.add(Dropout(0.2))
    # 后接全连接层，直接输出单个值，故units为10
    model.add(layers.Dense(units=10))
    model.add(layers.Activation('softmax'))  # 选用非线性激活函数
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])  # 损失函数为平均均方误差，优化器为Adam，学习率为0.001
    return model


def LSTM2():
    model = keras.Sequential()
    model.add(layers.LSTM(20, input_shape=(400, 1), return_sequences=True))
    model.add(layers.LSTM(20, return_sequences=True))
    model.add(layers.LSTM(20))
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='mae', metrics=['accuracy'])
    return model


def CNN():
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (1, 1), activation='relu', input_shape=(20, 20, 1)))
    model.add(layers.MaxPooling2D((8, 8), strides=8, padding='same'))
    model.add(layers.Conv2D(16, (1, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=1, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.009),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

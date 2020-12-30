from tensorflow import keras
from tensorflow.keras import layers
import data_init as dt
import models
import matplotlib.pyplot as plt


def LSTM():
    model = keras.Sequential()
    # 输入数据的shape为(n_samples, timestamps, features)
    # 隐藏层设置为20, input_shape元组第二个参数1意指features为1
    model.add(layers.LSTM(units=20, input_shape=(400, 1)))
    # model.add(Dropout(0.2))
    # 后接全连接层，直接输出单个值，故units为10
    model.add(layers.Dense(units=10))
    model.add(layers.Activation('sigmoid'))  # 选用非线性激活函数
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])  # 损失函数为平均均方误差，优化器为Adam，学习率为0.001
    return model

train_x, train_y, test_x, test_y = dt.dataTo3_H()

model = models.LSTM()
history = model.fit(train_x, train_y, epochs=2000, batch_size=100, validation_data=(test_x, test_y))  # 训练模型并进行测试

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()#显示图例
plt.title('The graph of accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


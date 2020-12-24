import data_init as dt
import models
import matplotlib.pyplot as plt

train_x, train_y, test_x, test_y = dt.dataTo2()

model = models.CNN()
print(model.summary())
# history = model.fit(train_x, train_y, epochs=160, batch_size=30, validation_data=(test_x, test_y))  # 训练模型并进行测试
#
# plt.plot(history.history['accuracy'],label='train')
# plt.plot(history.history['val_accuracy'],label='test')
# plt.title('The graph of accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()


# model.save('./saveModel/lstm.h5')
# 读取
# restored_model = tf.keras.models.load_model('./saveModel/lstm.h5')

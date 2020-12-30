import tensorflow as tf
import data_init as dt
import models
import matplotlib.pyplot as plt

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

model.save('./saveModel/lstm.h5')
# 读取
# restored_model = tf.keras.models.load_model('./saveModel/lstm.h5')

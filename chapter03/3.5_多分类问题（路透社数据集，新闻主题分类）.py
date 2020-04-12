#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ssl

import numpy as np
from keras.datasets import reuters

ssl._create_default_https_context = ssl._create_unverified_context

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# 看下第一句
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
text = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(text)

from keras.utils.np_utils import to_categorical


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension), dtype=np.float32)
    for index, item in enumerate(sequences):
        results[index, item] = 1.0
    return results


# 数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 标签向量化
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

### 开始构建网络

from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  # 这块的形状是怎么回事
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 留点验证集
x_val = x_train[:1000]
partial_x_val = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_val = one_hot_train_labels[1000:]

history = model.fit(partial_x_val, partial_y_val, epochs=9, batch_size=512, validation_data=(x_val, y_val))
evaluate_results = model.evaluate(x_test, one_hot_test_labels)
print(evaluate_results)

predict_results = model.predict(x_test)  # 返回的是(?,46)的这么一个矩阵 ，每一行列加起来=1
print(predict_results)
print(np.argmax(predict_results[0]))  # 输出最大的元素，这个就是预测的类别

# 画图验证 epochs 对训练结果的影响
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']  # 训练损失
val_loss_values = history_dict['val_loss']  # 验证损失
epochs = range(1, len(loss_values) + 1)  # x轴的数据集，训练损失有多少size，则取这个size

# 这是损失的图，可以看出损失在训练集上一直是下降的，但是在验证集上却有上升的情况
plt.plot(epochs, loss_values, '-', label='loss')
plt.plot(epochs, val_loss_values, '--', label='val_loss')
plt.xlabel('epochs', fontsize=16)
plt.ylabel('loss')
plt.title(f'{len(loss_values)} times loss')
plt.legend()
plt.show()

plt.clf()  # 清空图像
# 这是精度的图
accuracy_values = history_dict['accuracy']  # 训练精度
val_accuracy_values = history_dict['val_accuracy']  # 验证精度
plt.plot(epochs, accuracy_values, '-', label='accuracy')
plt.plot(epochs, val_accuracy_values, '--', label='val_accuracy')
plt.xlabel('epochs', fontsize=16)
plt.ylabel('loss')
plt.title(f'{len(loss_values)} times accuracy')
plt.legend()
plt.show()

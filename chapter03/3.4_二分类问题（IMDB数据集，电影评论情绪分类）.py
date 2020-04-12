#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ssl

import numpy as np
from keras.datasets import imdb

# 关闭签名证书，否则会下载失败
ssl._create_default_https_context = ssl._create_unverified_context

# 只取前 10000 个高频出现的单词，不然数据太多
(train_data, train_labels), (test_data, test_lables) = imdb.load_data(num_words=10000)

print(train_data.dtype)  # dtype是object 其实train_data里是 25000 行 list，list 里是单词序列
print(train_data.shape)  # (25000,)
print(train_labels.shape)  # (25000,)
print(train_labels.dtype)  # object
print(test_data.shape)  # (25000,)
print(test_lables.shape)  # (25000,)

# train_labels[0]=1  0/1 负面/正面评论   train_data[0]=[1, 14, 22,....32] 这样的格式，每个数字都对应一个单词，完整的就是一个句子
print('train_data[0]={}，train_labels[0]={}'.format(train_data[0], train_labels[0]))

# 从所有训练数据中获取所有单词的最大索引，因为我们只取得前 10000 个，所以最大的是 9999
max_word_index = max([max(item) for item in train_data])
print(max_word_index)  # 9999

# 将索引翻译成原始英文
word_index = imdb.get_word_index()

# 翻转下，原来 key 是单词，value 是索引，现在我们要根据 索引拿 value
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 翻译第一句话（将每个索引对应的单词拼接起来 -3 是因为它前面有填充，这里不用纠结）
text = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(text)  # ? this film was just brilliant 。。。


# 我们不能把整数直接序列输入神经网络，所以先进行下数据处理，转换为0 1张量
def vectorize_sequences(sequences, dimension=10000):
    """
    将整数序列编码为二进制矩阵
    """
    results = np.zeros((len(sequences), dimension), dtype=np.float32)
    for index, item in enumerate(sequences):
        results[index, item] = 1.0
    return results


x_train = vectorize_sequences(train_data)
print(x_train.shape)  # (25000, 10000)
x_test = vectorize_sequences(test_data)
print(x_test.shape)  # (25000, 10000)

"""
在没转换前，train_data.shape=(25000,) 25000行数据，每行都是一个 List如：list[1,3,5....]
转换后x_train.shape=(25000, 10000)，25000行数据，每行有10000列，那么对应的值为 0 1 0 1 0 1
"""

# 之前是object类型的
y_train = np.asarray(train_labels).astype(np.float)
y_test = np.asarray(test_lables).astype(np.float)

### 开始构建网络

from keras import models
from keras import layers

model = models.Sequential()

# relu是激活函数，类似的还有prelu，elu等，主要作用可以理解为解决只能为线性变换的问题 output = dot(W,input)+b
# output = dot(W,input)+b ->  output = relu(dot(W,input)+b)
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))  # 这块的形状是怎么回事
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # sigmoid 二分类问题概率值 使用此激活函数？

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',  # 二分类问题最好使用binary_crossentropy 交叉熵
              metrics=['accuracy'])

# 为了在训练过程中监控模型在前所未见数据上的精度，需要留10000个作为训练验证集
x_val = x_train[:10000]  # 前10000个用作训练验证集
partial_x_val = x_train[10000:]

print('partial_x_val shape{}'.format(partial_x_val.shape))  # partial_x_val shape(15000, 10000)

y_val = y_train[:10000]  # 前10000个
partial_y_val = y_train[10000:]

# 开始训练 ，batchsize大，学习效率高，精度低，反之效率低，精度高
history = model.fit(partial_x_val, partial_y_val, epochs=4, batch_size=512, validation_data=(x_val, y_val))
# 评估模型,不输出预测结果
results = model.evaluate(x_test, y_test)
print(results)

history_dict = history.history
print(history_dict.keys())  # dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])

loss_values = history_dict['loss']  # 训练损失
val_loss_values = history_dict['val_loss']  # 验证损失

# 画图验证 epochs 对训练结果的影响
import matplotlib.pyplot as plt

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

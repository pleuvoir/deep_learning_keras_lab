#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ssl

from keras.datasets import imdb

# 关闭签名证书，否则会下载失败
ssl._create_default_https_context = ssl._create_unverified_context

# 只取前 10000 个高频出现的单词，不然数据太多
(train_data, train_labels), (test_data, test_lables) = imdb.load_data(num_words=10000)

print(train_data.shape)  # (25000,)
print(train_labels.shape)  # (25000,)
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
print(text)

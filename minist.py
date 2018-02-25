#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/7 14:04
# @Author  : cb_lian
# @Site    : 
# @File    : minist.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 下载或读取数据集，one_hot编码
x = tf.placeholder(tf.float32, [None, 784])  # n行784列的输入矩阵
W = tf.Variable(tf.zeros([784, 10]))  # 权重矩阵，784行10列，初始化为零
b = tf.Variable(tf.zeros([10]))  # 偏置矩阵，10列的一个array
y = tf.nn.softmax(tf.matmul(x, W)+b)  # 矩阵相乘，并使用softmax激励函数
y_ = tf.placeholder("float", [None, 10])  # 训练集的标签
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  # 计算交叉熵,用来衡量模型好坏
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 执行训练，减小交叉熵，0.01学习率
init = tf.global_variables_initializer()  # 初始化变量（initialize_all_variables已经过时）
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 正确的标签与预测的标签进行比对，确定正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # argmax当axis=0时返回每一列的最大值的位置索引,当axis=1时返回每一行中的最大值的位置索引

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):  # 2000次迭代
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次迭代随机抓取训练集100个训练数据返回
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        #  每次迭代计算正确率
        print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

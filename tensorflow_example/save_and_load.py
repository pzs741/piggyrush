# -*- coding: utf-8 -*-
# @Time    : 18-12-3 下午5:32
# @Author  : Ex_treme
# @Email   : pzsyjsgldd@163.com
# @File    : save_and_load.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf

"""训练好了一个神经网络后，可以保存起来下次使用时再次加载"""


def save():
    ## Save to file
    # remember to define the same dtype and shape when restore
    W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    # 用 saver 将所有的 variable 保存到定义的路径
    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, "my_net/save_net.ckpt")
        print("Save to path: ", save_path)


"""tensorflow 现在只能保存 variables，还不能保存整个神经网络的框架，所以再使用的时候，需要重新定义框架，然后把 variables 放进去学习。"""


def load():
    # restore variables
    # redefine the same shape and same type for your variables
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

    # not need init step

    saver = tf.train.Saver()
    # 用 saver 从路径中将 save_net.ckpt 保存的 W 和 b restore 进来
    with tf.Session() as sess:
        saver.restore(sess, "my_net/save_net.ckpt")
        print("weights:", sess.run(W))
        print("biases:", sess.run(b))


if __name__ == '__main__':
    # save()
    load()

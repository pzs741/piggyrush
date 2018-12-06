# -*- coding: utf-8 -*-
# @Time    : 18-12-3 下午5:02
# @Author  : Ex_treme
# @Email   : pzsyjsgldd@163.com
# @File    : network_demo.py
# @Software: PyCharm
"""之前写过一篇文章 TensorFlow 入门 讲了 tensorflow 的安装，这里使用时直接导入"""
import numpy as np
import tensorflow as tf

"""Step2:先定义出参数 Weights，biases，拟合公式 y，误差公式 loss"""

"""
    添加神经层：
输入参数有 inputs, in_size, out_size, 和 activation_function"""

# # 添加层
# def add_layer(inputs, in_size, out_size, activation_function=None):
#     # add one more layer and return the output of this layer
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 用 tf.Variable 定义变量，与python不同的是，必须先定义它是一个变量
#     biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 它才是一个变量，初始值为0，还可以给它一个名字 counter
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases  # 矩阵乘法：tf.matmul
#     """激励函数:例如一个神经元对猫的眼睛敏感，那当它看到猫的眼睛的时候，就被激励了，相应的参数就会被调优，它的贡献就会越大。
#     激励函数在预测层，判断哪些值要被送到预测结果那里"""
#     if activation_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_function(Wx_plus_b)
#     return outputs


keep_prob = tf.placeholder(tf.float32)


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # here to dropout
    # 在 Wx_plus_b 上drop掉一定比例
    # keep_prob 保持多少不被drop，在迭代时在 sess.run 中 feed
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


"""Step1:导入或者随机定义训练的数据 x 和 y"""
# 1.训练的数据
# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

"""
要给节点输入数据时用 placeholder，在 TensorFlow 中用placeholder 来描述等待输入的节点，只需要指定类型即可，
然后在执行节点的时候用一个字典来“喂”这些节点。相当于先把变量 hold 住，然后每次从外部传入data，注意 placeholder 和 feed_dict 是绑定用的。
"""
# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l1, 10, 1, activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))

"""Step3:选择 Gradient Descent 这个最基本的 Optimizer,神经网络的 key idea，就是让 loss 达到最小"""
# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

"""Step4:前面是定义，在运行模型前先要初始化所有变量"""
# important step 对所有变量进行初始化
init = tf.initialize_all_variables()  # 如果有变量就一定要做初始化

"""Step5:接下来把结构激活，sesseion像一个指针指向要处理的地方"""
sess = tf.Session()  # 定义 Session，它是个对象，注意大写

"""Step6:init 就被激活了，不要忘记激活"""
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)  # result 要去 sess.run 那里取结果

# 结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""Step7:训练1000步"""
# 迭代 1000 次学习，sess.run optimizer
for i in range(1000):

    """Step8:要训练 train，也就是 optimizer"""
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data, keep_prob: 0.99})
    """Step9:每 50 步打印一下结果，sess.run 指向 Weights，biases 并被输出"""
    if i % 50 == 0:
        # to see the step improvement
        acc = sess.run(accuracy, feed_dict={xs: x_data, ys: y_data, keep_prob: 1.0})
        print("Testing Accuracy= " + str(acc),
              "Testing loss= " + str(sess.run(loss, feed_dict={xs: x_data, ys: y_data, keep_prob: 1.0})))
    """这里简单提一下 feed 机制， 给 feed 提供数据，作为 run()调用的参数， feed 只在调用它的方法内有效, 方法结束, feed 就会消失。"""

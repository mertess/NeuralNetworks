import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def f(x):
    return 2*x-3   # искомая функция


samples = 1000    # количество точек
packetSize = 100   # размер пакета
x_0 = -2.0    # начало интервала
x_2 = 2.0  # конец интервала
sigma = 0.5  # среднеквадратическое отклонение шума

np.random.seed(0)
data_x = np.arange(x_0, x_2, (x_2-x_0)/samples)
np.random.shuffle(data_x)
data_y = list(f(x) + np.random.normal(0.0, sigma) for x in data_x)

tf_data_x = tf.placeholder(tf.float32, shape=(packetSize,))
tf_data_y = tf.placeholder(tf.float32, shape=(packetSize,))

weight = tf.Variable(initial_value=0.1, dtype=tf.float32, name="a")
bias = tf.Variable(initial_value=0.0, dtype=tf.float32, name="b")
model = tf.add(tf.multiply(tf_data_x, weight), bias)

loss = tf.reduce_mean((model - tf_data_y) ** 2)
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as session:
        tf.compat.v1.global_variables_initializer().run()
        for i in range(samples//packetSize):
            feed_dict = {tf_data_x: data_x[i*packetSize: (i+1)*packetSize],
                         tf_data_y: data_y[i*packetSize: (i+1)*packetSize]}
            l, _ = session.run([loss, optimizer], feed_dict=feed_dict)
            print("ошибка: %f" % (l,))
            print("a = %f, b = %f" % (weight.eval(), bias.eval()))
            plt.plot(data_x, list(map(lambda x: weight.eval() * x + bias.eval(), data_x)), data_x, data_y, 'ro')

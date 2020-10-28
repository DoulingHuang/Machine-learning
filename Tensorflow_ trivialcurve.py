import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

xs = np.arange(0,3,0.01).reshape(-1,1) #reshape將數組換成矩陣
print("xs", xs.shape)
noise_data = 0.1*(2*np.random.normal(0,2,300).astype(np.float32)-1).reshape(-1,1)

ys = xs*xs + xs + 1 + noise_data
print("ys", ys.shape)
plt.title("curve")
plt.plot(xs,ys)
plt.show()

x = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])
print(x, y_)
w1 = tf.get_variable("w1", initializer=tf.random_normal([1,32]))
w2 = tf.get_variable("w2", initializer=tf.random_normal([32,1]))
b1 = tf.get_variable("b1", initializer=tf.zeros([1,32]))
b2 = tf.get_variable("b2", initializer=tf.zeros([1,1]))
print(w1,w2,b1,b2)
l1 = tf.matmul(x, w1) + b1
print(l1)
l1 = tf.nn.relu(l1)
print(l1)
y = tf.matmul(l1, w2) + b2
print(y)

loss = tf.reduce_mean(tf.square(y - y_))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    loss_list = []
    for i in range(1001):
        loss_val , _ = sess.run([loss, train], {x:xs, y_:ys})

        if i%100==0:
            print("%d steps, loss is %f" % (i , loss_val))
            loss_list.append(loss_val)
            ys_pre = sess.run(y, {x:xs})

            plt.title("curve")
            plt.plot(xs, ys_pre)
            plt.legend("ys", "ys_pre")

plt.plot(xs , ys)
plt.show()

plt.title("loss")
plt.plot(loss_list, lw=2)
plt.show()
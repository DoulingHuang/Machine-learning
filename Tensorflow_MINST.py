from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


minst = input_data.read_data_sets("MINST_data/", one_hot=True)

def data_to_matrix(data):
    return np.reshape(data, (28,28))

# matrix = data_to_matrix(minst.train.images[1])
# plt.figure()
#
# plt.imshow(matrix)
# plt.title('the first images and label is {}'.format(np.argmax(matrix)))
# plt.matshow(matrix, cmap=plt.get_cmap('gray'))
# plt.title('the first images and label is {}'.format(np.argmax(matrix)))
# plt.show()

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.matmul(X, W) + b

lr = 0.5
batch_size = 1000
epotchs = 1000
epotchs_list = []
accuracy_list = []
lost_list = []

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_))
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)

correct_predicts = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
cp = tf.cast(correct_predicts, tf.float32)
accuracy = tf.reduce_mean(cp)

# model_path = "/tmp/model.ckpt"
# saver = tf.train.Saver()

#執行階段
init = tf.global_variables_initializer() #初始化所有變數

with tf.Session() as sess:
    sess.run(init)
    for epotch in range(epotchs+1):
        batch_x, batch_y = minst.train.next_batch(batch_size)
        _, accuracy_, loss_, cp_ = sess.run([train, accuracy, loss, cp], feed_dict={X:batch_x, Y:batch_y})
        epotchs_list.append(epotchs)
        accuracy_list.append(accuracy_)
        lost_list.append(loss_)

        if epotch%1000 == 0:
            print("accurary_={}, loss={}, epotch={}".format(accuracy_, loss_, epotch))

            plt.subplot(1,2,1)
            plt.plot(epotchs_list, accuracy_list, lw=2)
            plt.xlabel("epotch")
            plt.ylabel("accuracy")
            plt.title("a")

            plt.subplot(1, 2, 2)
            plt.plot(epotchs_list, lost_list, lw=2)
            plt.xlabel("epotch")
            plt.ylabel("loss")
            plt.title("b")
            plt.show()

    print("訓練結束")

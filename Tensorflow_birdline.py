import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#準備資料
#隨機產生100處候鳥棲息地座標
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*8
noise_data = np.random.normal(0.0,0.5,100).astype(np.float32) #產生座標偏移植

plt.hist(noise_data) #統計座標偏移植
plt.show()

y_data = x_data*8+noise_data

plt.plot(x_data, y_data, 'o', label='data:y_data = x_data*8+noise_data')
plt.legend()
plt.show()

#開始設定演算法
a = tf.Variable(tf.random_normal([1],-1.0,1.0)) #初始化全重a:隨機產生一個-1~1的值
                                                #tf.Variable用為設定變數值，在tensorflow運算中，能隨者每一次學習改變，一般常數無法改變，故使用變數值
b = tf.Variable(tf.zeros([1])) #初始化全重質b:產生一個0值

y = a * x_data + b #機器要學習的特徵模型

lost = tf.reduce_mean(tf.square(y-y_data)) #透過(預測值-實際值)取平方之後平均調整損失函數

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5) #優化方法:梯度下降法

train = optimizer.minimize(lost) #找到最小的損失


#執行階段
init = tf.global_variables_initializer() #初始化所有變數

with tf.Session() as sess:
    sess.run(init)
    lost_list = []
    for step in range(100): #訓練100次
        sess.run(train)

        lost_list.append(sess.run(lost)) #將每一次誤差值收集起來

        if step%10==0: #每10次劃一條線
            print(step, sess.run(a), sess.run(b))
            plt.plot(x_data, sess.run(a)*x_data+sess.run(b),label='model train step={}'.format(step))

#標記原始座標
plt.plot(x_data, y_data, 'o', label='data:y_data = x_data*8+noise_data')
plt.legend()
plt.show()

#將誤差值劃出
plt.plot(lost_list, lw=2)
plt.show()
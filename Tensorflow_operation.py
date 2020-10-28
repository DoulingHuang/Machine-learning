import tensorflow as tf

c0 = tf.constant(1);
c1 = tf.constant([1,2]);
c2 = tf.constant([[1],[2]]);

print(c0) #零階張量
print(c1) #一階張量，一筆資料
print(c2) #兩階張量，兩筆資料集，每筆資料集有一筆資料

sess = tf.Session()  #也可以用with tf.Session as sess:
print(sess.run(c0))  #一定要透過tf.Session()，否則只會有形狀
print(sess.run(c1))
print(sess.run(c2))

#--------------------------------------------

zeros = tf.zeros([1,3]) #建構皆為零的常數張量
print(sess.run(zeros))

ones = tf.ones([1,3]) #建構皆為一的常數張量
print(sess.run(ones))

fills = tf.fill([1,3],5) #建構特定值的常數張量
print(sess.run(fills))

range1 = tf.range(5) #建構一個數值範圍內的常數張量，並以等差方式增加
range2 = tf.range(5,delta=2) #可調整等差之值
print(sess.run(range1))
print(sess.run(range2))

linspace = tf.linspace(1.0,5.0,3) #給一組數值上下界，並且設定裡面要產生多少元素
print(sess.run(linspace))

import matplotlib.pyplot as plt

random_normal = tf.random_normal([100],0,1) #tf.random_mormal(shape, 平均值, 標準差)
print(sess.run(random_normal))
plt.hist(sess.run(random_normal))
plt.show()

n = 5000000
A = tf.truncated_normal([n,])  #tf.truncated_normal(shape, 平均值, 標準差):將張量裡面全部填上隨機值，但不超過兩倍標準差
B = tf.random_normal([n,])
a,b = sess.run([A,B])
plt.hist(a, 100, (-5,5));
plt.show()
plt.hist(b, 100, (-5,5));
plt.show()

random_uniform = tf.random_uniform([100],0,100) #tf.random_uniform(shape,下限值,上限值):將張量裡面數值全部填滿，但不會超過上下界限
print(sess.run(random_uniform))
plt.hist(sess.run(random_uniform))
plt.show()


#張量基本運算
a = tf.constant(8)
b = tf.constant(5)
tf_add = a+b #也可以用函數:tf.add(a,b) 以下亦同
tf_subtract = a-b
tf_multiply = a*b
tf_divide = a/b
tf_pow = a**b
tf_mod = a%b   #求餘數
tf_div = a//b  #求商數

print(sess.run(tf_add))
print(sess.run(tf_subtract))
print(sess.run(tf_multiply))
print(sess.run(tf_divide))
print(sess.run(tf_pow))
print(sess.run(tf_mod))
print(sess.run(tf_div))


#二為張量基本運算
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
print(sess.run(tf.matmul(matrix1,matrix2)))
print(sess.run(tf.add(matrix1,matrix2)))
print(sess.run(tf.subtract(matrix1,matrix2)))
print(sess.run(tf.multiply(matrix1,matrix2)))
print(sess.run(tf.divide(matrix1,matrix2)))
print(sess.run(tf.pow(matrix1,matrix2)))
print(sess.run(tf.mod(matrix1,matrix2)))
print(sess.run(tf.div(matrix1,matrix2)))
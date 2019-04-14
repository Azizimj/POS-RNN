import tensorflow as tf
import numpy


x = tf.placeholder(tf.int64, [2,], 'X')
sess = tf.Session()
a = tf.constant(232)
a = sess.run(a, feed_dict={x: numpy.array([1,2])})
print(a)
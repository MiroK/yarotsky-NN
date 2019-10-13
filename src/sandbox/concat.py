import tensorflow as tf


x = tf.placeholder(tf.float32, [None, 1])

A1 = tf.Variable(tf.truncated_normal(shape=[1, 3], stddev=0.1))
y1 = tf.matmul(x, A1)

y = tf.concat([x, y1], axis=1)



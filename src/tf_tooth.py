import tensorflow as tf
import itertools


def tooth(x):
    '''The tooth function'''
    A1 = tf.Variable(tf.truncated_normal(shape=[1, 3], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[3]))

    hidden = tf.add(tf.matmul(x, A1), b1)
    hidden = tf.nn.relu(hidden)

    A2 = tf.Variable(tf.truncated_normal(shape=[3, 1], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))

    y = tf.add(tf.matmul(hidden, A2), b2)

    return y, {'A1': A1, 'b1': b1, 'A2': A2, 'b2': b2}

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    def predict(sess, NN, x, x_values):
        y_ = [sess.run(NN, feed_dict={x: np.array([[xi]])}) for xi in x_values]
        return np.array(y_).flatten()

    x = tf.placeholder(tf.float32, [None, 1])
    NN, params = tooth(x)

    # Before starting, initialize the variables
    init = tf.initialize_all_variables()

    # Launch the graph.
    with tf.Session() as sess:
        sess.run(init)
        # Set the variables as in numpy
        sess.run(params['A1'].assign(np.array([[1., 1., 1.]])))
        sess.run(params['b1'].assign(np.array([0, -0.5, -1.])))
        
        sess.run(params['A2'].assign(np.array([[2, -4, 2]]).T))
        sess.run(params['b2'].assign(np.array([0.])))
        
        unit_interval = np.linspace(0, 1, 1000)
    
        plt.figure()
        plt.plot(unit_interval, predict(sess, NN, x, unit_interval))            
        plt.show()



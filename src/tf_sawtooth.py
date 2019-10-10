import tensorflow as tf


def saw_tooth(x, m):
    '''The saw-tooth function of degree m'''
    # Input to first layer
    A0 = tf.Variable(tf.truncated_normal(shape=[1, 3], stddev=0.1))
    b0 = tf.Variable(tf.constant(0.1, shape=[3]))

    # Collapse final
    Aout = tf.Variable(tf.truncated_normal(shape=[3, 1], stddev=0.1))
    bout = tf.Variable(tf.constant(0.1, shape=[1]))

    params = {'A0': A0, 'b0': b0, 'Aout': Aout, 'bout': bout}

    y1 = tf.add(tf.matmul(x, A0), b0)
    y1 = tf.nn.relu(y1)

    for s in range(1, m):
        A = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[3]))

        y1 = tf.add(tf.matmul(y1, A), b)
        y1 = tf.nn.relu(y1)

        params['A%d' % s] = A
        params['b%d' % s] = b
        
    y1 = tf.add(tf.matmul(y1, Aout), bout)

    return y1, params 

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from common import predict
    import numpy as np

    m = 4
    
    x = tf.placeholder(tf.float32, [None, 1])
    NN, params = saw_tooth(x, m=m)

    # Before starting, initialize the variables
    init = tf.initialize_all_variables()


    A = np.array([[2, -4, 2],
                  [2, -4, 2],
                  [2, -4, 2]]).T
    b = np.array([0, -0.5, -1])

    # Launch the graph.
    with tf.Session() as sess:
        sess.run(init)
        # Set the variables as in numpy
        sess.run(params['A0'].assign(np.array([[1., 1., 1.]])))
        sess.run(params['b0'].assign(np.array([0, -0.5, -1.])))
        
        sess.run(params['Aout'].assign(np.array([[2, -4, 2]]).T))
        sess.run(params['bout'].assign(np.array([0.])))

        for i in range(1, m):
            sess.run(params['A%d' % i].assign(A))
            sess.run(params['b%d' % i].assign(b))
        
        unit_interval = np.linspace(0, 1, 1001)
    
        plt.figure()
        plt.plot(unit_interval, predict(sess, NN, x, unit_interval))            
        plt.show()



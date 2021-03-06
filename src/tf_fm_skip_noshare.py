import tensorflow as tf


def x2_approx_skip(x, m):
    '''Skip connection, no weight sharing'''
    # Input to first layer
    A0 = tf.Variable(tf.truncated_normal(shape=[1, 3], stddev=0.1, dtype=tf.float64))
    b0 = tf.Variable(tf.constant(0.1, shape=[3], dtype=tf.float64))

    params = {'A0': A0, 'b0': b0}

    y1 = tf.add(tf.matmul(x, A0), b0)
    y1 = tf.nn.relu(y1)

    gs = []
    for s in range(1, m):
        # Collapse y1 to get gs
        A = tf.Variable(tf.truncated_normal(shape=[3, 1], stddev=0.1, dtype=tf.float64))
        b = tf.Variable(tf.constant(0.1, shape=[1], dtype=tf.float64))
        # Record
        params['Ac%d' % s] = A
        params['bc%d' % s] = b

        gs.append(tf.add(tf.matmul(y1, A), b))
                  
        # Make composition
        A = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1, dtype=tf.float64))
        b = tf.Variable(tf.constant(0.1, shape=[3], dtype=tf.float64))

        # Record
        params['A%d' % s] = A
        params['b%d' % s] = b
                  
        y1 = tf.add(tf.matmul(y1, A), b)
        y1 = tf.nn.relu(y1)
        
    # Collapse y1 to get gs
    A = tf.Variable(tf.truncated_normal(shape=[3, 1], stddev=0.1, dtype=tf.float64))
    b = tf.Variable(tf.constant(0.1, shape=[1], dtype=tf.float64))
    # Record
    params['Ac%d' % m] = A
    params['bc%d' % m] = b

    gs.append(tf.add(tf.matmul(y1, A), b))
        
    # Collect
    y = tf.concat([x] + gs, axis=1)

    # And the last layer
    A = tf.Variable(tf.truncated_normal(shape=[1+len(gs), 1], stddev=0.1, dtype=tf.float64))
    b = tf.Variable(tf.constant(0.1, shape=[1], dtype=tf.float64))
    # Record
    params['Aout'] = A
    params['bout'] = b
    
    y = tf.add(tf.matmul(y, A), b)

    return y, params


def yarotsky_net_init(m, sess, params):
    '''Set networks weights in this session to run like Yarotsky'''
    # Composition
    A = np.array([[2., -4, 2],
                  [2, -4, 2],
                  [2, -4, 2]]).T
    b = np.array([0, -0.5, -1])

    # Narrowing for finalizing gs
    Ag = np.array([[2, -4, 2]]).T
    bg = np.array([0])

    # For first layer
    sess.run(params['A0'].assign(np.array([[1., 1., 1.]])))
    sess.run(params['b0'].assign(np.array([0, -0.5, -1.])))

    # Intermerdiate
    for i in range(1, m+1):
        sess.run(params['Ac%d' % i].assign(Ag))
        sess.run(params['bc%d' % i].assign(bg))

    for i in range(1, m):
        sess.run(params['A%d' % i].assign(A))
        sess.run(params['b%d' % i].assign(b))

    # Collapse last
    Aout = np.array([np.r_[1., -1./(2.**(2*np.arange(1, m+1)))]]).T
    bout = np.array([0])

    sess.run(params['Aout'].assign(Aout))
    sess.run(params['bout'].assign(bout))

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tooth import x2_approx_skip as Yaro
    from common import predict
    import numpy as np

    m = 4
    
    x = tf.placeholder(tf.float64, [None, 1])
    NN, params = x2_approx_skip(x, m=m)

    # Before starting, initialize the variables
    init = tf.initialize_all_variables()


    # Launch the graph.
    with tf.Session() as sess:
        sess.run(init)

        yarotsky_net_init(m, sess, params)
        #for w in tf.trainable_variables():
        #    print sess.run(w)
        
        weight_count = sum(np.prod(w.shape) for w in tf.trainable_variables())
        print 'Number of weights', weight_count
        
        unit_interval = np.linspace(0, 1, 1001)

        print m
        yL = predict(sess, NN, x, unit_interval)
        yY = Yaro(unit_interval, m)
        print np.linalg.norm(yL - yY, np.inf)
        
        plt.figure()
        plt.plot(unit_interval, yL)
        plt.plot(unit_interval, yY)
        plt.show()

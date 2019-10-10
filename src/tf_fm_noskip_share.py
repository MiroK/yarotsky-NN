import tensorflow as tf


def x2_approx_skip(x, m):
    '''Skip connection, share composition and gs weights'''
    # Input to first layer
    A0 = tf.Variable(tf.truncated_normal(shape=[1, 3], stddev=0.1))
    b0 = tf.Variable(tf.constant(0.1, shape=[3]))

    params = {'A0': A0, 'b0': b0}
    
    # Collapse y1 to get gs
    Ac = tf.Variable(tf.truncated_normal(shape=[3, 1], stddev=0.1))
    bc = tf.Variable(tf.constant(0.1, shape=[1]))
    # Record
    params['Ac'] = Ac
    params['bc'] = bc

    Ag = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1))
    bg = tf.Variable(tf.constant(0.1, shape=[3]))
    # Record
    params['Ag'] = Ag
    params['bg'] = bg
                  
    # ----------------------------------------------------------------

    y1 = tf.add(tf.matmul(x, A0), b0)
    y1 = tf.nn.relu(y1)

    gs = []
    for s in range(1, m):
        # Collapse this g
        gs.append(tf.add(tf.matmul(y1, Ac), bc))
                  
        # Make composition another g
        y1 = tf.add(tf.matmul(y1, Ag), bg)
        y1 = tf.nn.relu(y1)
        
    # Collapse y1 to get gs
    gs.append(tf.add(tf.matmul(y1, Ac), bc))
        
    # Collect
    y = tf.concat([x] + gs, axis=1)

    # And the last layer
    A = tf.Variable(tf.truncated_normal(shape=[1+len(gs), 1], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[1]))
    # Record
    params['Aout'] = A
    params['bout'] = b
    
    y = tf.add(tf.matmul(y, A), b)

    return y, params 

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    m = 12
    
    def predict(sess, NN, x, x_values):
        y_ = [sess.run(NN, feed_dict={x: np.array([[xi]])}) for xi in x_values]
        return np.array(y_).flatten()

    x = tf.placeholder(tf.float32, [None, 1])
    NN, params = x2_approx_skip(x, m=m)

    # Before starting, initialize the variables
    init = tf.initialize_all_variables()

    # Composition
    Ag = np.array([[2, -4, 2],
                  [2, -4, 2],
                  [2, -4, 2]]).T
    bg = np.array([0, -0.5, -1])

    # Narrowing for finalizing gs
    Ac = np.array([[2, -4, 2]]).T
    bc = np.array([0])

    # Launch the graph.
    with tf.Session() as sess:
        sess.run(init)
        # Set the variables as in numpy
        # For first layer
        sess.run(params['A0'].assign(np.array([[1., 1., 1.]])))
        sess.run(params['b0'].assign(np.array([0, -0.5, -1.])))

        sess.run(params['Ac'].assign(Ac))
        sess.run(params['bc'].assign(bc))

        sess.run(params['Ag'].assign(Ag))
        sess.run(params['bg'].assign(bg))
        # Collapse last
        Aout = np.array([np.r_[1., -1./(2.**(2*np.arange(1, m+1)))]]).T
        bout = np.array([0])

        sess.run(params['Aout'].assign(Aout))
        sess.run(params['bout'].assign(bout))

        #for w in tf.trainable_variables():
        #    print sess.run(w)
        
        unit_interval = np.linspace(0, 1, 1001)
    
        plt.figure()
        plt.plot(unit_interval, predict(sess, NN, x, unit_interval))            
        plt.show()

import tensorflow as tf


def fourier_series(x, m):
    '''Shallow network that is c_k*sin(k*x+b)'''
    # Input to first layer
    A0 = tf.Variable(tf.truncated_normal(shape=[1, m], stddev=0.1, dtype=tf.float64))
    b0 = tf.Variable(tf.constant(0.1, shape=[m], dtype=tf.float64))

    # Collapse final
    A1 = tf.Variable(tf.truncated_normal(shape=[m, 1], stddev=0.1, dtype=tf.float64))
    b1 = tf.Variable(tf.constant(0.1, shape=[1], dtype=tf.float64))

    params = {'A0': A0, 'b0': b0, 'A1': A1, 'b1': b1}

    y1 = tf.add(tf.matmul(x, A0), b0)
    y1 = tf.math.sin(y1)        
    y1 = tf.add(tf.matmul(y1, A1), b1)

    return y1, params 


def deep_fourier(x, hidden_layer_dims):
    '''Layered Fourier series'''
    # R^1 to R^1
    layer_dims = [1] + hidden_layer_dims + [1]
    assert x.shape[1] == layer_dims[0]

    layer_i = x  # The previous one
    # Buld graph for all up to last hidden
    for dim_i, dim_o in zip(layer_dims[:-2], layer_dims[1:]):
        # Random weights
        weights = tf.Variable(tf.truncated_normal(shape=[dim_i, dim_o], stddev=0.1, dtype=tf.float64))
        # NOTE: for fitting it seems better to have bias as constant
        biases = tf.Variable(tf.constant(0.1, shape=[dim_o], dtype=tf.float64))
        
        layer_o = tf.add(tf.matmul(layer_i, weights), biases)
        # With ReLU activation
        layer_o = tf.math.sin(layer_o)

        layer_i = layer_o

    # Now from hidden to output
    dim_i, dim_o = layer_dims[-2:]
    weights = tf.Variable(tf.truncated_normal([dim_i, dim_o], stddev=0.1, dtype=tf.float64))
    biases = tf.Variable(tf.constant(0.1, shape=[dim_o], dtype=tf.float64))
    
    layer_o = tf.add(tf.matmul(layer_i, weights), biases)

    return layer_o

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from common import predict
    from math import pi
    import numpy as np

    m = 16
    
    x = tf.placeholder(tf.float32, [None, 1])
    # -------------------------------------------------------------------------
    # Check the idea:
    # -------------------------------------------------------------------------
    if False:
        NN, params = fourier_series(x, m=m)

        # Check the idea
        init = tf.initialize_all_variables()

        A0 = np.array([[0, pi, 0, 3*pi]])
        b0 = np.array([0, 0, 0, -1])
        A1 = np.array([[0, 1, 0, 1]]).T
        b1 = np.array([0])

        # Launch the graph.
        with tf.Session() as sess:
            # Set the variables as in numpy
            sess.run(params['A0'].assign(A0))
            sess.run(params['b0'].assign(b0))
            
            sess.run(params['A1'].assign(A1))
            sess.run(params['b1'].assign(b1))
            xx = np.linspace(0, pi, 1001)
        
            plt.figure()
            plt.plot(xx, predict(sess, NN, x, xx))     
            plt.plot(xx, np.sin(pi*xx)+np.sin(3*pi*xx-1))       
            plt.show()

    # -------------------------------------------------------------------------
    # Can we learn something
    # -------------------------------------------------------------------------
    x = tf.placeholder(tf.float64, [None, 1])

    # NN, params = fourier_series(x, m)
    # NOTE: with hidden layers the interpretation of first layer size 
    # is the number of terms in the series
    # NN, _ = fourier_series(x, m)
    
    NN = deep_fourier(x, hidden_layer_dims=[m, m])
    # y = NN(x)
    y = tf.placeholder(tf.float64, [None, 1])

    # The loss functional
    loss = tf.reduce_sum(tf.abs(NN - y))  # reduce_[sum, mean]
    
    learning_rate = 1E-3
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # FIXME: Make training depend on m
    training_epochs = 10000
    batch_size = 1000
    display_step = 200

    init = tf.global_variables_initializer()

    # f = lambda x: np.minimum(np.maximum(x-0.25, 0), np.maximum(0.75-x, 0))
    f = lambda x: np.maximum(x-0.5, 0)

    x_data = np.vstack([0, 1, np.random.rand(1000000, 1)])
    y_data = f(x_data)
    idx = np.arange(len(x_data))
    
    with tf.Session() as session:
        session.run(init)

        for step in range(training_epochs):
            
            idx_ = np.random.choice(idx, batch_size)
            session.run(train, feed_dict={x: x_data[idx_], y: y_data[idx_]})
        
            if step % display_step == 0:
                x_test = np.random.rand(1, 1).astype(np.float32)
                y_test = x_test**2

                error = session.run(loss, feed_dict={x: x_test, y: y_test})
            
                print('At step %d error %g' % (step, error))
    
        x_data = np.array([np.linspace(0, 1, 10000)]).T
        y_data = f(x_data)

        plt.figure()
        plt.plot(x_data, y_data)
        plt.plot(x_data, session.run(NN, feed_dict={x: x_data}))
        plt.show()
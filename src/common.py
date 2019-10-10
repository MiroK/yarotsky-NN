import tensorflow as tf
import numpy as np


def sup_norm(x):
    '''l^oo norm'''
    return np.linalg.norm(x, np.inf)

def predict(sess, NN, x, x_values):
    '''eval NN(x) in this session using x_values'''
    y_ = [sess.run(NN, feed_dict={x: np.array([[xi]])}) for xi in x_values]
    return np.array(y_).flatten()


def train(session, get_NN, verbose=True):
    '''Train NN to approx x^2'''
    x = tf.placeholder(tf.float32, [None, 1])
    NN, params = get_NN(x)

    # y = NN(x)
    y = tf.placeholder(tf.float32, [None, 1])

    # The loss functional
    loss = tf.reduce_mean(tf.square(NN - y))  # reduce_[sum, mean]
    
    learning_rate = 1E-4
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    training_epochs = 20000
    batch_size = 750
    display_step = 200

    # Before starting, initialize the variables
    init = tf.global_variables_initializer()

    # Launch the graph.
    session.run(init)

    for step in range(training_epochs):
        x_data = np.vstack([0., 1., np.random.rand(batch_size-2, 1)])
        x_data = x_data.astype(np.float32)
        # Always have bcs
        y_data = x_data**2  # This is what we learn
        session.run(train, feed_dict={x: x_data, y: y_data})
    
        if verbose and step % display_step == 0:
            x_test = np.random.rand(1, 1).astype(np.float32)
            y_test = x_test**2

            error = session.run(loss, feed_dict={x: x_test, y: y_test})
        
            print('At step %d error %g' % (step, error))
    # Bound
    NN_predict = lambda x0, s=session, x=x, NN=NN: predict(s, NN, x, x0)

    return NN_predict

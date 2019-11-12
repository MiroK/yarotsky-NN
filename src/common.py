import tensorflow as tf
import numpy as np


def sup_norm(x):
    '''l^oo norm'''
    return np.linalg.norm(x, np.inf)

def predict(sess, NN, x, x_values):
    '''eval NN(x) in this session using x_values'''
    y_ = [sess.run(NN, feed_dict={x: np.array([[xi]])}) for xi in x_values]
    return np.array(y_).flatten()


def train(session, m, get_NN, verbose=True, points='random', penalty=0):
    '''
    Train NN to approx x^2. Here we use a **full** gradient in gradient
    descent, so no minibatch.
    '''
    x = tf.placeholder(tf.float64, [None, 1])
    NN, params = get_NN(x)

    # Count
    ndofs = sum(np.prod(var.shape) for var in params.values())

    # y = NN(x)
    y = tf.placeholder(tf.float64, [None, 1])

    # The loss functional
    loss = tf.reduce_mean(tf.square(NN - y))  # reduce_[sum, mean]
    if penalty > 0:
        print('Regularization')
        traces = sum(tf.linalg.trace(tf.matmul(tf.linalg.transpose(p), p))
                     for p in params.values() if p.shape == (3, 3))
        loss = loss + tf.constant(penalty, shape=[1], dtype=tf.float64)*traces
    
    learning_rate = 1E-3
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Make training depend on m
    training_epochs = int(10000*max(1, np.log2(m)))
    batch_size = min(500, 200*m)
    display_step = 200

    # tf.set_random_seed(1234)
    # Before starting, initialize the variables
    init = tf.global_variables_initializer()

    # Launch the graph.
    session.run(init)

    if points == 'random':
        x_data = np.vstack([0, 1, np.random.rand(1000000, 1)])
    # Points for building best P0 approximation
    else:
        assert points == 'dg0'
        print('Random points')
        x_data = np.linspace(0, 1, 1000001)
        x_data = x_data.reshape((-1, 1))
        x_data = np.sqrt(0.5*(x_data[:-1] + x_data[1:]))
        # Add bcs
        x_data = np.row_stack([0, x_data, 1])
        
    y_data = x_data**2
    idx = np.arange(len(x_data))
    for step in range(training_epochs):
        
        idx_ = np.random.choice(idx, batch_size)
        session.run(train, feed_dict={x: x_data[idx_], y: y_data[idx_]})
    
        if verbose and step % display_step == 0:
            x_test = np.random.rand(1, 1).astype(np.float32)
            y_test = x_test**2

            error = session.run(loss, feed_dict={x: x_test, y: y_test})
        
            print('At step %d error %g' % (step, error))
    # Bound
    NN_predict = lambda x0, s=session, x=x, NN=NN: predict(s, NN, x, x0)

    return NN_predict, ndofs


def train_minibatch(session, m, get_NN, verbose=True):
    '''
    Train NN to approx x^2. Here we use a incremental updates of 
    gradient computed using minibatches.
    '''
    x = tf.placeholder(tf.float64, [None, 1])
    NN, params = get_NN(x)
    # Count
    ndofs = sum(np.prod(var.shape) for var in params.values())

    # y = NN(x)
    y = tf.placeholder(tf.float64, [None, 1])

    # The loss functional
    loss = tf.reduce_mean(tf.square(NN - y))  # reduce_[sum, mean]
    
    learning_rate = 1E-4
    optimizer = tf.train.AdamOptimizer(learning_rate)
    minimize = optimizer.minimize(loss)

    # Make training depend on m
    training_epochs = int(10000*max(1, np.log2(m)))
    batch_size = min(500, 200*m)
    minibatch_size = batch_size
    display_step = 200

    weights = tf.trainable_variables() 
    # Want to store gradient wrt each weight
    gradients = [tf.Variable(w.initialized_value(), trainable=False) for w in weights]
    # Zero before minibatch loop
    zero_gradients  = [g.assign(tf.zeros_like(g)) for g in gradients]

    compute_gradients = optimizer.compute_gradients(loss)
    # Update with the computed
    update_gradients = [gradients[i].assign_add(grad[0])
                       for i, grad in enumerate(compute_gradients)]
    # After minibatch we make the step
    apply_gradients = optimizer.apply_gradients([(gradients[i], w) for i, w in enumerate(weights)])

    # Before starting, initialize the variables
    init = tf.global_variables_initializer()

    # Launch the graph.
    session.run(init)

    x_data = np.vstack([0, 1, np.random.rand(1000000, 1)])
    y_data = x_data**2
    idx = np.arange(len(x_data))

    for step in range(training_epochs):
        
        idx_ = np.random.choice(idx, batch_size)
        # Zero gradient before minibatch
        session.run(zero_gradients)
        
        for j in range(len(idx_)/minibatch_size):
            f, l = j*minibatch_size, (j+1)*minibatch_size
            session.run(update_gradients,
                        feed_dict={x: x_data[idx_[f:l]], y: y_data[idx_[f:l]]})
            
        session.run(apply_gradients)
    
        if verbose and step % display_step == 0:
            x_test = np.random.rand(1, 1).astype(np.float32)
            y_test = x_test**2

            error = session.run(loss, feed_dict={x: x_test, y: y_test})
        
            print('At step %d error %g' % (step, error))
    # Bound
    NN_predict = lambda x0, s=session, x=x, NN=NN: predict(s, NN, x, x0)

    return NN_predict, ndofs

import tensorflow as tf
import numpy as np


def hat(x, points):
    '''Neural net encoding of a hat function
    
    min(rho((x-x0)/(x1-x0)), rho((x2-x)/(x2-x1)))
    
    '''
    x0, x1, x2 = points
    assert x0 < x1 < x2

    A = tf.constant(np.array([[1./(x1-x0),  -1./(x2-x1)]]),
                    dtype=tf.float64)
    b = tf.constant(np.array([[-x0/(x1-x0), x2/(x2-x1)]]),
                    dtype=tf.float64)
    # Create rhos
    y1 = tf.add(tf.matmul(x, A), b)
    y1 = tf.nn.relu(y1)

    # Now we want to create abs of rhos since min(a, b) equals
    # (a + b - abs(b-a))/2
    # The first step is abs(x) = rho(x) + rho(-x)
    #                             rho(b-a) + rho(a-b)
    A = tf.constant(np.array([[1., -1], [-1, 1]]), dtype=tf.float64)
    y2 = tf.nn.relu(tf.matmul(y1, A))
    # Narrow to scalar
    A = tf.constant(np.array([[1., 1]]).T, dtype=tf.float64)
    y3 = tf.matmul(y2, A)  # abs

    y = tf.concat([y1, y3], axis=1)
    A = tf.constant(np.array([[0.5, 0.5, -0.5]]).T, dtype=tf.float64)

    out = tf.matmul(y, A)
    
    return out


def p1_net(x, mesh):
    '''A neural network representing piecesewise linear functions in 1d'''
    # It is just a sum of hats and the coefficients are the only parameters
    # of the network
    assert len(mesh) > 2
    hats = []
    for i_mid in range(1, len(mesh)-1):
        hat_i = hat(x, mesh[i_mid-1:i_mid+2])
        hats.append(hat_i)

    y = tf.concat(hats, axis=1)
    # Final representing the sum
    A = tf.Variable(tf.truncated_normal(shape=[len(hats), 1], stddev=0.1, dtype=tf.float64))
    # A are the only parameters of the net
    return tf.matmul(y, A), A

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from common import predict
    
    x = tf.placeholder(tf.float64, [None, 1])

    # NN = hat(x, (0.25, 0.5, 0.75))
    mesh = np.linspace(0, 1, 11)
    f, A = p1_net(x, mesh)

    grad_f = tf.gradients(f, x)[0]
    # Before starting, initialize the variables
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        
        x_ = np.linspace(0, 1, 1001).astype(np.float64)
        y_ = predict(sess, f, x, x_)

        dy_dx_ = predict(sess, grad_f, x, x_)

    plt.figure()
    plt.plot(x_, y_)
    plt.plot(x_, dy_dx_)
    plt.show()

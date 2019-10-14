import tensorflow as tf
import numpy as np


def hat(x, points):
    '''Neural net encoding of a hat function
    
    min(rho((x-x0)/(x1-x0)), rho((x2-x)/(x2-x1)))
    
    '''
    x0, x1, x2 = points

    if x0 is None:
        assert x1 < x2
        A = np.array([[0.,  -1./(x2-x1)]])
        b = np.array([[0., x2/(x2-x1)]])
        reduce_ = np.array([[1., 1]]).T
        combine_ = np.array([[0.5, 0.5, 0.5]]).T
    elif x2 is None:
        assert x0 < x1
        A = np.array([[1./(x1-x0),  0]])
        b = np.array([[-x0/(x1-x0), 0]])
        reduce_ = np.array([[1., -1]]).T
        combine_ = np.array([[0.5, 0.5, 0.5]]).T
    else:
        assert x0 < x1 < x2
        A = np.array([[1./(x1-x0),  -1./(x2-x1)]])
        b = np.array([[-x0/(x1-x0), x2/(x2-x1)]])
        reduce_ = np.array([[1., 1]]).T
        combine_ = np.array([[0.5, 0.5, -0.5]]).T
        
    A = tf.constant(A, dtype=tf.float64)
    b = tf.constant(b, dtype=tf.float64)
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
    A = tf.constant(reduce_, dtype=tf.float64)
    y3 = tf.matmul(y2, A)  # abs

    y = tf.concat([y1, y3], axis=1)
    A = tf.constant(combine_, dtype=tf.float64)

    out = tf.matmul(y, A)
    
    return out


def p1_net(x, mesh, dir_bcs=True):
    '''A neural network representing piecesewise linear functions in 1d'''
    if not dir_bcs:
        mesh = [None] + list(mesh) + [None]
    # It is just a sum of hats and the coefficients are the only parameters
    # of the network
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

    dir_bcs = False
    x = tf.placeholder(tf.float64, [None, 1])

    mesh = np.linspace(0, 1, 101)
    # Set coefs to interpolant
    if not dir_bcs:
        A_values = np.array([np.cos(2*np.pi*mesh)]).T
    else:
        A_values = np.array([np.cos(2*np.pi*mesh[1:-1])]).T
        
    f, A = p1_net(x, mesh, dir_bcs=dir_bcs)

    grad_f = tf.gradients(f, x)[0]
    # Before starting, initialize the variables
    init = tf.initialize_all_variables()
    
    x_ = np.linspace(0, 1, 1001).astype(np.float64)
    x_mid = 0.5*(x_[:-1] + x_[1:])
    with tf.Session() as sess:
        sess.run(init)
        sess.run(A.assign(A_values))
        
        y_ = predict(sess, f, x, x_)

        # dy_dx_ = predict(sess, grad_f, x, x_mid)

    plt.figure()
    plt.plot(x_, y_)
    # plt.plot(x_mid, dy_dx_)
    # plt.plot(x_mid, np.cos(2*np.pi*x_mid)*2*np.pi)
    plt.show()

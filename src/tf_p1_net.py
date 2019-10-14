from scipy.sparse import diags
import tensorflow as tf
import numpy as np


def hat(x, points):
    '''Neural net encoding of a hat function
    
    min(rho((x-x0)/(x1-x0)), rho((x2-x)/(x2-x1)))
    
    '''
    x0, x1, x2 = points
    # \
    #  \_______
    if x0 is None:
        assert x1 < x2
        A = np.array([[0.,  -1./(x2-x1)]])
        b = np.array([[0., x2/(x2-x1)]])
        reduce_ = np.array([[1., 1]]).T
        combine_ = np.array([[0.5, 0.5, 0.5]]).T
    #        /
    # ______/
    elif x2 is None:
        assert x0 < x1
        A = np.array([[1./(x1-x0),  0]])
        b = np.array([[-x0/(x1-x0), 0]])
        reduce_ = np.array([[1., -1]]).T
        combine_ = np.array([[0.5, 0.5, 0.5]]).T
    #      /\
    # ____/ `\____
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
    #                            rho(b-a) + rho(a-b)
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


def mass_matrix(mesh, dir_bcs):
    '''coef.dot(M.dot(coef)) is the L2 norm^2'''
    h = np.diff(mesh)
    left, right = h[:-1], h[1:]

    if dir_bcs:
        main = left/3. + right/3.
        off = right[:-1]/6.
    else:
        main = np.r_[left[0]/3., left/3. + right/3., right[-1]/3]
        off = h/6.

    A = diags([off, main, off], [-1, 0, 1])

    return A


def stiffness_matrix(mesh, dir_bcs):
    '''coef.dot(M.dot(coef)) is the H1 semi norm^2'''
    diff = np.diff(mesh)
    
    if dir_bcs:
        # xi - xi-1
        # 1/h{i} + 1/h{i+1}
        main = 1./diff[:-1] + 1./diff[1:]
        off = -1./diff[1:-1]
    else:
        main = np.r_[1./diff[0], 1./diff[:-1] + 1./diff[1:], 1./diff[-1]]
        off = -1./diff

    A = diags([off, main, off], [-1, 0, 1])

    return A
        
# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from common import predict

    dir_bcs = False
    x = tf.placeholder(tf.float64, [None, 1])

    mesh = np.linspace(0, 1, 101)
    # mesh = np.r_[0, np.random.rand(1000), 1]
    # mesh = np.sort(mesh)
    #print mesh

    A = mass_matrix(mesh, dir_bcs) # stiffness_matrix(mesh, dir_bcs)

    # f_ = lambda x: x*(x-1)
    # f_ = lambda x: x*(x-1)+x
    f_ = lambda x: np.sin(2*np.pi*x)

    # Set coefs to interpolant
    if not dir_bcs:
        A_values = np.array([f_(mesh)]).T
    else:
        A_values = np.array([f_(mesh[1:-1])]).T
        
    f, weights = p1_net(x, mesh, dir_bcs=dir_bcs)

    grad_f = tf.gradients(f, x)[0]
    # Before starting, initialize the variables
    init = tf.initialize_all_variables()
    
    x_ = np.linspace(0, 1, 1001).astype(np.float64)
    x_mid = 0.5*(x_[:-1] + x_[1:])
    with tf.Session() as sess:
        sess.run(init)
        sess.run(weights.assign(A_values))
        
        y_ = predict(sess, f, x, x_)

        dy_dx_ = predict(sess, grad_f, x, x_mid)

    plt.figure()
    plt.plot(x_, y_)
    plt.plot(x_mid, dy_dx_)
    plt.plot(x_mid, np.cos(2*np.pi*x_mid)*2*np.pi)
    plt.show()

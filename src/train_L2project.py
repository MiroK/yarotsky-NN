from tf_p1_net import mass_matrix, p1_net
from common import predict
import tensorflow as tf
import numpy as np


def target(x):
    '''What we want to approximate'''
    return np.sin(2*np.pi*x) + np.cos(3*np.pi*x)


optimum = 8/(5*np.pi) - 1


def project(session, n, use_preconditioner=False):
    '''Minimize inner(u, u)*dx - 2*inner(f, u)*dx'''
    # Here f is only a P1 approximation of f
    dir_bcs = False
    mesh = np.linspace(0, 1, n)

    x = tf.placeholder(tf.float64, [len(mesh), 1])

    f, weights = p1_net(x, mesh, dir_bcs)

    M = mass_matrix(mesh, dir_bcs).todense()

    if use_preconditioner:
        D = np.diag(np.diagonal(M))
        M = D.dot(M.dot(D))

    # rhs
    Mf = 2*M.dot(target(mesh))

    tfM = tf.constant(M, dtype=tf.float64)
    # Let's Mf
    optimum_ = (-0.5*np.inner(target(mesh), Mf))[0, 0]

    Mf = tf.constant(Mf, dtype=tf.float64)

    inner_f_u = tf.matmul(Mf, f)
    inner_u_u = tf.matmul(tf.transpose(f), tf.matmul(tfM, f))

    loss = inner_u_u - inner_f_u

    learning_rate = 1E-1
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Make training depend on m
    training_epochs = 200

    # Before starting, initialize the variables
    init = tf.global_variables_initializer()

    # Launch the graph.
    cvrg = []
    session.run(init)
    x_ = mesh.reshape((-1, 1))
    for step in range(training_epochs):
        session.run(optimizer, feed_dict={x: x_})
        cvrg.append(session.run(loss, feed_dict={x: x_})[0, 0])

    return x, f, np.array(cvrg) - optimum

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ns = (8, 16, 32, 64)#, 128, 256):
  
    fig, ax = plt.subplots()
    
    mesh = np.linspace(0, 1, 1024)
    ax.plot(mesh, target(mesh), label='true')

    histories = []
    for n in ns:
        with tf.Session() as session:
            x, f, cvrg = project(session, n, use_preconditioner=False)
            histories.append(cvrg)
            
            print n, cvrg[-1]
            
            fx = np.array([])
            for i in range(len(mesh)/n):
                fx = np.r_[fx,
                           session.run(f, feed_dict={x: mesh[i*n:(i+1)*n].reshape((n, 1))}).flatten()]
            ax.plot(mesh, fx, label=str(n))
            # plt.semilogy(np.arange(1, len(cvrg)+1), cvrg, label=str(n))

    # Some heuristic to determine convergence
    window_avg = lambda x: np.convolve(x, np.ones((20,))/20, mode='valid')
    stop_iter = lambda x: np.where(np.abs(np.diff(window_avg(x)))/np.min(np.abs(x)) < 5E-6)[0][0]

    stops = map(stop_iter, histories)

    plt.legend()

    fig, ax = plt.subplots()
    ax_right = ax.twinx()
    ax.set_yscale('log')
    for i, n in enumerate(ns):
        ax.plot(histories[i], label=str(n))
    
    errors = [histories[i][stop_i] for i, stop_i in enumerate(stops)]

    ax.plot(stops, errors, linestyle='dashed')
    plt.legend()
        
    plt.show()

from tf_p1_net import mass_matrix, stiffness_matrix, p1_net
import matplotlib.pyplot as plt
from common import predict
import tensorflow as tf
import numpy as np


def solution(x):
    '''u in -u + u'' = f'''
    return np.cos(2*np.pi*x) + np.cos(4*np.pi*x)


def rhs(x):
    '''f in -u + u'' = f'''
    return ((2*np.pi)**2 + 1)*np.cos(2*np.pi*x) + (1 + (4*np.pi)**2)*np.cos(4*np.pi*x)


# optimum = -13*np.pi**2/2

def poisson(session, n, use_preconditioner):
    # Minimize inner(u', u')*dx - 2*inner(f, u)*dx
    # Here f is only a P1 approximation of f
    dir_bcs = False
    mesh = np.linspace(0, 1, n)

    use_scipy = False

    x = tf.placeholder(tf.float64, [len(mesh), 1])

    f, weights = p1_net(x, mesh, dir_bcs)

    M = mass_matrix(mesh, dir_bcs).todense()
    A = stiffness_matrix(mesh, dir_bcs).todense()

    tfM, tfA = tf.constant(M, dtype=tf.float64), tf.constant(A, dtype=tf.float64)

    Mf = 2*M.dot(rhs(mesh))
    true = solution(mesh)
    # optimum_ = (-np.inner(true, A.dot(true)))[0, 0]

    Mf = tf.constant(Mf, dtype=tf.float64)

    if use_preconditioner:
        Ainv = tf.constant(np.linalg.inv(A+M), dtype=tf.float64)
    else:
        Ainv = tf.constant(np.eye(len(mesh)), dtype=tf.float64)
    
    inner_f_u = tf.matmul(tf.matmul(Mf, Ainv), f)
    inner_gu_gu = tf.matmul(tf.matmul(tf.transpose(f), Ainv), tf.matmul(tfA, f))
    inner_u_u = tf.matmul(tf.matmul(tf.transpose(f), Ainv), tf.matmul(tfM, f))

    loss = inner_gu_gu + inner_u_u - inner_f_u

    learning_rate = 1.0

    if use_scipy:
        # This one uses exact gradient and is done in one step
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='cg')
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Make training depend on m
    training_epochs = 1 if use_scipy else 200

    # Before starting, initialize the variables
    init = tf.global_variables_initializer()

    # Launch the graph.
    cvrg = []
    session.run(init)
    x_ = mesh.reshape((-1, 1))
    for step in range(training_epochs):

        if use_scipy:
            optimizer.minimize(session, feed_dict={x: x_})
        else:
            session.run(optimizer, feed_dict={x: x_})
            
        cvrg.append(session.run(loss, feed_dict={x: x_})[0, 0])

        print step, cvrg[-1]#, optimum, cvrg[-1] - optimum_
            
    cvrg = np.array(cvrg)        
    #cvrg, cvrg_ = cvrg - optimum, cvrg - optimum_
    
    return x, f, cvrg  # cvrg, cvrg_

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

  
    ns = (8, 16, 32, 64, 128, 256, 512)
    ns = ns
    
    mesh = np.linspace(0, 1, 1024)

    fig, ax = plt.subplots()
    true = solution(mesh)
    ax.plot(mesh, true, label='true')

    A, M = stiffness_matrix(mesh, False), mass_matrix(mesh, False)
    AM = A + M
    
    histories, errors = [], []
    for n in ns:
        with tf.Session() as session:
            x, f, cvrg = poisson(session, n, use_preconditioner=True)
            histories.append(cvrg)

            fx = np.array([])
            for i in range(len(mesh)/n):
                fx = np.r_[fx,
                           session.run(f, feed_dict={x: (mesh[i*n:(i+1)*n]).reshape((-1, 1))}).flatten()]
            ax.plot(mesh, fx, label=str(n))

            e = true - fx
            errors.append(np.sqrt(np.sum(e*(AM.dot(e)))))
            if len(errors) > 1:
                e1, e0 = errors[-2:]
                rate = np.log(e1/e0)/np.log(2)
            else:
                rate = np.nan

            print '\t', errors[-1], rate
    plt.legend()

    plt.figure()
    for n, cvrg in zip(ns, histories):
        plt.plot(cvrg, label=str(n))
    plt.legend()
    
    plt.show()

    # FIXME: get error rates of the converged solution in some norm

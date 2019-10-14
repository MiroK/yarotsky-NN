from tf_p1_net import mass_matrix, stiffness_matrix, p1_net
import matplotlib.pyplot as plt
from common import predict
import tensorflow as tf
import numpy as np

# Minimize inner(u, u)*dx - 2*inner(f, u)*dx
# Here f is only a P1 approximation of f
dir_bcs = True
mesh = np.linspace(0, 1, 101)

use_scipy = False

x = tf.placeholder(tf.float64, [len(mesh)-2, 1])

f, weights = p1_net(x, mesh, dir_bcs)

M = mass_matrix(mesh, dir_bcs).todense()
A = stiffness_matrix(mesh, dir_bcs).todense()

tfM, tfA = tf.constant(M, dtype=tf.float64), tf.constant(A, dtype=tf.float64)

# Let's Mf
solution = lambda x: np.sin(2*np.pi*x) + np.sin(3*np.pi*x)
rhs = lambda x: (2*np.pi)**2*np.sin(2*np.pi*x) + (3*np.pi)**2*np.sin(3*np.pi*x)
                                            
Mf = 2*M.dot(rhs(mesh[1:-1]))
# optimum = (-0.5*np.inner(target(mesh), Mf))[0, 0]

Mf = tf.constant(Mf, dtype=tf.float64)

inner_f_u = tf.matmul(Mf, f)
inner_gu_gu = tf.matmul(tf.transpose(f), tf.matmul(tfA, f))

loss = inner_gu_gu - inner_f_u

learning_rate = 1.0

if use_scipy:
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='cg')
else:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# This one uses exact gradient and is done in one step

# Make training depend on m
training_epochs = 1 if use_scipy else 1000

# Before starting, initialize the variables
init = tf.global_variables_initializer()

# Launch the graph.
cvrg = []
with tf.Session() as session:
    session.run(init)
    x_ = mesh[1:-1].reshape((-1, 1))
    for step in range(training_epochs):

        if use_scipy:
            optimizer.minimize(session, feed_dict={x: x_})
        else:
            session.run(optimizer, feed_dict={x: x_})
        
        cvrg.append(session.run(loss, feed_dict={x: x_})[0, 0])
        print step, cvrg[-1]

    plt.figure()
    plt.plot(mesh, solution(mesh))
    plt.plot(x_, session.run(f, feed_dict={x: x_}))

    #plt.figure()
    #plt.semilogy(np.arange(1, len(cvrg)+1), np.abs(cvrg-optimum), 'rx-')
    plt.show()

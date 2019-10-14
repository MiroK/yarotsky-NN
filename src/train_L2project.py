from tf_p1_net import mass_matrix, p1_net
import matplotlib.pyplot as plt
from common import predict
import tensorflow as tf
import numpy as np

# Minimize inner(u, u)*dx - 2*inner(f, u)*dx
# Here f is only a P1 approximation of f
dir_bcs = False
mesh = np.linspace(0, 1, 1001)

x = tf.placeholder(tf.float64, [len(mesh), 1])

f, weights = p1_net(x, mesh, dir_bcs)

M = mass_matrix(mesh, dir_bcs).todense()
tfM = tf.constant(M, dtype=tf.float64)
      
# Let's Mf
target = lambda x: np.sin(2*np.pi*x) + np.cos(3*np.pi*x)
Mf = 2*M.dot(target(mesh))
optimum = (-0.5*np.inner(target(mesh), Mf))[0, 0]

Mf = tf.constant(Mf, dtype=tf.float64)

inner_f_u = tf.matmul(Mf, f)
inner_u_u = tf.matmul(tf.transpose(f), tf.matmul(tfM, f))

loss = inner_u_u - inner_f_u

learning_rate = 1E-1
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Make training depend on m
training_epochs = 100

# tf.set_random_seed(1234)
# Before starting, initialize the variables
init = tf.global_variables_initializer()

# Launch the graph.
cvrg = []
with tf.Session() as session:
    session.run(init)
    x_ = mesh.reshape((-1, 1))
    for step in range(training_epochs):
        session.run(optimizer, feed_dict={x: x_})
        cvrg.append(session.run(loss, feed_dict={x: x_})[0, 0])
        print step, cvrg[-1]

    plt.figure()
    plt.plot(mesh, target(x_))
    plt.plot(mesh, session.run(f, feed_dict={x: x_}))

    plt.figure()
    plt.semilogy(np.arange(1, len(cvrg)+1), np.abs(cvrg-optimum), 'rx-')
    plt.show()

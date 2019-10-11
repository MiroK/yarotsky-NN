import matplotlib.pyplot as plt
from tf_fm_skip_noshare import x2_approx_skip as get_noshare_NN
from tf_fm_skip_share import x2_approx_skip as get_share_NN
from tooth import x2_approx_skip as Yaro
from common import train, sup_norm
from functools import partial
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import argparse, os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Which (template) preconitioner to use
parser.add_argument('-m', type=int, default=1, help='m in f_m approx')
parser.add_argument('-nnets', type=int, default=10, help='Total of nets to train')
parser.add_argument('-architecture', type=str, default='share', choices=['share', 'noshare'])
args = parser.parse_args()

# --------------------------------------------------------------------

m = args.m
get_NN = {'share': get_share_NN, 'noshare': get_noshare_NN}[args.architecture]
# Bound nets to m
Yaro_m = partial(Yaro, m=m)
NN_m = partial(get_NN, m=m)

comm = MPI.COMM_WORLD
nlocal_jobs = args.nnets/comm.size
if comm.rank == comm.size - 1:
    nlocal_jobs = args.nnets - nlocal_jobs*comm.rank

x = np.linspace(0, 1, 20001)
yY = Yaro_m(x)  # Reference we want to beat
eY = sup_norm(x**2 - yY)

local_errors = []
for i in range(nlocal_jobs):
    with tf.Session() as session:
        # Get back the trained net
        NN, ndofs = train(session, NN_m, verbose=True)
        yL = NN(x)

        eL = sup_norm(x**2 - yL)
        # FIXME: How shall we eval the error?
        # print i, 'Learned error', eL, 'vs', eY
        local_errors.append(eL)

errors = comm.gather(local_errors)

if comm.rank == 0:
    errors = sum(errors, [])

    print 'Learned Max/Min/Mean', np.max(errors), np.min(errors), np.mean(errors)
    print 'Yarotsky', eY
    
    plt.figure()
    plt.plot(errors, marker='x', linestyle='dashed')
    plt.plot(eY*np.ones_like(errors))

    plt.figure()
    plt.plot(x, yY, label='Yarotsky')
    plt.plot(x, yL, label='Learned')
    plt.legend()
    
    plt.show()

    # Store
    not os.path.exists('results') and os.mkdir('results')

    np.savetxt('./results/%s_m%d_errors.txt' % (args.architecture, args.m),
               np.r_[eY, errors],
               header='First is Yarotsky, rest is learned. Number of weights %d' % ndofs)

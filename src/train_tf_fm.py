import matplotlib.pyplot as plt

from tf_fm_noskip_noshare import x2_approx_skip as get_NN
from tooth import x2_approx_skip as Yaro
from common import train, sup_norm
from functools import partial
import tensorflow as tf
import numpy as np


m = 3
# Bound nets to m
Yaro_m = partial(Yaro, m=m)
NN_m = partial(get_NN, m=m)

x = np.linspace(0, 1, 20001)
yY = Yaro_m(x)  # Reference we want to beat
eY = sup_norm(x**2 - yY)

for i in range(1):
    with tf.Session() as session:
        # Get back the trained net
        NN = train(session, NN_m, verbose=True)
        yL = NN(x)

        eL = sup_norm(x**2 - yL)
        # FIXME: How shall we eval the error?
        print i, 'Learned error', eL, 'vs', eY

plt.figure()
plt.plot(x, yY, label='Yarotsky')
plt.plot(x, yL, label='Learned')
plt.legend()
plt.show()

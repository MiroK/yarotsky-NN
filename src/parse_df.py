import matplotlib.pyplot as plt 
import numpy as np 
import os

template = './results/df_%d.txt'

fig, ax = plt.subplots()

for n in (4, 8, 16, 32):
    data = np.loadtxt(template % n)

    ax.plot(data[:, 0], data[:, 2], label='N=%d' % n)
ax.plot(data[:, 0], data[:, 1])

# Cosine series approx
x = data[:, 0]
y = 0.5*np.ones_like(x)
for l in range(1, 33):
    cl = 2*(np.sin(l*np.pi)-np.sin(l*np.pi/2))/l/np.pi
    y += cl*np.cos(l*np.pi*x)

plt.plot(x, y, label='series')

plt.legend()
plt.show()
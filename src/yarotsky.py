import numpy as np


def relu(x):
    '''Vectorized RELU'''
    return np.maximum(0, x)


def tooth(x):
    '''On [0, 1]'''
    return 2*relu(x) - 4*relu(x-0.5) + 2*relu(x-1)


def sawtooth(x, s):
    ''' ^^^^ '''
    assert s >= 1

    if s == 1:
        return tooth(x)
    return sawtooth(tooth(x), s-1)


def x2_approx(x, m):
    ''' Approximation of x^2 on [0, 1] using sawtooth functions'''
    y = np.zeros_like(x)
    for s in range(1, m+1):
        y += sawtooth(x, s)/(2**(2*s))

    return x - y

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 300)


    plt.figure()
    plt.plot(x, sawtooth(x, 4))
    plt.show()


    exit()
    y0 = x**2
    
    # Error-ish
    ms = np.arange(2, 11)
    errors = []
    for m in ms:
        y = x2_approx(x, m)
        e = np.linalg.norm(y0 - y, np.inf)
        
        print m, e
        errors.append(e)

    c0, c1 = np.polyfit(ms, np.log2(errors), deg=1)
    
    plt.figure()
    plt.semilogy(ms, errors, basey=2, linestyle='none', marker='o')
    plt.semilogy(ms, 2**(c0*ms + c1), basey=2)
                            
    plt.show()
        

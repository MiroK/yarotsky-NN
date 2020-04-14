import numpy as np


def relu(x):
    '''Vectorized RELU'''
    return np.maximum(0, x)


def hat(x, shift):
    '''Piecewise linear f in H^1_0 with max at shift'''
    return np.minimum(relu(x)/shift, relu(1-x)/(1-shift))


def hat_basis(nlevels, shift):
    '''Basis of hat, hat o hat, hat o hat o ... of shifted'''
    f = lambda x, shift=shift: hat(x, shift)
    
    basis = [f]
    while nlevels > 1:
        nlevels -= 1
        basis.append(lambda x, f=f, fk=basis[-1]: f(fk(x)))

    return basis

import numpy as np


def relu(x):
    '''Vectorized RELU'''
    return np.maximum(0, x)


def tooth(x, shift=0.5):
    '''On [0, 1] with a kink at shift. Defaults to Yarotsky'''
    # NOTE: here we use 2 relus as opposed to Yerotsky 3-ReLU construction
    return (1./shift)*relu(x) + (1/(shift*(shift-1)))*relu(x-shift)# + 2*relu(x-1)


def sawtooth(x, s, shift=0.5):
    ''' ^^^^ Composition'''
    assert s >= 1

    if s == 1:
        return tooth(x, shift)
    return sawtooth(tooth(x, shift), s-1)


def x2_approx(x, m, shift=0.5):
    ''' With shift 0.5 this is approximation of x^2 on [0, 1] using sawtooth functions'''
    y = np.zeros_like(x)
    for s in range(1, m+1):
        # NOTE: we use these weights that Yarotsky made for shift == 0.5
        # also for the shifted sawtooth
        y += sawtooth(x, s, shift)/(2**(2*s))
    return y

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 10000)

    shift = 0.125
    # We claim that based on the shift what the x2_approx above computes
    # is the approximation of a function f which is
    # f(0) = 0
    # f(1) = 0
    # f has maximum in shift with value 0.25
    # on intervals (0, shift) and (shift, 1) is quadratic
    y0 = np.where(x < shift,
                  -0.25*(x-shift)**2/(0-shift)**2 + 0.25,
                  -0.25*(x-shift)**2/(1-shift)**2 + 0.25)
    # To verify the claim we check the (exponential) decay of the approximation
    # error
    # Error-ish
    ms = np.arange(2, 11)
    errors = []
    for m in ms:
        y = x2_approx(x, m, shift)
        e = np.linalg.norm(y0 - y, np.inf)
        errors.append(e)

    c0, c1 = np.polyfit(ms, np.log2(errors), deg=1)
    
    plt.figure()
    plt.semilogy(ms, errors, basey=2, linestyle='none', marker='o')
    plt.semilogy(ms, 2**(c0*ms + c1), basey=2)
    plt.xlabel('nlevels')
    plt.ylabel('error')
    plt.show()

import numpy as np
# Yarotsky on [-1, 1]
#
#

def relu(x):
    '''Vectorized RELU'''
    return np.maximum(0, x)


def hat(x, x0, a=0, b=1):
    '''    /\              y = 1      
    ------/  \-------          0
         a x0 b
    '''
    a, b = float(a), float(b)
    assert a < x0 < b
    # To reference domain which is (0, 1)
    x = (x-a)/(b-a)
    x0 = (x0-a)/(b-a)
    return relu(x)/x0 - relu(x-x0)/x0/(1-x0) + relu(x-1)/(1-x0)


def odd_tooth(x, x0=-0.5, x1=0.5):
    '''v^: [-1, 1] -> [-1, 1]
           /\            y = 1
    ---\  /  \-------        0
        \/                  -1
     -1 x0  x1
    '''
    assert -1 < x0 < x1 < 1
    return -1.0*hat(x, x0=x0, a=-1, b=0) + 1.0*hat(x, x0=x1, a=0, b=1)


def even_tooth(x, x0=-2/3., x1=2/3.):
    '''^v^: [-1, 1] -> [-1, 1]
        /\    /\         y = 1
    ---/  \  /  \---       = 0
           \/                -1
      -1 x0 0  x1 1
    '''
    assert -1 < x0 < x1 < 1
    
    return hat(x, x0=x0, a=-1, b=x0+(x0+1)) - hat(x, x0=0, a=x0+(x0+1), b=x1-(1-x1))+ hat(x, x0=x1, a=x1-(1-x1), b=1)


def odd_sawtooth(x, s):
    ''' v^ Composition'''
    assert s >= 1
    if s == 1:
        return odd_tooth(x)
    return odd_sawtooth(odd_tooth(x), s-1)


def even_sawtooth(x, s):
    ''' ^v^ Composition'''
    assert s >= 1
    if s == 1:
        return even_tooth(x)
    return even_sawtooth(even_tooth(x), s-1)


def eval_hierarchy(x, depth, basis):
    '''Compositions of functions in basis @ x'''
    values = [[f(x) for f in basis]]
    for k in range(1, depth):
        values.append([f(y) for y in values[-1] for f in basis])
    return values

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from itertools import chain

    # Plot functions for each level
    if False:
        x = np.linspace(-1, 1, 10001)

        m = 4

        hierarchy = eval_hierarchy(depth=m, basis=(odd_tooth, even_tooth), x=x)
        basis_names = ('O', 'E')
        level_names = [['O', 'E']]
        for level in hierarchy:
            fig, axarr = plt.subplots(len(level)/2, 2, sharex=True, sharey=True)
            axarr = axarr if len(axarr.shape) == 1 else chain(*axarr)

            names = level_names.pop()

            for ax, f, name in zip(axarr, level, names):
                ax.plot(x, f)
                ax.set_title(name)

            level_names.append([' o '.join([f, prev]) for prev in names for f in basis_names])

            
    # NOTE: v^ with weights from yarotsky approximates x**2 wave
    if True:
        x = np.linspace(-1, 1, 10000)
        
        y0 = np.where(x < 0, x*(x+1), x*(1-x))

        # To verify the claim we check the (exponential) decay of the approximation
        # error
        ms = np.arange(2, 11)
        errors = []
        for m in ms:
            y = np.zeros_like(y0)
            x_ = x
            for k in range(1, 1+m):
                x_ = odd_tooth(x_)
                y += x_/(2**(2*k))
            
            e = np.linalg.norm(y0 - y, np.inf)
            errors.append(e)

        c0, c1 = np.polyfit(ms, np.log2(errors), deg=1)
    
        plt.figure()
        plt.semilogy(ms, errors, basey=2, linestyle='none', marker='o')
        plt.semilogy(ms, 2**(c0*ms + c1), basey=2)
        plt.xlabel('nlevels')
        plt.ylabel('error')
        plt.show()

        
    # FIXME: what compose(odd_tooth) approximates
    #

    # (0, ), (1, )
    # (0, 0), (1, 0), (0, 1), (1, 1)
    #
    # (0, ), (1, ), (2, )
    #

    # Generate expression based on fs and 
    # (0, 1, 2, 3)

    #y0 = fs[3](x)
    #y1 = fs[2](y0)
    #y2 = fs[1](y1)
    #y3 = fs[0](y2)

    #        mass matrix
    #        stiffness matrix

    plt.show()

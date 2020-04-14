from yarotsky import relu
import numpy as np


def tooth(x):
    '''
    Shallow network that is the tooth function

       /r\
    in -r- out
       \r/
    '''
    if not isinstance(x, (float, int)):
        return np.array([map(tooth, x)]).flatten()
    
    # x is a scalar
    # Inputs to first layer are
    A1 = np.array([[1, 1]]).T
    b1 = np.array([[0, -0.5]]).T
    y1 = A1.dot(x) + b1

    y1 = relu(y1)
    
    # Inputs to second layer
    A2 = np.array([[2, -4]])
    b2 = np.array([[0]])

    return A2.dot(y1) + b2


def identity(x, nlayers):
    '''Deep relu NN that is the identity'''
    assert nlayers >= 1

    if not isinstance(x, (float, int)):
        return np.array([identity(xi, nlayers) for xi in x]).flatten()
    
    A1 = np.array([[1, -1]]).T
    b1 = np.array([[0, 0]]).T
    y1 = A1.dot(x) + b1

    y1 = relu(y1)
    # Now the remaining ones
    A = np.array([[1, -1], [-1, 1]])

    while nlayers > 1:
        y1 = A.dot(y1)
        y1 = relu(y1)

        nlayers -= 1
        
    # Inputs to second layer
    A2 = np.array([[1, -1]])
    b2 = np.array([[0]])

    return A2.dot(y1) + b2


def saw_tooth(x, s):
    '''Deep relu NN that is saw tooth function'''
    assert s >= 1

    if not isinstance(x, (float, int)):
        return np.array([saw_tooth(xi, s) for xi in x]).flatten()

    # x is a scalar
    # Inputs to first layer are
    A1 = np.array([[1, 1]]).T
    b1 = np.array([[0, -0.5]]).T
    y1 = A1.dot(x) + b1

    y1 = relu(y1)
    A = np.array([[2, -4],
                  [2, -4]])
    b = np.array([[0, -0.5]]).T

    while s > 1:
        y1 = A.dot(y1) + b
        y1 = relu(y1)

        s -= 1
        
    # Inputs to second layer
    A2 = np.array([[2, -4]])
    b2 = np.array([[0]])

    return A2.dot(y1) + b2
    

def x2_approx_skip(x, m):
    '''Yarotsky's neural net (with skip connections)'''
    assert m >= 1
    
    if not isinstance(x, (float, int)):
        return np.array([x2_approx_skip(xi, m) for xi in x]).flatten()

    # Composition
    Ac = np.array([[2, -4],
                   [2, -4]])
    bc = np.array([[0, -0.5]]).T

    # Narrowing for finalizing gs
    Ag = np.array([[2, -4]])
    bg = np.array([[0]])

    # x is a scalar
    # The initial layer consist of id(x) and g(x)
    A1 = np.array([[1, 1]]).T
    b1 = np.array([[0, -0.5]]).T
    y1 = A1.dot(x) + b1
    y1 = relu(y1)

    # This layer connects x with results of finished gs composition 
    y_out = np.zeros((m+1, 1))
    y_out[0] = x
    for s in range(1, m+1):
        # Get the sawtooth for this layer
        y_out[s] = Ag.dot(y1) + bg  # This would be the skip connection
        # Compose for the next one
        y1 = Ac.dot(y1) + bc
        y1 = relu(y1)
        
    # Weights for collapsing concatenated to scalar
    A = np.array([np.r_[1., -1./(2.**(2*np.arange(1, m+1)))]])
    b = np.array([[0]])
    
    return A.dot(y_out) + b 
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from yarotsky import x2_approx

    m = 5
    x = np.linspace(0, 1, 101)
    y = x2_approx_skip(x, m)
    y0 = x2_approx(x, m)

    print np.linalg.norm(y - y0, np.inf)
    
    #plt.figure()
    #plt.plot(x, y0)
    #plt.show()

    
    abs_ = lambda x: relu(x) + relu(-x)
    max_ = lambda x, y: 0.5*(x+y+abs_(x-y))
    min_ = lambda x, y: 0.5*(x+y-abs_(y-x))

    
    print abs_(2)
    print abs_(-2)
    print abs_(-212)
    print abs_(22)

    print max_(2, 3), min_(2, 3)
    print max_(-2, 3), min_(-2, 3)
    print max_(-1, 3), min_(-1, 3)

    hat_ = lambda x, shift=0.5: min_(relu(x)/shift,
                                       relu(1-x)/(1-shift))

    x = np.linspace(0, 1, 10000)
    plt.figure()
    y = hat_(x)
    s = np.zeros_like(y)
    s += y
    for k in range(3):
        plt.plot(x, y, label=str(k))
        y = hat_(y)
        s += y
    plt.legend()

    plt.figure()
    plt.plot(x, s)
    
    plt.show()

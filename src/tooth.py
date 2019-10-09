from yarotsky import relu
import numpy as np


def tooth(x):
    '''
    Shallow network that is the tooth function

       /r\
    in -r- out
       \r/
    '''
    # x is a scalar
    # Inputs to first layer are
    A1 = np.array([[1, 1, 1]]).T
    b1 = np.array([[0, -0.5, -1]]).T
    y1 = A1.dot(x) + b1

    y1 = relu(y1)
    
    # Inputs to second layer
    A2 = np.array([[2, -4, 2]])
    b2 = np.array([[0]])

    return A2.dot(y1) + b2


def identity(x, nlayers):
    '''Deep relu NN that is the identity'''
    assert nlayers >= 1
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
    
    # x is a scalar
    # Inputs to first layer are
    A1 = np.array([[1, 1, 1]]).T
    b1 = np.array([[0, -0.5, -1]]).T
    y1 = A1.dot(x) + b1

    y1 = relu(y1)
    A = np.array([[2, -4, 2],
                  [2, -4, 2],
                  [2, -4, 2]])
    b = np.array([[0, -0.5, -1]]).T

    while s > 1:
        y1 = A.dot(y1) + b
        y1 = relu(y1)

        s -= 1
        
    # Inputs to second layer
    A2 = np.array([[2, -4, 2]])
    b2 = np.array([[0]])

    return A2.dot(y1) + b2
    

def x2_approx_skip(x, m):
    '''Yarotsky's neural net (with skip connections)'''
    assert m >= 1

    # Composition
    Ac = np.array([[2, -4, 2],
                   [2, -4, 2],
                   [2, -4, 2]])
    bc = np.array([[0, -0.5, -1]]).T

    # Narrowing for finalizing gs
    Ag = np.array([[2, -4, 2]])
    bg = np.array([[0]])

    # x is a scalar
    # The initial layer consist of id(x) and g(x)
    A1 = np.array([[1, 1, 1]]).T
    b1 = np.array([[0, -0.5, -1]]).T
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
    y = np.array([x2_approx_skip(xi, m) for xi in x]).flatten()
    y0 = np.array([x2_approx(xi, m) for xi in x]).flatten()

    print np.linalg.norm(y - y0, np.inf)
    
    plt.figure()
    plt.plot(x, y)
    plt.show()

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
    

def x2_approx_noskip(x, m):
    '''Yarotsky's neural net as Feed Forward Neural Net'''
    assert m >= 1

    # The idea here is that output of each gs is propagated through
    # the next layers as identity
    # To get composition (ie next gs)
    Ac = np.array([[2, -4, 2],
                  [2, -4, 2],
                  [2, -4, 2]])
    bc = np.array([[0, -0.5, -1]]).T

    # To get output of gs
    Ag = np.array([[2, -4, 2],
                   [-2, 4, -2]])
    bg = np.array([[1, -1]]).T

    # Propagating identity
    Ai = np.array([[1, -1],
                   [-1, 1]])
    bi = np.array([[0, 0]]).T
    
    # x is a scalar
    # The initial layer consist of id(x) and g(x)
    A1 = np.array([[1, -1, 1, 1, 1]]).T
    b1 = np.array([[0, 0, 0, -0.5, -1]]).T
    y1 = A1.dot(x) + b1

    y1 = relu(y1)

    y_next = np.array
    # I don't build one by matrix here for the mapping. Instead, use
    # its block structure and build the layer by component
    while m > 1:
        m -= 1
        # x by identity
        
        # previous g_s by identity

        # next g_x

    # In the final layer we want to collapse final g_s and then
    # do the sum of results 
    print m
    # Collapsing the sum x - ...
    A = np.r_[1, -1,  # Collapse identity
              #-1./np.array([2**(2*s) for s in range(1, m)]),
              #1./np.array([2**(2*s) for s in range(1, m)]),
              -np.array([2., -4., 2.])/4]
    print A, y1
    return A.dot(y1)
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt


    # print x2_approx_noskip(x=0.2, m=2)
    
    x = np.linspace(0, 1, 10001)
    y = np.array([x2_approx_noskip(xi, 2) for xi in x]).flatten()

    plt.figure()
    plt.plot(x, y)
    plt.show()

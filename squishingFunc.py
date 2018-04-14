#!usr/bin/env python3

"""
    File used to store different function used to train the neural
    network.
"""

import numpy as np


# ----------------------------- Sigmoid ------------------------------


def Sigmoid(x):
    """
        Sigmoid function.
        Oldschool way to train networks.
        Base function used to train neural network but unefficient
        x in [-inf, +inf] and return a value in ]0, 1[.
    """
    return 1/(1 + np.exp(-x))



def InvSigmoid(x):
    """
        Inverse Sigmoid function.
        x in ]0, 1[ and return a value in [-inf, +inf].
        y = 1/(1 + np.exp(-x)) <=> x = ln(y) - ln(1)
    """
    return np.log(x) - np.log(1-x)



def DerSigmoid(x):
    """
        Derivative of Sigmoid function.
        x in [-inf, +inf] and return a value in ]0, 1[.
        Sigmoid'(x) = (1/2)(1/(cosh(x)+1))
    """
    return (1/2)(1/(np.cosh(x)+1))

# ------------------------------ ReLU --------------------------------


def ReLU(x):
    """
        Rectified Linear Unit function.
        The idea is that there is a real activation in neurals from
        our brains. This function is inspired by biological researchs.
        If there is no activation, here x < 0, we return 0. This means
        that the neuron doesn't emit any output signal.
        However if x >= 0, the neuron emit a signal that correspond to
        the entry.

        Beware : this function tends to work well, however neurons in
        each layer except the first one are not values in ]0, 1[ but
        in ]0, +inf[ when using this function.
    """
    return x * (x > 0)



def InvReLU(x):
    """
        Inverse Rectified Linear Unit function.
        Return a value in [0, +inf] if we suppose the entry x in
        [0, +inf].
    """
    return x



def DerReLU(x):
    """
        Derivative of Rectified Linear Unit function.
        If x > 0, return 1, if x < 0 return 0.
        In the case of x = 0, return 1/2. It is not mathematically true, however
        in pratical in works well because it is the value that makes sense.
    """
    return (x > 0) + (1/2)*(x == 0)

# ------------------------------ ReEU --------------------------------


def ReEU(x):
    """
        Rectified Exponential Unit function.
        Kind of a mix between sigmoid and ReLU.
        Function that returns a value between [0, 1[ if the entry x
        is in [0, +inf] otherwise it returns 0.
        Used this method because it is faster than np.maximum(0, x).
    """
    return (1 - np.exp(-x)) * (x > 0)



def InvReEU(x):
    """
        Inverse Rectified Exponential Unit function.
        Function that returns a value between ]0, +inf[ and we suppose
        the entry x in ]0, 1[.
        y =  1-exp(-x) <=> x = -ln(1-y)
    """
    # create an array of the same size than x with only 1 value in it
    one = np.array([1]*len(x))
    return -np.log(1 + 10e-16 -x)



def DerReEU(x):
    """
        Derivative of Rectified Exponential Unit function.
        Function that returns a value between ]0, 1] if the entry x
        is in [0, +inf] otherwise it returns 0.
        In the case of x = 0, return 1/2. It is not mathematically true, however
        in pratical in works well because it is the value that makes sense.
    """
    return np.exp(-x) * (x > 0) + (1/2)*(x == 0)

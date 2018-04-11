#!usr/bin/env python3

"""
    File used to store different function used to train the neural
    network.
"""

import numpy as np


# ----------------------------- Sigmoid ------------------------------


def sigmoid(x):
    """
        Sigmoid function.
        Oldschool way to train networks.
        Base function used to train neural network but unefficient
        x in [-inf, +inf] and return a value in ]0, 1[
    """
    return 1/(1 + np.exp(-x))



def Invsigmoid(x):
    """
        Inverse Sigmoid function.
        x in ]0, 1[ and return a value in [-inf, +inf].
        y = 1/(1 + np.exp(-x)) <=> x = ln(y) - ln(1)
    """
    return np.log(x) - np.log(1-x)


# ------------------------------ ReLU --------------------------------


def ReLU(x):
    """
        Rectified Linear Unit function.
        The idea is that there is a real activation in neurals from
        our brains. This function is inspired by biological researchs.
        If there is no activation, here x < 0, we return 0. This means
        that the neuron doesn't emit any output signal.
        However if x > 0, the neuron emit a signal that correspond to
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
    one = np.array([1]*len(x))
    return -np.log(one-x)

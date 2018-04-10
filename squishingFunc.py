#!usr/bin/env python3

"""
    file used to store different function used to train the neural network
"""

import numpy as np


def sigmoid(x):
    """
        Sigmoid function
        Oldschool way to train networks.
        Base function used to train neural network but unefficient
        x in [-inf, +inf] and return a value in [0, 1]
    """
    return 1/(1 + np.exp(-x))



def ReLU(x):
    """
        Rectified Linear Unit function
        The idea is that there is a real activation in neurals from our brains.
        This function is inspired by biological researchs.
        If there is no activation, here x < 0, we return 0. This means that the
        neuron doesn't emit any output signal. However if x > 0, the neuron emit
        a signal that correspond to the entry.
    """
    return x * (x > 0)



def ReEU(x):
    """
        Rectified Exponential Unit function
        Kind of a mix between sigmoid and ReLU
        Function that returns a value between [0, 1] if the entry x
        is in [0, +inf] otherwise it returns 0.
        Used this method because it is faster than np.maximum(0, x).
    """
    return (1 - np.exp(-x)) * (x > 0)

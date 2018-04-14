#!usr/bin/env python3

"""
    File used to separate the code and therefore make it cleaner by
    separating some functions in that file when that makes sense.
"""

import numpy as np



# ----------------------------- Gradient Descent ------------------------------

def NegPower(n, value=1.3):
    """
        Generate a natural gradient descent factor.

        Inputs :

        -> n        : INT in [0, +inf[

        -> value    : FLOAT in ]0, +inf[

        Output :

        <-          : FLOAT in ]0, 1[. Return value^(-n).
    """
    return np.power(value, -n)



def Constant(n, value=1):
    """
        Generate a constant gradient descent factor.
    """
    return value

# ------------------------------- Cost Function -------------------------------

def CostFunction(output_layer, perfect_output):
    """
        Generate the cost array. To calculate the cost, you just have to
        compute sum(CostFunction(output_layer, perfect_output)).

        Inputs :

        -> output_layer, perfect_output : two NUMPY ARRAYS with the same size.

        Output :

        <-          : NUMPY ARRAY which size is equal to the size of the inputs
    """
    assert(len(output_layer) == len(perfect_output))
    return np.power(training_output - perfect_output, 2)



def DerCostFunction(output_layer, perfect_output):
    """
        Generate an numpy array where each of its composant is the derivative of
        the cost array.
        Exact same inputs and output as the function above.
    """
    return 2.0*np.array(training_output - perfect_output)



#  -------------------------------- other ------------------------------------

def isfloat(string):
    """
        Function useful to test whether or not a string is a float or not.
    """
    try:
        num = float(string)
    except ValueError:
        return False
    return True

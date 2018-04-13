#!/usr/bin/env python3

"""
    File used to test a lot of stuff
"""

import numpy as np
import random
from squishingFunc import *

def testFunc(func, param):
    """
        Function used to test how we implement a func as a parameter in
        another function
    """
    return func(param)


def main():
    """
        Main function where there are a lot of stuff tested about
        python3 language
    """
    # generation of list test
    tab = [0]*5
    assert(len(tab) == 5)
    for element in tab:
        assert(element == 0)

    # function applied to a numpy array test
    array = np.array([1, 2, 3, 4, 5])
    new_array = Sigmoid(array)
    assert(round(new_array[0], 8) == 0.73105858)
    assert(round(new_array[4], 8) == 0.99330715)

    # string reduction test
    string = "abcdef"
    assert(string[:-2] == "abcd")

    # for in range negative
    for index in range(3, -1, -1):
        # print(index)
        continue
    assert(index == 0)

    # insert in list test
    test = ([0.05]*9)
    test.insert(2, 0.99)
    # print(test)

    # function as a parameter in another function test
    t1 = testFunc(Sigmoid, 0)
    t2 = testFunc(Sigmoid, 1)
    t3 = testFunc(Sigmoid, -1)
    array = testFunc(Sigmoid, np.array([i for i in range(10)]))
    assert(t1 == 0.5)
    assert(round(t2, 11) == 0.73105857863)
    assert(round(t3, 11) == 0.26894142137)

    liste2 = [Sigmoid, DerReLU]
    array = testFunc(liste2[0], np.array([1, -1]))
    assert(round(array[0], 11) == 0.73105857863)
    assert(round(array[1], 11) == 0.26894142137)

    training_output = np.array([0.05, 0.82, 0.3])
    perfect_output = np.array([0, 1, 0])
    assert(round(sum(np.power(training_output-perfect_output, 2)), 4) == 0.1249)

    training_output = np.array([0.05, 0.82, 0.3])
    perfect_output = np.array([0, 1, 0])
    print(np.multiply(training_output, perfect_output))

if __name__ == "__main__":
    main()

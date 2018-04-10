#!/usr/bin/env python3

"""
    File used to test a lot of stuff
"""

import numpy as np
from squishingFunc import *

def main():
    """
        Main function where there are a lot of stuff tested about
        python3 language
    """
    tab = [0]*5
    print(tab)
    array = np.array([1, 2, 3, 4, 5])
    new_array = sigmoid(array)
    print(new_array)
    string = "abcdef"
    print(string[:-2])
    print(max([0, 0, 0, 0]))

if __name__ == "__main__":
    main()

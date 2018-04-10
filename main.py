#!/usr/bin/env python3

"""
    Description of the whole problem and the implementation.
"""

import numpy as np
import sys
from mnistHandwriting import *
from neuralNetwork import *

# main function to execute the whole thing
def main():
    """
        Main function. It calls everything to make the whole thing work
        http://sametmax.com/les-docstrings/
    """
    print(len(sys.argv))
    if len(sys.argv) <= 1:
        print("ERROR : There is no file arguement.")
        sys.exit(1)

    # initilization of the dataset
    data = MNISTexample(0, 1)
    print(len(data[0][0]), len(data[0][1]))

    # creation of the network
    network = NeuralNetwork(sys.argv[1])
    print(network.test(data[0][0]))

if __name__ == '__main__':
    main()

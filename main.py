#!/usr/bin/env python3

"""
    Description of the whole problem and the implementation.

    Advice linter for the code :
        - linter
        - linter-pylint
        - minimap

    Documentation about docstring convention in Python :
    http://sametmax.com/les-docstrings/
"""

import numpy as np
import sys
import time
from mnistHandwriting import *
from neuralNetwork import *

# main function to execute the whole thing
def main():
    """
        Main function. It calls everything to make the whole thing work
    """


    # ./main.py network/network1.txt => len(sys.argv) == 2
    if len(sys.argv) <= 1:
        print("ERROR : There is no file arguement.")
        sys.exit(1)

    # initilization of the dataset
    elapsed_time = time.time()
    data = MNISTexample(0, 1000)
    elapsed_time = time.time() - elapsed_time
    print("Time to load the training data set :", elapsed_time, "s")
    # print(len(data[0][0]), len(data[0][1]))

    # timie = time.time()
    # timie = time.time() -timie
    # print("ezg", timie)

    # creation of the network
    network = NeuralNetwork(sys.argv[1])
    # print(network.generateAllLayers(data[0][0]))

    # generate a image thank to the neural network
    # print(network.generateInputLayer(test))

if __name__ == '__main__':
    main()

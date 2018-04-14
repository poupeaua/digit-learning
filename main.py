#!/usr/bin/env python3

"""
    Description of the whole problem and the implementation.

    Advice packages for the code :
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

    # initilization of the training_dataset
    elapsed_time = time.time()
    training_data = MNISTexample(0, 1000, bTrain=True)
    elapsed_time = time.time() - elapsed_time
    print("Time to load the training data set :", elapsed_time, "s")

    assert(len(training_data[0][0]) == 784)
    assert(len(training_data[0][1]) == 10)

    # creation of the network
    network = NeuralNetwork(sys.argv[1])
    test1 = network.generateAllLayers(training_data[0][0])[0][network.nb_layer-1]
    assert(len(test1) == 10)

    # train the network
    network.train()

    # test the network
    testing_data = MNISTexample(0, 1000, bTrain=False)

    assert(len(testing_data[0][0]) == 784)
    assert(len(testing_data[0][1]) == 10)

    # generate a image thank to the neural network
    test_generated_output = network.generateInputLayer(test1)
    assert(len(test_generated_output) == 784)

if __name__ == '__main__':
    main()

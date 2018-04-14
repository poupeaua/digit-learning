#!/usr/bin/env python3

"""
    Description of the whole problem and the implementation.

    Required for this to work:
        - python3 (while coding 'python3 --version' = Python 3.5.2)
        - numpy (install by doing 'pip3 install numpy')
        - progressbar2 (install by doing 'pip3 install progressbar2')

    Recommended packages for the code :
        - linter
        - linter-pylint
        - minimap

    Documentation about docstring convention in Python :
    http://sametmax.com/les-docstrings/
"""

import numpy as np
import sys
import time
from progressbar import *
from mnistHandwriting import *
from neuralNetwork import *
from argumentsManager import *


# main function to execute the whole thing
def main():
    """
        Main function. It calls everything to make the whole thing work
    """
    args = ArgsManager(sys.argv)
    args.display()

    liste = [1]*1000000
    widget = ['Test: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()]
    bar = ProgressBar()
    a = 0
    for i in bar(liste):
        a += 1

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

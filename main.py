#!/usr/bin/env python3

"""
    Required for this to work:
        - python3 (while coding 'python3 --version' = Python 3.5.2)
        - numpy (install by doing 'pip3 install numpy')

    Recommended packages for the code :
        - linter by steelbrain
        - linter-pylint by AtomLinter
        - minimap by atom-minimap

    Pratical stuff:
        - find . -name "*csv"

    Documentation about docstring convention in Python :
    http://sametmax.com/les-docstrings/
"""

import numpy as np
import sys, time
from src.mnistHandwriting import *
from src.neuralNetwork import *
from src.argumentsManager import *


# main function to execute the whole thing
def main():
    """
        Main function. It calls everything to make the whole thing work
    """
    args = ArgsManager(sys.argv)
    if args.to_display:
        args.display()

    # initilization of the training_dataset
    training_data = MNISTexample(0, args.learning_size, bTrain=True)

    # creation of the network
    network = NeuralNetwork(args.neural_network, args.squishing_funcs,
                args.dir_load)

    # train the network
    network.train(training_data, args.batches_size, args.grad_desc_factor,
                   args.repeat)

    # save the network after training (if args.save != False)
    if args.dir_save != None:
        network.save(args.dir_save)

    # test the network
    testing_data = MNISTexample(0, args.testing_size, bTrain=False)
    error_rate, average_cost = network.test(testing_data)
    print("The error rate is", error_rate*100, "%.")

    if args.dir_save != None and args.to_info:
        network.inform(args, error_rate, average_cost)

    # temporal tests
    # print(training_data[0][1])
    # assert(len(training_data[0][0]) == 784)
    # assert(len(training_data[0][1]) == 10)
    # assert(len(testing_data[0][0]) == 784)
    # assert(len(testing_data[0][1]) == 10)
    # test1 = network.generateOuputLayer(training_data[0][0])
    # assert(len(test1) == 10)
    # test_generated_output = network.generateInputLayer(test1)
    # assert(len(test_generated_output) == 784)
    # for _ in range(10):
    #     network.train(training_data, args.batches_size, args.grad_desc_factor,
    #                    args.repeat)
    #     error_rate = network.test(testing_data)






if __name__ == '__main__':
    main()

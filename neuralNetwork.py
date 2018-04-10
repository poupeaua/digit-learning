#!/usr/bin/env python3

import numpy as np
from squishingFunc import *

SIZE_INPUT = 784 # 28 * 28 = 784 pixels
SIZE_OUTPUT = 10 # number of numbers between 0 and 9


class NeuralNetwork:
    """
        class neural network
    """

    def __init__(self, entry):
        """
            initialize a neural network
            arguement:
            -> entry : str variable that is the name of a txt document.
                      This document contains all the information concerning
                      the layers and their sizes. It will be used as followed :
                      "./main.py information.txt"
                      ex of entry : network1.txt
        """

        document = open(entry, "r")

        # number of layers in the neural network
        self.nb_layer = int(document.readline(1))

        # number of neurals in each layer (except the input and output layers)
        self.len_layers = [None]*self.nb_layer
        index = 0
        for line in document:
            string = line[:-1]
            if string != "":
                self.len_layers[index] = int(string)
                index += 1

        # (nb_layer + 1) matrix and bias vectors composed the neural network
        self.weights = [None]*(self.nb_layer+1)
        self.bias = [None]*(self.nb_layer+1)

        if self.nb_layer > 0:
            #  shape = (row, column)
            self.weights[0] = 0 * np.ones(shape=(self.len_layers[0], SIZE_INPUT))
            self.bias[0] = np.array([0]*self.len_layers[0])

            # in case there is more than one layer
            if self.nb_layer != 1:
                for index in range(0, self.nb_layer-1):
                    self.weights[index+1] = 0 * np.ones(
                        shape=(self.len_layers[index+1], self.len_layers[index]))
                    self.bias[index+1] = np.array([0]*self.len_layers[index+1])

            self.weights[self.nb_layer] = 0 * np.ones(
                shape=(SIZE_OUTPUT, self.len_layers[self.nb_layer-1]))
            self.bias[self.nb_layer] = np.array([0]*SIZE_OUTPUT)
        else:
            print("ERROR : The number of layer is negative or equal to zero.")
            print("nb_layer = ", self.nb_layer)



    def train(self, in_out_layers):
        """
            method used to train the neural network
            arguement :
            -> inOutLayers : a tuple that contains two numpy arrays
                          the first numpy array has a length of 28x28 = 784
                          each element is in [0, 1] 0 means a dark pixel
                          and 1 means a white pixel
                          the second numpy array has length of 10.
                          This is the best output that could be obtain when
                          we test the neural network with the according image
        """
        training_input = in_out_layers[0]
        perfect_output = in_out_layers[1]



    def test(self, input_layer):
        """
            method used to test the neural network model
            arguement :
            -> input_layer : it is a numpy array which len is 784.
                             it contains the color of pixel (white / black) with
                             number notation from 0 to 1 (everything was divided
                             by 255)
            <- new_array   : it is a numpy array of size 10. Each element is in
                             [0, 1]
        """

        new_array = input_layer
        for index in range(0, self.nb_layer+1):
            # print(len(new_array))
            new_array = ReEU(self.weights[index].dot(new_array)
                                + self.bias[index])
        # print(len(new_array))
        return new_array

    def __str__(self):
        return "hop"

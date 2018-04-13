#!/usr/bin/env python3

"""
    File neuralNetwork.py is used to create and handle a Neural Network
    as an object.
"""

import sys
import numpy as np
from squishingFunc import *
import random

SIZE_INPUT = 784 # 28 * 28 = 784 pixels
SIZE_OUTPUT = 10 # number of numbers between 0 and 9


class NeuralNetwork:
    """
        Class neural network.
    """

    def __init__(self, entry):
        """
            Initialize a neural network.

            Inputs :

            -> entry : STRING variable that is the name of a txt document.
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

        # (nb_layer + 1) matrix and biases vectors composed the neural network
        self.weights = [None]*(self.nb_layer+1)
        self.biases = [None]*(self.nb_layer+1)

        if self.nb_layer > 0:
            # shape=(row, column) / np.random.rand(row, column) values in [0,1[
            # weights and biases to calculate the second layer
            self.weights[0] = np.random.rand(self.len_layers[0], SIZE_INPUT)
            self.biases[0] = np.random.rand(self.len_layers[0])

            # in case there is more than one layer (except the in and output)
            if self.nb_layer != 1:
                for index in range(0, self.nb_layer-1):
                    self.weights[index+1] = np.random.rand(
                            self.len_layers[index+1], self.len_layers[index])
                    self.biases[index+1] = np.random.rand(
                            self.len_layers[index+1])

            # weights and biases to calculate the last layer
            self.weights[self.nb_layer] = np.random.rand(
                    SIZE_OUTPUT, self.len_layers[self.nb_layer-1])
            self.biases[self.nb_layer] = np.random.rand(SIZE_OUTPUT)
        else:
            print("ERROR : The number of layer is negative or equal to zero.")
            print("nb_layer = ", self.nb_layer)
            sys.exit(1)



    def train(self, data, batch_size=100, repeat=1):
        """
            Method used to train the neural network.
        """

        # for _ in range(repeat):

    def update(self, index, dweights, dbaises, der_func_z, der_cost_to_a,
                current_layer):
        """

        """
        der_func_z = DerReEU(z_values[index]) # derivative of function to z
        der_cost_to_a = dcost_array # load the derivative of cost function to a
        current_layer = values_layers[index] # take the second to last layer


    def calculateNegGradient(self, in_out_layers):
        """
            Method used to train the neural network.

            Inputs :

            -> inOutLayers : a TUPLE that contains two numpy arrays
                          the first numpy array has a length of 28x28 = 784
                          each element is in [0, 1] 0 means a dark pixel
                          and 1 means a white pixel
                          the second numpy array has length of 10.
                          This is the best output that could be obtain when
                          we test the neural network with the according image

            Output :

            <- negativeGradient : NUMPY ARRAY of a size equal to the number of
                          weights et biases in the neural network.
        """
        training_input = in_out_layers[0]
        values_layers, z_values = self.generateAllLayers(training_input)
        training_output = values_layers[self.nb_layer+1]

        perfect_output = in_out_layers[1]

        # cost function = sum (a_j^N - y)^2
        cost_array = np.power(training_output - perfect_output, 2)
        cost = sum(cost_array)
        dcost_array = 2.0*np.array(training_output - perfect_output)

        # (nb_layer + 1) matrix and biases vectors composed the neural network
        dweights = [None]*(self.nb_layer+1)
        dbiases = [None]*(self.nb_layer+1)

        # n = N initialization
        der_func_z = DerReEU(z_values[self.nb_layer]) # derivative of function to z
        der_cost_to_a = dcost_array # load the derivative of cost function to a
        current_layer = values_layers[self.nb_layer] # take the second to last layer

        der_cost_to_a = self.update(index, dweights, dbaises, der_func_z, der_cost_to_a, current_layer)

        dweights[self.nb_layer] = np.zeros(
                SIZE_OUTPUT, self.len_layers[self.nb_layer-1])


        if self.nb_layer != 1:
            for index in range(self.nb_layer-1, 0, -1):

                dweights[self.nb_layer] = np.zeros(
                        SIZE_OUTPUT, self.len_layers[self.nb_layer-1])

        dweights[0] = np.zeros(self.len_layers[0], SIZE_INPUT)

        return (dweights, dbiases)

    def generateAllLayers(self, input_layer):
        """
            Method used to test the neural network model by giving it a array
            that contains all the pixels from an handwriting image.
            Each step consists in calculating f(A_i x_i + b_i) = x_(i+1).
            A_i is a weight matrix, b_i is a biases vector, x_i is the neural
            vector or layer of index i and x_(i+1) of index (i+1).

            Input :

            -> input_layer : NUMPY ARRAY which len is 784.
                             it contains the color of pixel (white / black) with
                             number notation from 0 to 1 (everything was divided
                             by 255)

            Output :

            <- values_layers : LIST of NUMPY ARRAY for each layer in the neural
                             network (they all have different sizes) which size
                             is nb_layer + 2.
                             This will be used to calculate the
                             negative gradient and therefore to do the back
                             propagation. Thus values_layers[self.nb_layer+1] is
                             a NUMPY ARRAY of size 10.
                             ex : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] in the best
                             case scenario if the input is a handwriting five.

            <- z_values :    LIST of NUMPY ARRAY for each layer in the neural
                             network minus one (except the first one).
                             Thus its size is nb_layer+1.
        """

        new_array = input_layer
        values_layers = [new_array]
        z_values = []
        for index in range(0, self.nb_layer+1):
            # print(len(new_array))
            z = self.weights[index].dot(new_array) + self.biases[index]
            new_array = ReEU(z)
            values_layers.append(new_array)
            z_values.append(z)
        # print(len(new_array))
        return (values_layers, z_values)



    def generateInputLayer(self, output_layer):
        """
            Method used to generate an input of the neural network.
            This input can by used after to reconstitute an image thanks to the
            function writeMNISTimage in mnistHandwriting.py using
            Python Image Library.
            Each step consists in calculating x_i=A_i^(-1)[f^(-1)(x_(i+1))-b_i].
            A_i is a weight matrix, b_i is a biases vector, x_i is the neural
            vector or layer of index i and x_(i+1) of index (i+1).

            Inputs :

            -> output_layer : NUMPY ARRAY of size 10. Each element is in
                             [0, 1]. ex : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

            Output :

            <- new_array   : NUMPY ARRAY which len is 784.
                             it contains the color of pixel (white / black) with
                             number notation from 0 to 1
        """
        new_array = output_layer
        # iteration from nb_layer => 0
        for index in range(self.nb_layer, -1, -1):
            # print(len(new_array))
            invA = np.linalg.pinv(self.weights[index])
            new_array = invA.dot(InvReEU(new_array)-self.biases[index])
        # print(len(new_array))
        return new_array



    def __str__(self):
        return "hop"

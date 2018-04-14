#!/usr/bin/env python3

"""
    File neuralNetwork.py is used to create and handle a Neural Network
    as an object.
"""

import sys
import numpy as np
from squishingFunc import *
from externalFunc import *
import random

SIZE_INPUT = 784 # 28 * 28 = 784 pixels
SIZE_OUTPUT = 10 # number of numbers between 0 and 9


class NeuralNetwork:
    """
        Class neural network.
    """

    def __init__(self, entry):
        """
            Initialize an object NeuralNetwork.

            Inputs :

            -> entry : STRING variable that is the name of a txt document.
                      This document contains all the information concerning
                      the layers and their sizes. It will be used as followed :
                      "./main.py information.txt"
                      ex of entry : network1.txt
        """

        document = open(entry, "r")

        # number of layers in the neural network + output and input layer
        self.nb_layer = int(document.readline(1)) + 2

        # number of neurals in each layer
        self.len_layers = [None]*self.nb_layer
        self.len_layers[0] = SIZE_INPUT
        index = 1
        for line in document:
            string = line[:-1]
            if string != "":
                self.len_layers[index] = int(string)
                index += 1
        self.len_layers[self.nb_layer-1] = SIZE_OUTPUT

        # (nb_layer - 1) matrix and biases vectors composed the neural network
        self.weights = [None]*(self.nb_layer-1)
        self.biases = [None]*(self.nb_layer-1)

        if self.nb_layer > 2:
            for index in range(0, self.nb_layer-1):
                self.weights[index] = np.random.rand(
                        self.len_layers[index+1], self.len_layers[index])
                self.biases[index] = np.random.rand(
                        self.len_layers[index+1])
        else:
            print("ERROR : The number of layer is inferior to 3.")
            print("nb_layer = ", self.nb_layer)
            sys.exit(1)



    def initializeEmptyDParamArrays(self):
        """
            Method used to do the initialization of dw and db
            for the method train.
        """
        dweights = [None]*(self.nb_layer-1)
        dbiases = [None]*(self.nb_layer-1)
        for index in range(0, self.nb_layer-1):
            dweights[index] = np.zeros(
                    self.len_layers[index+1], self.len_layers[index])
            dbiases[index] = np.zeros(
                    self.len_layers[index+1])
        return (dweights, dbiases)



    def train(self, training_data, batch_size, gradientDescentFactor, repeat):
        """
            Method used to train the neural network.

            If batch_size == 1 => individual training
            Else               => mini_batching training
        """
        size_training_data = len(training_data)


        for i in range(0, round(size_training_data/batch_size)-1, 1):

            # we can choose how many time we want to repeat the operation
            # in order to get a deeper and a more efficent learning
            for nb_repetition in range(repeat):

                (dw,db) = self.initializeEmptyDParamArrays()

                # iteration on the size of a batch
                for index in range(i*batch_size, (i+1)*batch_size-1):

                    # generate the image to use for the training and its
                    # expected output
                    in_out_layers = training_data[index]
                    (dw2, db2) = self.calculateNegGradient(in_out_layers)

                    if batch_size == 1:
                        dw, db = dw2, db2
                        break

                    # add the gradient due to weights and biases
                    for index2 in range(0, self.nb_layer-1):
                        dw[index2] += dw2[index2]
                        db[index2] += db2[index2]

                for index in range(0, self.nb_layer-1):

                    # apply the gradient descent factor to all the weights
                    # and biases gradients
                    # morevover, divide by the batch_size to normalize the grad
                    dw[index] *= gradientDescentFactor(nb_repetition)/batch_size
                    db[index] *= gradientDescentFactor(nb_repetition)/batch_size

                    # finally update the weights and the biases in the neural
                    # network
                    self.weights[index] += dw[index]
                    self.biases[index] += db[index]



    def derivativeCostToParam(self, index, a, der_func_z, der_cost_to_a):
        """
            Optimized calculation for derivative of the cost function according
            to the following parameters : bias, weight and a.

            Complexity : (column+1) * row
        """
        row = self.len_layers[index+1]
        column = self.len_layers[index]

        dweight_matrix = np.zeros(row, column)
        dbiases_array = np.zeros(row)
        da_array = np.zeros(column)

        for i in range(0, row - 1):
            dbiases_array[i] = der_func_z[i] * der_cost_to_a[i]

        for j in range(0, column - 1):
            da_array[j] = 0
            for i in range(0, row - 1):
                da_array[j] += self.weights[index][i][j] * dbiases_array[i]
                dweight_matrix[i][j] = a[j] * dbiases_array[i]

        # return negative of the matrix and array because we look for the
        # negative gradient so we have to multiply by -1
        return -dweight_matrix, -dbiases_array, da_array



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

            <- (dweights, dbiases) : TUPLE of LISTS.
                          The first one contains NUMPY MATRIX for all the
                          weight matrix in the neural network.
                          The second one contains NUMPY ARRAY for all the
                          biases array in the neural network.
        """
        training_input = in_out_layers[0]
        values_layers, z_values = self.generateAllLayers(training_input)
        training_output = values_layers[self.nb_layer+1]

        perfect_output = in_out_layers[1]

        # cost function = sum (a_j^N - y)^2 we do not really need that
        # cost = sum(CostFunction(training_output, perfect_output))

        # (nb_layer - 1) matrix and biases vectors composed the neural network
        dweights = [None]*(self.nb_layer+1)
        dbiases = [None]*(self.nb_layer+1)

        der_cost_to_a = DerCostFunction(training_output, perfect_output)
        # from (nb_layer - 2) to 0
        for index in range(nb_layer-2, -1, -1):
            der_func_z = DerReEU(z_values[index])
            a = values_layers[index]
            # derivative cost to param weights, biases and a
            dweights[index], dbiases[index], der_cost_to_a = \
                self.derivativeCostToParam(index, a, der_func_z, der_cost_to_a)

        return (dweights, dbiases)



    def generateOuputLayer(self, input_layer):
        """
            Method used to test the neural network model by giving it a array
            that contains all the pixels from an handwriting image. Use the same
            principle as generateAllLayers method.

            Input :

            -> input_layer : NUMPY ARRAY which len is 784.
                             it contains the color of pixel (white / black) with
                             number notation from 0 to 1 (everything was divided
                             by 255)

            Output :

            <-output_layer : NUMPY ARRAY of size 10.
                             ex : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] in the best
                             case scenario if the input is a handwriting five.
        """
        new_array = input_layer
        for index in range(0, self.nb_layer-1):
            # print(len(new_array))
            z = self.weights[index].dot(new_array) + self.biases[index]
            new_array = ReEU(z)
        # print(len(new_array))
        return new_array


    def generateAllLayers(self, input_layer):
        """
            Method used when training the neural network model by giving it a
            array that contains all the pixels from an handwriting image.
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
        for index in range(0, self.nb_layer-1):
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
        # iteration from nb_layer-2 => 0
        for index in range(self.nb_layer-2, -1, -1):
            # print(len(new_array))
            invA = np.linalg.pinv(self.weights[index])
            new_array = invA.dot(InvReEU(new_array)-self.biases[index])
        # print(len(new_array))
        return new_array



    def test(self, testing_data):
        """
            Method used to test the neural network after its training.
        """
        nb_correct = 0
        nb_test = len(testing_data)

        for element in testing_data:
            input_layer = element[0]
            perfect_output = element[1]
            generated_output = generateOuputLayer(input_layer)
            # not really necessary just for information
            cost_array = CostFunction(generated_output, perfect_output)
            cost = sum(cost_array)
            index_max_value = np.argmax(generated_output)
            expected_answer = np.argmax(perfect_output)
            if index_max_value == expected_answer:
                nb_correct += 1

        return nb_correct/nb_test


    def __str__(self):
        return "hop"

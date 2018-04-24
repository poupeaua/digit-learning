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
from progressbar import *

SIZE_INPUT = 784 # 28 * 28 = 784 pixels
SIZE_OUTPUT = 10 # number of numbers between 0 and 9


class NeuralNetwork:
    """
        Class neural network.
    """

    def __init__(self, len_layers, squishing_funcs, dir_load):
        """
            Initialize an object NeuralNetwork.

            Inputs :

            -> entry : STRING variable that is the name of a txt document.
                      This document contains all the information concerning
                      the layers and their sizes. It will be used as followed :
                      "./main.py information.txt"
                      ex of entry : network1.txt
        """

        # number of layers in the neural network + output and input layer
        self.nb_layer = len(len_layers)

        # number of neurals in each layer
        self.len_layers = len_layers

        # (nb_layer - 1) matrix and biases vectors composed the neural network
        self.weights = [None]*(self.nb_layer-1)
        self.biases = [None]*(self.nb_layer-1)

        if self.nb_layer > 2:
            self.initializeWeightsBiases(dir_load)
        else:
            print("ERROR : The number of layer is inferior to 3.")
            print("nb_layer = ", self.nb_layer)
            sys.exit(1)

        # squishing functions used for each layer. Except the last one,
        # because the last layer doesn't calculate another layer.
        self.squishing_funcs = squishing_funcs



    def initializeWeightsBiases(self, dir_load):
        """
            Method used to initialize the matrix weights and the vectors biases.
            If load == None, it means that this will not load weights and biases
            Else load == "dir/doc.txt" load weight and biases from that doc
        """
        if dir_load == None:
            for index in range(0, self.nb_layer-1):
                self.weights[index] = 0.01*((-1)**index)*np.random.rand(
                        self.len_layers[index+1], self.len_layers[index])
                self.biases[index] = 0.01*((-1)**index)*np.random.rand(
                        self.len_layers[index+1])
        else:
            for index in range(0, self.nb_layer-1):
                data = np.load(dir_load+"/"+str(index)+".npz")
                self.weights[index] = data["w"]
                self.biases[index] = data["b"]


    def initializeEmptyDParamArrays(self):
        """
            Method used to do the initialization of dw and db
            for the method train.
        """
        dweights = [None]*(self.nb_layer-1)
        dbiases = [None]*(self.nb_layer-1)
        for index in range(0, self.nb_layer-1):
            dweights[index] = np.zeros(shape=(
                    self.len_layers[index+1], self.len_layers[index]))
            dbiases[index] = np.zeros(
                    self.len_layers[index+1])
        return (dweights, dbiases)



    def train(self, training_data, batch_size, gradientDescentFactor, repeat):
        """
            Method used to train the neural network.

            If batch_size == 1 => individual training
            Else               => mini_batching training

            Repeat is the number of repetition of learning for each batch.
        """
        size_training_data = len(training_data)

        gdf_func = gradientDescentFactor[0]
        gdf_param = gradientDescentFactor[1]

        bar2 = ProgressBar()
        print("Train process :")
        for i in bar2(range(0, round(size_training_data/batch_size))):
            # we can choose how many time we want to repeat the operation
            # in order to get a deeper and a more efficent learning
            for nb_repetition in range(0, repeat+1):

                (dw,db) = self.initializeEmptyDParamArrays()

                # iteration on the size of a batch
                for index in range(i*batch_size, (i+1)*batch_size):
                    # extract the image to use for the training and its
                    # expected output
                    in_out_layers = training_data[index]
                    (dw2, db2) = self.calculateNegGradient(in_out_layers)

                    if batch_size == 1:
                        dw, db = dw2, db2
                        break
                    else:
                        # add the gradient due to weights and biases
                        for index2 in range(0, self.nb_layer-1):
                            dw[index2] += dw2[index2]
                            db[index2] += db2[index2]

                for index in range(0, self.nb_layer-1):

                    # apply the gradient descent factor to all the weights
                    # and biases gradients
                    # morevover, divide by the batch_size to normalize the grad
                    dw[index] *= 0.1*gdf_func(nb_repetition, gdf_param)/batch_size
                    db[index] *= 0.1*gdf_func(nb_repetition, gdf_param)/batch_size

                    # OVERHERE
                    # print("Sum dw = ", np.sum(dw[index]))
                    # print("Sum db = ", np.sum(db[index]))

                    # print(dw[index])
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

        dweight_matrix = np.zeros(shape=(row, column))
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
        training_output = values_layers[self.nb_layer-1]

        perfect_output = in_out_layers[1]

        # cost function = sum (a_j^N - y)^2 we do not really need that
        # cost = sum(CostFunction(training_output, perfect_output))

        # (nb_layer - 1) matrix and biases vectors composed the neural network
        dweights = [None]*(self.nb_layer+1)
        dbiases = [None]*(self.nb_layer+1)

        der_cost_to_a = DerCostFunction(training_output, perfect_output)
        # from (nb_layer - 2) to 0
        for index in range(self.nb_layer-2, -1, -1):
            # extract the good squishing function for this layer
            # [2] means the derivative function not normal or inverse one
            DerFunction = self.squishing_funcs[index][2]
            der_func_z = DerFunction(z_values[index])
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
            # extract the good squishing function for this layer
            # [0] means the function not inverse or derivative one
            Function = self.squishing_funcs[index][0]
            new_array = Function(z)
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
            # extract the good squishing function for this layer
            # [0] means the function not inverse or derivative one
            Function = self.squishing_funcs[index][0]
            new_array = Function(z)
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
            # extract the good squishing function for this layer
            # [1] means the inverse function not inormal or derivative one
            InvFunction = self.squishing_funcs[index][1]
            new_array = invA.dot(InvFunction(new_array)-self.biases[index])
        # print(len(new_array))
        return new_array



    def save(self, dir_save):
        """
            Method used to save the neural network.
            In fact, it simply writes every matrix weights and biases in the
            document named doc_save.
        """
        for index in range(0, self.nb_layer-1):
            np.savez(dir_save+"/"+str(index), w=self.weights[index],
                    b=self.biases[index])



    def test(self, testing_data):
        """
            Method used to test the neural network after its training.
        """
        nb_correct = 0
        nb_test = len(testing_data)

        bar3 = ProgressBar()
        print("Test process")
        for element in bar3(testing_data):
            input_layer = element[0]
            perfect_output = element[1]
            generated_output = self.generateOuputLayer(input_layer)
            # not really necessary just for information
            cost_array = CostFunction(generated_output, perfect_output)
            cost = sum(cost_array)
            # print(cost)
            index_max_value = np.argmax(generated_output)

            expected_answer = np.argmax(perfect_output)
            if index_max_value == expected_answer:
                nb_correct += 1

        print("Generated ouput", generated_output)
        print("Number =", index_max_value)
        print("Perfect output", perfect_output)
        print("Number =", expected_answer)

        return (nb_test-nb_correct)/nb_test


    def display(self):
        for index in range(0, nb_layer-1):
            self.weights[index]

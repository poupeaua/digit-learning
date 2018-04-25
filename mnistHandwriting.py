#!/usr/bin/env python3
# pylint: skip-file

"""
    File mnistHandwriting.py used to load the data.
"""


from struct import unpack
from PIL import Image
import numpy as np
from externalFunc import progressbar
# from progressbar import *


def MNISTexample(startN, howMany, bTrain=True, only01=False):
    """
        This function reads data from the MNIST handwriting files.  To use this
        you need to download the MNIST files from :
        http://yann.lecun.com/exdb/mnist/
        The train file has 60,000 examples, and the test has 10,000.

        The data format is described towards the bottom of the page, but this
        function MNISTexample takes care of reading it for you.  It will return
        a list of labeled examples.  Each image in the training files are
        28x28 grayscale pictures, so the input for each example will have
        28*28=784 different inputs.  In the function, I have scaled these values
        so they are each between 0.0 and 1.0.  Each of the images could be any
        of the digits 0, 1, ..., 9.  So we should make a neural net that has 10
        different output neurons, each one testing whether the input corresponds
        to one of the digits.  In the examples that are returned by MNISTexample,
        y is a list of length 10, with a 1 in the spot for the correct digit,
        and 0's elsewhere.

        NOTE: you should try running the MNISTexample function to get
        just a single example, like MNISTexample(0,1), to make sure it looks
        right.  The header information should look like what they talked about
        on the website, and you can print those values in the function below
        to make sure it looks like it is working.  If it seems messed up,
        let Jeff know.

        Inputs to this function :

        -> startN says which example to start reading from in the file.

        -> howMany says how many examples to read from that point.

        -> bTrain : says whether to read from the train file for from the test
        file. For the test file, they made sure the examples came from different
        people than were used for producing the training file.

        -> only01 : is set to True to only return examples where the correct
        answer is 0 or 1.  This makes the task simpler because we're only trying
        to distinguish between two things instead of 10, meaning we won't need
        to train as long to start getting good results.
    """
    if bTrain:
        fImages = open("data/train-images-idx3-ubyte",'rb')
        fLabels = open('data/train-labels-idx1-ubyte','rb')
        type_data = "training data : "
    else:
        fImages = open('data/t10k-images-idx3-ubyte','rb')
        fLabels = open('data/t10k-labels-idx1-ubyte','rb')
        type_data = "testing data  : "

    # read the header information in the images file.
    s1, s2, s3, s4 = fImages.read(4), fImages.read(4), fImages.read(4), fImages.read(4)
    mnIm = unpack('>I',s1)[0]
    numIm = unpack('>I',s2)[0]
    rowsIm = unpack('>I',s3)[0]
    colsIm = unpack('>I',s4)[0]
    # seek to the image we want to start on
    fImages.seek(16+startN*rowsIm*colsIm)

    # read the header information in the labels file and seek to position
    # in the file for the image we want to start on.
    mnL = unpack('>I',fLabels.read(4))[0]
    numL = unpack('>I',fLabels.read(4))[0]
    fLabels.seek(8+startN)

    T = [] # list of (input, correct label) pairs

    # iteration on all the images with a progress bar to follow the process
    # bar = ProgressBar()

    for _ in progressbar(range(0, howMany), "Importing " + type_data, 40):
        # get the input from the image file
        x = []
        for i in range(0, rowsIm*colsIm):
            val = unpack('>B',fImages.read(1))[0]
            x.append(val/255.0)

        # get the correct label from the labels file.
        val = unpack('>B',fLabels.read(1))[0]
        y = []
        for i in range(0,10):
            if val==i: y.append(1)
            else: y.append(0)

        # if only01 is True, then only add this example if 0 or 1 is the
        # correct label.
        if not only01 or y[0]==1 or y[1]==1:
            x = np.array(x)
            y = np.array(y)
            T.append((x,y))

    fImages.close()
    fLabels.close()

    return T



def writeMNISTimage(T, display, antialias=False):
    """
        This function is not needed to do the training, but just in case you
        want to see what one of the training images looks like. This will take
        the training data that was produced from the MNSTexample function and
        write it out to a file that you can look at to see what the picture
        looks like. It will write out a separate image for each thing in the
        training set.

        Inputs to this function :

        -> T

        -> display

        -> antialias
    """
    # note that you need to have the Python Imaging Library installed to
    # run this function.  If you search for it online, you'll find it.
    for i in range(0, len(T)):
        im = Image.new('L',(28,28))
        pixels = im.load()
        for x in range(0,28):
            for y in range(0,28):
                pixels[x,y] = int(T[i][0][x+y*28]*255)
        if antialias:
            im = im.resize((500,500), Image.ANTIALIAS)
        else:
            im = im.resize((500,500))
        im.save('./img/mnistFile'+str(i)+'.bmp')
        if display:
            im.show()


# data = MNISTexample(0, 1)
# # print(data[0][1])
# writeMNISTimage(data, True)

# This if my function to do the learning on the MNIST handwriting
# data.  It will not work for you unless you set your functions up like
# mine.  But you can look at it to get an idea of what worked for me.
# def learnMNIST():
#     # this creates a network with all 0 weights initially that has
#     # 28*28 inputs, 300 hidden nodes, and 10 output nodes.  You could try
#     # other choices for the number of hidden nodes, or could try a
#     # deeper network with fewer hidden nodes.
#     network = createNetwork(28*28,[300,10])
#
#     # get the training examples.  I am just doing it with 0 and 1 values.
#     # Out of the first 100 examples in the file, 27 were 0 and 1.  The
#     # learning algorithm takes kind of a long time with a network this big,
#     # so I wanted to see if it would do reasonable with just a small number
#     # of examples to train on.
#     T = MNISTexample(0,100,only01=True)
#
#     # do the back propogation to get the weights.  tries=10 goes into my
#     # backprop function to tell it to go through all the examples 10 times.
#     # So that is running the backprop 10*100 times total.
#     network = backPropLearning(T,network,tries=10)
#
#     # now test the network on some examples that were not part of the
#     # learning process.  There are 22 0/1 examples in the range from
#     # 0->100 in the test file.
#     T = MNISTexample(0,100,only01=True,bTrain=False)
#     # go through each of those things in T, run feedforward to see
#     # what it ends up with on the example, and keep a count of how many
#     # it got correct.  For mine, it got 22 out of 22 correct.  For other
#     # examples I tried from the test and training files, it seemed to
#     # be normally getting about 80% correct.  This could probably be made
#     # higher by experimenting with how many hidden nodes to use, how many
#     # levels to use, and training on more examples.
#     correct = 0
#     for i in range(0, len(T)):
#         x, y = T[i]
#         inn, a = feedForward(x,y,network)
#         if y[0]==1 and a[2][0] > a[2][1]: correct += 1
#         elif y[1]==1 and a[2][1] > a[2][0]: correct += 1
#     print('Total examples tried: ' + str(len(T)))
#     print('Total correct: ' + str(correct))

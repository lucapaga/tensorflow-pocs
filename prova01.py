#!/usr/bin/env python

# PREREQUISITES:
# --------------
#  pip install --upgrade tensorflow
#  pip install --upgrade matplotlib
#
#  Install scipy and numpy+mkl
#   : http://www.lfd.uci.edu/~gohlke/pythonlibs/
#   : download for your Python version (cpXX) and platform (win / amd64 / ...)
#   : pip install downloaded_file
#

import tensorflow as tf
import numpy as np

#from scipy import ndimage
import matplotlib.image as mpimg

from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))
    #-----------------------------------
    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    #-----------------------------------
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    #-----------------------------------
    return tf.matmul(h2, w_o)


print("Loading MNIST data ...")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

print("Preparing INPUT Neurons (placeholder): 784 items to 'see' all the 28x28 pixels of the image")
X = tf.placeholder("float", [None, 784])

print("Preparing OUTPUT Neurons (placeholder): 10 items to classify image into digits from '0' to '9'")
Y = tf.placeholder("float", [None, 10])

print("First HIDDEN Layer is of 1568 neurons, initializing weights for 784->1568 matmul ...")
w_h = init_weights([784, 1568])
print("Second HIDDEN Layer is of 1568 neurons as well, initializing weights for 1568->1568 matmul ...")
w_h2 = init_weights([1568, 1568])
print("The Output Layer backs to 10 neurons, initializing weights for 1568->10 matmul ...")
w_o = init_weights([1568, 10])

print("There are 'keep_input' and 'keep_hidden' parameters for 'dropout' functions that must be investigated further! Initializing placeholders")
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

print("")
print("-----------------------------------------------------------------------------------------------------------------------")
print("Building up the model:       INPUT > DROPOUT | (MATMUL + ReLU) > DROPOUT | (MATMUL + ReLU) > DROPOUT + MATMUL > OUTPUT")
print("                           | INPUT |       HIDDEN LAYER 1      |       HIDDEN LAYER 2      |           OUTPUT          |")
print("-----------------------------------------------------------------------------------------------------------------------")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

print("Training Configuration:")
print("  - Softmax + logits")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
print("  - RMS Optmizer")
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

print("")
predict_op = tf.argmax(py_x, 1)

print("")
print("PREPARING HANDMADE IMAGE, NOT FROM MNIST!")

img = mpimg.imread('MY_data/aDigit_4.png')
print("     ORIGINAL IMAGE: ", img)
img = img[:,:,0] # slicking: picking only one channel of RGB (black'n'white!)
print("       SLICED IMAGE: ", img)
img = img.flatten('C')
print("      IMAGE AS 1D V: ", img)


#testSample = teX[15]
#testData = teX
#expected = teY[15]
#benchmark = teY
testSample = img
testData = [testSample]
expected = [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.] # il 3
benchmark = [expected]

print("")
print(" EXPECTED CLASSIFICATION VECTOR: ", expected)
print("EXPECTED CLASSIFICATION OUTCOME: ", np.argmax(expected))

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        print("")
        print("--------------------[ Iteration '", i, "' ]---------------------------")
        print("------------------------------------------------------------------")

        print("[", i, "] - Training session in progress ...")
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={
                                        X: trX[start:end],
                                        Y: trY[start:end],
                                        p_keep_input: 0.8,
                                        p_keep_hidden: 0.5})

        print("")
        print("------------------------------------------------------------------")
        print("[", i, "] -    TEST DATA: ", testSample)
        print("[", i, "] -       LENGTH: ", len(testSample))
        print("[", i, "] -     EXPECTED: ", expected)

        elaborated = sess.run(py_x, feed_dict={X: [testSample], p_keep_input: 1.0, p_keep_hidden: 1.0})[0]

        print("[", i, "] -       OUTPUT: ", elaborated)
        print("[", i, "] -       ANSWER: ", np.argmax(elaborated))
        print("[", i, "] -  ERROR/MNIST: ", np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={
                                                    X: teX,
                                                    p_keep_input: 1.0,
                                                    p_keep_hidden: 1.0})))

        print("------------------------------------------------------------------")

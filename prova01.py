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

# ------------------------------------------------------------------------------
# --    WEIGHT INITIALIZATION: GAUSSIAN
# ------------------------------------------------------------------------------
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: MODEL
# ------------------------------------------------------------------------------
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    # -- [ INPUT LAYER      ] -----------------------------------------------
    X = tf.nn.dropout(X, p_keep_input)
    # -- [ 1st HIDDEN LAYER ] -----------------------------------------------
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    # -- [ 2nd HIDDEN LAYER ] -----------------------------------------------
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    # -- [ OUTPUT LAYER     ] -----------------------------------------------
    return tf.matmul(h2, w_o)
# ------------------------------------------------------------------------------


print("Loading MNIST data ...")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: INITIALIZATION
# ------------------------------------------------------------------------------
# -- [ INPUT LAYER      ] -----------------------------------------------
input_layer_neurons=784
X = tf.placeholder("float", [None, input_layer_neurons ])
p_keep_input = tf.placeholder("float")
# -- [ 1st HIDDEN LAYER ] -----------------------------------------------
first_h_layer_neurons=400
w_h =  init_weights([input_layer_neurons,    first_h_layer_neurons  ])
p_keep_hidden = tf.placeholder("float")
# -- [ 2nd HIDDEN LAYER ] -----------------------------------------------
second_h_layer_neurons=400
w_h2 = init_weights([first_h_layer_neurons,  second_h_layer_neurons ])
#p_keep_hidden = tf.placeholder("float")
# -- [ OUTPUT LAYER     ] -----------------------------------------------
output_layer_neurons=10
w_o =  init_weights([second_h_layer_neurons, output_layer_neurons   ])
Y = tf.placeholder("float", [None, output_layer_neurons])
# -- [ TRAINING SPEED   ] -----------------------------------------------
training_speed=0.001
# -- [ TRAINING MODEL   ] -----------------------------------------------
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(training_speed).minimize(cost)
predict_op = tf.argmax(py_x, 1)
# ------------------------------------------------------------------------------

print("")
print("------------------------------------------------------------------------------------------------------------------------")
print(" Building up the model:      INPUT > DROPOUT | (MATMUL + ReLU) > DROPOUT | (MATMUL + ReLU) > DROPOUT + MATMUL > OUTPUT ")
print("                           | INPUT |       HIDDEN LAYER 1      |       HIDDEN LAYER 2      |           OUTPUT          |")
print("------------------------------------------------------------------------------------------------------------------------")
print("  - Neurons >      INPUT LAYER: ", input_layer_neurons,    " - (placeholder) items to 'see' all the 28x28 pixels of the image")
print("  - Neurons > 1st HIDDEN LAYER: ", first_h_layer_neurons,  " - (variable)    initializing weights for ", input_layer_neurons, "->", first_h_layer_neurons, " matmul ...")
print("  - Neurons > 2nd HIDDEN LAYER: ", second_h_layer_neurons, " - (variable)    initializing weights for ", first_h_layer_neurons, "->", second_h_layer_neurons, " matmul ...")
print("  - Neurons >     OUTPUT LAYER: ", output_layer_neurons,   " - (placeholder) items to classify image into digits from '0' to '9'")
print("")
print("  - There are 'keep_input' and 'keep_hidden' parameters for 'dropout' functions that must be investigated further!")
print("")
print("  - TRAINING Configuration:")
print("     * Softmax + logits")
print("     * RMS Optmizer, SPEED: ", training_speed)
print("")

img = mpimg.imread('MY_data/aDigit_SEI.png')
#print("     ORIGINAL IMAGE: ", img)
img = img[:,:,0] # slicking: picking only one channel of RGB (black'n'white!)
#print("       SLICED IMAGE: ", img)
img = img.flatten('C')
#print("      IMAGE AS 1D V: ", img)


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

# Add ops to save and/or restore all the variables.
saver = tf.train.Saver()

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

        do_save=0
        if i == 0:
            do_save = 1
        elif i == 4:
            do_save = 1
        elif i == 9:
            do_save = 1
        elif i == 99:
            do_save = 1

        if do_save == 1:
            save_path = 'sessions/prova01/' + str(i+1) + '/model'
            print("Saving Model ... on ", save_path)
            save_path = saver.save(sess, save_path)
            print("Model saved in file: %s" % save_path)

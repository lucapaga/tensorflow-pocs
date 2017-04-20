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
# --    NETWORK ARCHITECTURE: SETTINGS
# ------------------------------------------------------------------------------
input_layer_neurons=784                 # number of input neurons
first_h_layer_neurons=500               # number of neurons into 1st hidden layer
#second_h_layer_neurons=1568             # number of neurons into 1st hidden layer
output_layer_neurons=10                 # number of outputs (output neurons?)
training_speed=0.001                    # training speed
weight_init_gauss_std_dev_value=0.01    # shape of the gaussian distribution used to init weights

# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: WEIGHT INITIALIZATION: GAUSSIAN
# ------------------------------------------------------------------------------
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=weight_init_gauss_std_dev_value))
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: MODEL
# ------------------------------------------------------------------------------
#def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
def model(X, w_h, w_o, p_keep_input, p_keep_hidden):
    # -- [ INPUT LAYER      ] -----------------------------------------------
    X = tf.nn.dropout(X, p_keep_input)
    # -- [ 1st HIDDEN LAYER ] -----------------------------------------------
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    # -- [ 2nd HIDDEN LAYER ] -----------------------------------------------
#    h2 = tf.nn.relu(tf.matmul(h, w_h2))
#    h2 = tf.nn.dropout(h2, p_keep_hidden)
    # -- [ OUTPUT LAYER     ] -----------------------------------------------
#    return tf.matmul(h2, w_o)
    return tf.matmul(h, w_o)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: CONFIGURATION
# ------------------------------------------------------------------------------
# -- [ INPUT LAYER      ] -----------------------------------------------
X = tf.placeholder("float", [None, input_layer_neurons ])
p_keep_input = tf.placeholder("float")
# -- [ 1st HIDDEN LAYER ] -----------------------------------------------
w_h =  init_weights([input_layer_neurons,    first_h_layer_neurons  ])
p_keep_hidden = tf.placeholder("float")
# -- [ 2nd HIDDEN LAYER ] -----------------------------------------------
#w_h2 = init_weights([first_h_layer_neurons,  second_h_layer_neurons ])
#p_keep_hidden = tf.placeholder("float")
# -- [ OUTPUT LAYER     ] -----------------------------------------------
#w_o =  init_weights([second_h_layer_neurons, output_layer_neurons   ])
w_o =  init_weights([first_h_layer_neurons,  output_layer_neurons   ])
Y = tf.placeholder("float", [None, output_layer_neurons])
# -- [ TRAINING MODEL   ] -----------------------------------------------
#py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
py_x = model(X, w_h, w_o, p_keep_input, p_keep_hidden)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
#train_op = tf.train.RMSPropOptimizer(training_speed).minimize(cost)
#predict_op = tf.argmax(py_x, 1)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# --   HERE STARTS THE PROGRAM!
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Select what checkpoint you want to use
#  (it's a number that recalls the number of cycles performed to train the net)
# 1, 5, 10, 100
training_progress=10

# Select the IMAGE DIGIT you want to try to recognize
img = mpimg.imread('../MY_data/aDigit_UNO.png')
# Tell me what digit is represented by the image
expected = 1.

# IMAGE (PRE)PROCESSING
img = img[:,:,0]        # slicking: picking only one channel of RGB (black'n'white!)
img = img.flatten('C')  # from matrix to vector
testSample = img

#print("")
#print("EXPECTED CLASSIFICATION OUTCOME: ", expected)

# Launch the graph in a session
with tf.Session() as sess:
    # Seems to need this!
    tf.global_variables_initializer().run()

    # Restoring TRAINED MODEL > 'sessions/prova01/' + str(i+1) + '/model'
    new_saver = tf.train.import_meta_graph('sessions/' + str(training_progress) + '/model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('sessions/' + str(training_progress) + '/'))

#    print("Global Variables: ", tf.global_variables())
#    for Gv in tf.global_variables():
#        print("Vars/x: ", Gv)
#        print("Vars/R: ", sess.run(Gv))
#
#    print("Local Variables: ", tf.local_variables())
#    for Gv in tf.local_variables():
#        print("Vars/x: ", Gv)
#        print("Vars/R: ", sess.run(Gv))
#
#    all_vars = tf.get_collection('vars')
#    print("ALL VARS: ", all_vars)
#    for v in all_vars:
#        print("Vars/x: ", v)
#        print("Vars/R: ", sess.run(v))

    # Try!!
    print("")
    print("------------------------------------------------------------------")
#    print("  TEST DATA: ", testSample)
#    print("     LENGTH: ", len(testSample))
    print("   PROGRESS: ", training_progress)
    print("   EXPECTED: ", expected)

#    py_x = tf.get_collection('Y')[0]
    elaborated = sess.run(py_x, feed_dict={X: [testSample], p_keep_input: 0.8, p_keep_hidden: 0.5})[0]

    print("     OUTPUT: ", elaborated)
    print("     ANSWER: ", np.argmax(elaborated))
    print("------------------------------------------------------------------")

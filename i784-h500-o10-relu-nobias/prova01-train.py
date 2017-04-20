#!/usr/bin/env python

import tensorflow as tf
import numpy as np
#from scipy import ndimage
#import matplotlib.image as mpimg
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
#print("  - Neurons > 2nd HIDDEN LAYER: ", second_h_layer_neurons, " - (variable)    initializing weights for ", first_h_layer_neurons, "->", second_h_layer_neurons, " matmul ...")
print("  - Neurons >     OUTPUT LAYER: ", output_layer_neurons,   " - (placeholder) items to classify image into digits from '0' to '9'")
print("")
print("  - There are 'keep_input' and 'keep_hidden' parameters for 'dropout' functions that must be investigated further!")
print("")
print("  - TRAINING Configuration:")
print("     * Softmax + Logits")
print("     * RMS Optmizer      -> SPEED: ", training_speed)
print("")


# Training data ----------------------------------------------------------------
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Add ops to save and/or restore all the variables -----------------------------
saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables -------------------------------------
    tf.global_variables_initializer().run()

    for i in range(100):
        print("")
        print("--------------------[ Iteration '" + str(i) + "' ]---------------------------")
        print("------------------------------------------------------------------")

        print("[", i, "] - Training session in progress ...")
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={
                                        X: trX[start:end],
                                        Y: trY[start:end],
                                        p_keep_input: 0.8,
                                        p_keep_hidden: 0.5})

        print("")
        print("[", i, "] -  ERROR/MNIST: ", np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={
                                                    X: teX,
                                                    p_keep_input: 1.0,
                                                    p_keep_hidden: 1.0})))


        do_save=0
        if i == 0:
            do_save = 1
        elif i == 4:
            do_save = 1
        elif i == 9:
            do_save = 1
        elif i == 24:
            do_save = 1
        elif i == 49:
            do_save = 1
        elif i == 59:
            do_save = 1
        elif i == 69:
            do_save = 1
        elif i == 79:
            do_save = 1
        elif i == 89:
            do_save = 1
        elif i == 99:
            do_save = 1

        if do_save == 1:
            save_path = 'sessions/' + str(i+1) + '/model'
            print("Saving Model ... on ", save_path)
            save_path = saver.save(sess, save_path)
            print("Model saved in file: %s" % save_path)

        print("------------------------------------------------------------------")

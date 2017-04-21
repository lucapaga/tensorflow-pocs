#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response, url_for
#from flask.ext.httpauth import HTTPBasicAuth

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg


#-------------------------------------------------------------------------------
#---   CONFIGURATION
#-------------------------------------------------------------------------------
context_path='/dnn/api/v1.0'
default_network_architecture='i784-h400-h400-o10-relu-nobias'
default_network_progress=10
#-------------------------------------------------------------------------------

app = Flask(__name__, static_url_path = "")
#auth = HTTPBasicAuth()

#@auth.get_password
#def get_password(username):
#    if username == 'miguel':
#        return 'python'
#    return None

#@auth.error_handler
#def unauthorized():
#    return make_response(jsonify( { 'error': 'Unauthorized access' } ), 403)
    # return 403 instead of 401 to prevent browsers from displaying the default auth dialog

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)


#---------[ IMPLEMENTATION ]----------------------------------------------------

# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: SETTINGS (defaults)
# ------------------------------------------------------------------------------
input_layer_neurons=784                 # INPUT NEURONS
first_h_layer_neurons=1568              # 1st HIDDEN LAYER NEURONS
second_h_layer_neurons=1568             # 2nd HIDDEN LAYER NEURONS
output_layer_neurons=10                 # OUTPUT NEURONS
training_speed=0.001                    # TRAINING SPEED
weight_init_gauss_std_dev_value=0.01    # GAUSSIAN DISTRIBUTION used to INIT WEIGHTS

# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: TAILOR SETTINGS
# ------------------------------------------------------------------------------
def tailor_settings(net_architecture):
    if net_architecture == 'i784-h1568-h1568-o10-relu-nobias':
        input_layer_neurons=784                 # INPUT NEURONS
        first_h_layer_neurons=1568              # 1st HIDDEN LAYER NEURONS
        second_h_layer_neurons=1568             # 2nd HIDDEN LAYER NEURONS
        output_layer_neurons=10                 # OUTPUT NEURONS
    elif net_architecture == 'i784-h400-h400-o10-relu-nobias':
        input_layer_neurons=784                 # INPUT NEURONS
        first_h_layer_neurons=400               # 1st HIDDEN LAYER NEURONS
        second_h_layer_neurons=400              # 2nd HIDDEN LAYER NEURONS
        output_layer_neurons=10                 # OUTPUT NEURONS
    elif net_architecture == 'i784-h500-o10-relu-nobias':
        input_layer_neurons=784                 # INPUT NEURONS
        first_h_layer_neurons=500               # 1st HIDDEN LAYER NEURONS
        second_h_layer_neurons=0                # 2nd HIDDEN LAYER NEURONS
        output_layer_neurons=10                 # OUTPUT NEURONS

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: WEIGHT INITIALIZATION: GAUSSIAN
# ------------------------------------------------------------------------------
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=weight_init_gauss_std_dev_value))
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: MODELs
# ------------------------------------------------------------------------------
def model_2hl(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)      # INPUT LAYER
    h = tf.nn.relu(tf.matmul(X, w_h))       # 1st HIDDEN LAYER / STAGE A
    h = tf.nn.dropout(h, p_keep_hidden)     # 1st HIDDEN LAYER / STAGE B
    h2 = tf.nn.relu(tf.matmul(h, w_h2))     # 2nd HIDDEN LAYER / STAGE A
    h2 = tf.nn.dropout(h2, p_keep_hidden)   # 2nd HIDDEN LAYER / STAGE B
    return tf.matmul(h2, w_o)               # OUTPUT LAYER
# ------------------------------------------------------------------------------
def model_1hl(X, w_h, w_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)      # INPUT LAYER
    h = tf.nn.relu(tf.matmul(X, w_h))       # 1st HIDDEN LAYER / STAGE A
    h = tf.nn.dropout(h, p_keep_hidden)     # 1st HIDDEN LAYER / STAGE B
    return tf.matmul(h, w_o)                # OUTPUT LAYER
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# --    IMAGE PROCESSING
# ------------------------------------------------------------------------------
def load_image(image_file_name):
    img = mpimg.imread('../MY_data/' + image_file_name)
    img = img[:,:,0]        # slicing: picking only one channel of RGB (black'n'white!)
    img = img.flatten('C')  # from matrix to vector
    return img
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# --    NETWORK ARCHITECTURE: CLASSIFICATION
# ------------------------------------------------------------------------------
def use_dnn(net_architecture, image_file_name, training_progress):
    testSample = load_image(image_file_name)

    if net_architecture == '':
        net_architecture = 'i784-h400-h400-o10-relu-nobias'
        print('NET Architecture (defaulted): ' + net_architecture)
    else:
        print('NET Architecture ( defined ): ' + net_architecture)

    if net_architecture == 'i784-h1568-h1568-o10-relu-nobias' or net_architecture == 'i784-h400-h400-o10-relu-nobias':
        X = tf.placeholder("float", [None, input_layer_neurons ])                   # INPUT             / SENSORS
        p_keep_input = tf.placeholder("float")                                      # INPUT             / DROPOUT
        w_h =  init_weights([input_layer_neurons,    first_h_layer_neurons  ])      # 1st HIDDEN LAYER  / WEIGHTS
        p_keep_hidden = tf.placeholder("float")                                     # 1st HIDDEN LAYER  / DROPOUT
        w_h2 = init_weights([first_h_layer_neurons,  second_h_layer_neurons ])      # 2nd HIDDEN LAYER  / WEIGHTS
        #p_keep_hidden = tf.placeholder("float")                                    # 2nd HIDDEN LAYER  / DROPOUT
        w_o =  init_weights([second_h_layer_neurons, output_layer_neurons   ])      # OUTPUT LAYER      / WEIGHTS
        Y = tf.placeholder("float", [None, output_layer_neurons])                   # OUTPUT LAYER      / DISPLAY
        # --------------------------------------------------------------------------------------------------------
        py_x = model_2hl(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)            # MODEL
        # --------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------
    elif net_architecture == 'i784-h500-o10-relu-nobias':
        X = tf.placeholder("float", [None, input_layer_neurons ])                   # INPUT             / SENSORS
        p_keep_input = tf.placeholder("float")                                      # INPUT             / DROPOUT
        w_h =  init_weights([input_layer_neurons,    first_h_layer_neurons  ])      # 1st HIDDEN LAYER  / WEIGHTS
        p_keep_hidden = tf.placeholder("float")                                     # 1st HIDDEN LAYER  / DROPOUT
        w_o =  init_weights([first_h_layer_neurons, output_layer_neurons   ])       # OUTPUT LAYER      / WEIGHTS
        Y = tf.placeholder("float", [None, output_layer_neurons])                   # OUTPUT LAYER      / DISPLAY
        # --------------------------------------------------------------------------------------------------------
        py_x = model_1hl(X, w_h, w_o, p_keep_input, p_keep_hidden)                   # MODEL
        # --------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver_path = '../' + default_network_architecture + '/sessions/' + str(training_progress)
        new_saver = tf.train.import_meta_graph(saver_path + '/model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(saver_path + '/'))
        elaborated = sess.run(py_x, feed_dict={X: [testSample], p_keep_input: 0.8, p_keep_hidden: 0.5})[0]
        print('     elaborated=', elaborated)
        print(' classification=', np.argmax(elaborated))
        return np.argmax(elaborated)


@app.route(context_path + '/classify/<int:training_progress>/<string:image_name>', methods = ['GET'])
#@auth.login_required
def guess_number(image_name, training_progress, net_architecture=default_network_architecture):
    print('[' + context_path + '/classify] --------------------------------------')
    print('[' + context_path + '/classify]        IMAGE NAME: ' + image_name)
    print('[' + context_path + '/classify] TRAINING PROGRESS: ' + str(training_progress))
    print('[' + context_path + '/classify]   ... EVALUATING ...')
    guessed_val = use_dnn(net_architecture, image_name, training_progress);
    print('[' + context_path + '/classify]         THIS IS A: ' + str(guessed_val))
    print('[' + context_path + '/classify] ---<END>------------------------------')
    return jsonify( { 'guessed': str(guessed_val) } )

#-------------------------------------------------------------------------------

'''  ALL COMMENTED OUT!
#---------[ FROM THE EXAMPLE ]--------------------------------------------------
tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]
#------------------------------------------
def make_public_task(task):
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id = task['id'], _external = True)
        else:
            new_task[field] = task[field]
    return new_task
#------------------------------------------
@app.route('/todo/api/v1.0/tasks', methods = ['GET'])
#@auth.login_required
def get_tasks():
    return jsonify( { 'tasks': map(make_public_task, tasks) } )
#------------------------------------------
@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods = ['GET'])
#@auth.login_required
def get_task(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    if len(task) == 0:
        abort(404)
    return jsonify( { 'task': make_public_task(task[0]) } )
#------------------------------------------
@app.route('/todo/api/v1.0/tasks', methods = ['POST'])
#@auth.login_required
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify( { 'task': make_public_task(task) } ), 201
#------------------------------------------
@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods = ['PUT'])
#@auth.login_required
def update_task(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and type(request.json['title']) != unicode:
        abort(400)
    if 'description' in request.json and type(request.json['description']) is not unicode:
        abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description', task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify( { 'task': make_public_task(task[0]) } )
#------------------------------------------
@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods = ['DELETE'])
#@auth.login_required
def delete_task(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    if len(task) == 0:
        abort(404)
    tasks.remove(task[0])
    return jsonify( { 'result': True } )
#-------------------------------------------------------------------------------
'''


#---------[ THIS MUST BE PRESENT ]----------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug = True)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

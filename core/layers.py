import tensorflow as tf
import math

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.contrib.layers.xavier_initializer()([3, 3, size_in, size_out]), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)

        return conv + b

def activation_layer(input, name="act"):
    with tf.name_scope(name):
        act = tf.nn.relu(input)
        tf.summary.histogram("activations", act)
        return act

def pool_layer(input, ksize, strides, name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding="SAME")

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.contrib.layers.xavier_initializer()([size_in, size_out]), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)

        return tf.matmul(input, w) + b

def factor(n):
    nsqrt = math.ceil(math.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n/val)
        if val2 * val == float(n):
            solution = True
        else:
            val -= 1
    return int(val), int(val2)

def feature_map(layer):
    _, layer_width, layer_height, n_filters = layer.get_shape().as_list()
    features = tf.slice(layer, (0, 0, 0, 0), (1, -1, -1, -1))
    features = tf.reshape(features, (layer_height, layer_width, n_filters))

    grid_width, grid_height = factor(n_filters)
    features = tf.reshape(features, (layer_height, layer_width, grid_height, grid_width))
    features = tf.transpose(features, (2, 0, 3, 1))
    features = tf.reshape(features, (1, grid_height * layer_height, grid_width * layer_width, 1))

    return features

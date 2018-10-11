import tensorflow as tf
import math

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

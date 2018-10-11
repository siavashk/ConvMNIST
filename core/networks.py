import tensorflow as tf
from .layers import *

def simple_mnist_cnn():
    input_vector = tf.placeholder(tf.float32, shape=[None, 28 * 28 * 1], name="input_vector")
    input_layer = tf.reshape(input_vector, [-1, 28, 28, 1])

    labels = tf.placeholder(tf.int64, shape=[None, ])

    tf.summary.image('input_layer', input_layer, max_outputs=10)
    conv_1 = conv_layer(input_layer, 1, 64, "conv_1")
    act_1  = activation_layer(conv_1, "act_1")
    pool_1 = pool_layer(act_1, [1, 2, 2, 1], [1, 2, 2, 1], "pool_1")
    f_conv1 = feature_map(conv_1)
    tf.summary.image('f_conv1', f_conv1, max_outputs=64)

    conv2_1 = conv_layer(pool_1, 64, 64, "conv2_1")
    act2_1  = activation_layer(conv2_1, "act2_1")
    pool2_1 = pool_layer(act2_1, [1, 2, 2, 1], [1, 2, 2, 1], "pool2_1")
    f_conv2_1 = feature_map(conv2_1)
    tf.summary.image('f_conv2_1', f_conv2_1, max_outputs=64)

    conv2_2 = conv_layer(pool_1, 64, 64, "conv2_2")
    act2_2  = activation_layer(conv2_2, "act2_2")
    pool2_2 = pool_layer(act2_2, [1, 2, 2, 1], [1, 2, 2, 1], "pool2_2")
    f_conv2_2 = feature_map(conv2_2)
    tf.summary.image('f_conv2_2', f_conv2_2, max_outputs=64)

    conv3_1 = conv_layer(pool2_1, 64, 256, "conv3_1")
    act3_1  = activation_layer(conv3_1, "act3_1")
    pool3_1 = pool_layer(act3_1, [1, 2, 2, 1], [1, 2, 2, 1], "pool3_1")
    f_conv3_1 = feature_map(conv3_1)
    tf.summary.image('f_conv3_1', f_conv3_1, max_outputs=64) # Can't display them all

    conv3_2 = conv_layer(pool2_2, 64, 256, "conv3_2")
    act3_2  = activation_layer(conv3_2, "act3_2")
    pool3_2 = pool_layer(act3_2, [1, 2, 2, 1], [1, 2, 2, 1], "pool3_2")
    f_conv3_2 = feature_map(conv3_2)
    tf.summary.image('f_conv3_2', f_conv3_2, max_outputs=64) # Can't display them all

    pool_3    = tf.concat([pool3_1, pool3_2], axis=1)
    flattened = tf.reshape(pool_3, [-1, 7 * 7 * 512])

    fc1    = fc_layer(flattened, 7 * 7 * 512, 1000, "fc1")
    fc2    = fc_layer(fc1, 1000, 500, "fc2")
    logits = fc_layer(fc2, 500, 10, "fc3")

    return logits

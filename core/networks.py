import tensorflow as tf
from .layers import *

def compact_conv_layer(input_tensor, size_in=1, n_filter=64, pool_size=[1, 2, 2, 1], pool_stride=[1, 2, 2, 1],
    conv_scope="conv", activation_scope="act", pool_scope="pool", feature_summary="feature", max_features=64):
    conv = conv_layer(input_tensor, size_in, n_filter, conv_scope)
    act  = activation_layer(conv, activation_scope)
    pool = pool_layer(act, pool_size, pool_stride, pool_scope)
    features = feature_map(conv)
    tf.summary.image(feature_summary, conv, max_outputs=max_features)
    return pool, features

def simple_mnist_cnn():
    input_tensor = tf.placeholder(tf.float32, shape=[None, 28 * 28 * 1], name="input_tensor")
    input_image = tf.reshape(input_tensor, [-1, 28, 28, 1])

    labels = tf.placeholder(tf.int64, shape=[None, ])

    tf.summary.image('input_image', input_image, max_outputs=10)
    conv_1, f_conv1 = compact_conv_layer(input_image, conv_scope="conv_1", activation_scope="act_1",
        pool_scope="pool_1", feature_summary="f_conv1")

    conv2_1, f_conv2_1 = compact_conv_layer(conv_1, size_in=64, conv_scope="conv2_1", activation_scope="act2_1",
        pool_scope="pool2_1", feature_summary="f_conv2_1")

    conv2_2, f_conv2_2 = compact_conv_layer(conv_1, size_in=64, conv_scope="conv2_2", activation_scope="act2_2",
        pool_scope="pool2_2", feature_summary="f_conv2_2")

    conv3_1, f_conv3_1 = compact_conv_layer(conv2_1, size_in=64, n_filter=256, conv_scope="conv3_1", activation_scope="act3_1",
        pool_scope="pool3_1", feature_summary="f_conv3_1")

    conv3_2, f_conv3_2 = compact_conv_layer(conv2_2, size_in=64, n_filter=256, conv_scope="conv3_2", activation_scope="act3_2",
        pool_scope="pool3_2", feature_summary="f_conv3_2")

    conv_3 = tf.concat([conv3_1, conv3_2], axis=1)
    flattened =  tf.reshape(conv_3, [-1, 7 * 7 * 512])

    fc1    = fc_layer(flattened, 7 * 7 * 512, 1000, "fc1")
    fc2    = fc_layer(fc1, 1000, 500, "fc2")
    logits = fc_layer(fc2, 500, 10, "fc3")

    return input_tensor, logits, f_conv1, f_conv2_1, f_conv2_2, f_conv3_1, f_conv3_2

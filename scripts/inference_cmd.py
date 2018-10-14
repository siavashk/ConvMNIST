import tensorflow as tf
from PIL import Image
import argparse
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Do not clutter output.

def load_graph(filename):
    with tf.gfile.GFile(filename, "rb") as handle:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(handle.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="inference")
    return graph

def parse_args():
    parser = argparse.ArgumentParser(description='Inference Script for MNIST')
    parser.add_argument('--model', dest='model', type=str, default='model/frozenmodel.pb')
    parser.add_argument('--image', dest='image', type=str, default='images/1.jpg')
    return parser.parse_args()

def main():
    args = parse_args()
    graph = load_graph(args.model)

    input_tensor = graph.get_tensor_by_name('inference/input_tensor:0')
    softmax = graph.get_tensor_by_name('inference/output/softmax:0')
    predicted = tf.argmax(softmax, axis=1)
    
    input_image = np.asarray(Image.open(args.image)) / 255.0
    input_image = input_image[None, :]

    with tf.Session(graph=graph) as sess:
        [label, probability] = sess.run([predicted, softmax], feed_dict={ input_tensor: input_image })
        print('Predicted label: {:1d} , probability: {:.3f}'.format(label[0], probability[0][label][0]))

if __name__ == '__main__':
    main()

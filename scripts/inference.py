import tensorflow as tf
from core import MNISTBatch
import argparse
import os

def load_graph(filename):
    with tf.gfile.GFile(filename, "rb") as handle:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(handle.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="inference")
    return graph

def parse_args():
    parser = argparse.ArgumentParser(description='Inference Script for MNIST')
    parser.add_argument('--model', dest='model', type=str, default='/tmp/mnist/freeze/frozenmodel.pb')
    return parser.parse_args()

def main():
    args = parse_args()
    graph = load_graph(args.model)

    mnist = MNISTBatch()
    tensors, labels = mnist.test(batch_size=10)

    input_tensor = graph.get_tensor_by_name('inference/input_tensor:0')
    softmax = graph.get_tensor_by_name('inference/output/softmax:0')

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(tf.argmax(softmax), feed_dict={ input_tensor: tensors })
        print(predictions)
        print(labels)

if __name__ == '__main__':
    main()

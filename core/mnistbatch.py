import tensorflow as tf
import os

def get_batch(tensors, labels, index, batch_size=64):
    batch_start = index % tensors.shape[0]
    batch_end   = (index + batch_size) % tensors.shape[0]
    tensor_batch = tensors[batch_start:batch_end, :]
    label_batch = labels[batch_start:batch_end]
    incremented_index = index + batch_size + 1
    return tensor_batch, label_batch, incremented_index

class MNISTBatch(object):
    def __init__(self):
        (train_tensors, train_labels), (test_tensors, test_labels) = tf.keras.datasets.mnist.load_data()
        if train_tensors.shape[0] != train_labels.shape[0]:
            raise ValueError("There needs to be a one-to-one mapping between training tensors and labels.")
        if test_tensors.shape[0] != test_labels.shape[0]:
            raise ValueError("There needs to be a one-to-one mapping between testing tensors and labels.")

        train_tensors, test_tensors = train_tensors / 255.0, test_tensors / 255.0

        self.train_tensors = train_tensors
        self.train_labels = train_labels
        self.test_tensors = test_tensors
        self.test_labels = test_labels
        self.train_index = 0
        self.test_index = 0

    def train(self, batch_size=64):
        tensor_batch, label_batch, inc_index= get_batch(self.train_tensors, self.train_labels, self.train_index, batch_size)
        self.train_index = inc_index
        return tensor_batch, label_batch

    def test(self, batch_size=64):
        tensor_batch, label_batch, inc_index= get_batch(self.test_tensors, self.test_labels, self.test_index, batch_size)
        self.test_index = inc_index
        return tensor_batch, label_batch

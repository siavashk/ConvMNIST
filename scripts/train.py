import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from core import simple_mnist_cnn
from core import MNISTBatch
import math
import argparse
import os

def calculate_learning_rate(step=0, lr_max=1e-6, lr_min=1e-10, lr_decay=2e4):
    return lr_min + (lr_max - lr_min) * math.exp(-step / lr_decay)

def print_header():
    print('{}|{}|{}|{}|{}'.format('epoch', 'train loss','test loss',
        'train accuracy', 'test accuracy'))

def print_progress(epoch, train_loss, test_loss, train_accuracy, test_accuracy):
    print('{:03d}{:2}|{:4.4f}{:4}|{:4.4f}{:3}|{:.4f}{:8}|{:.3f}{:11} \
        '.format(epoch, '', train_loss, '', test_loss, '', train_accuracy, '',
        test_accuracy, ''))

def parse_args():
    parser = argparse.ArgumentParser(description='Traing Script for MNIST')
    parser.add_argument('--save', dest='save_dir', type=str, default='/tmp/mnist')
    parser.add_argument('--epochs', dest='epochs', type=int, default=600)
    parser.add_argument('--steps', dest='steps', type=int, default=100)
    parser.add_argument('--train-batch', dest='train_batch', type=int, default=100)
    parser.add_argument('--test-batch', dest='test_batch', type=int, default=1000)
    parser.add_argument('--lr-max', dest='lr_max', type=float, default=1e-6)
    parser.add_argument('--lr-min', dest='lr_min', type=float, default=1e-10)
    parser.add_argument('--lr-decay', dest='lr_decay', type=float, default=2e4)
    parser.add_argument('--checkpoint', dest='checkpoint', type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    train_dir = os.path.join(args.save_dir, 'train')
    test_dir = os.path.join(args.save_dir, 'test')
    ckpt_dir = os.path.join(args.save_dir, 'ckpt')
    freeze_dir = os.path.join(args.save_dir, 'freeze')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Do not clutter training progress.

    tf.reset_default_graph()

    learning_rate = tf.placeholder(tf.float32)

    input_tensor, logits, labels = simple_mnist_cnn()

    mnist = MNISTBatch()

    saver = tf.train.Saver()

    with tf.name_scope('output'):
        softmax = tf.nn.softmax(logits, name='softmax')

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(train_dir)
        test_writer = tf.summary.FileWriter(test_dir)

        train_writer.add_graph(sess.graph)
        summary = tf.summary.merge_all()

        test_tensor, test_labels = mnist.test(batch_size=args.test_batch)

        print_header()
        for epoch in range(args.epochs):
            train_tensor, train_labels = mnist.train(batch_size=args.train_batch)

            [train_loss, train_accuracy, train_summary] = sess.run([cross_entropy, accuracy, summary],
                feed_dict={input_tensor: train_tensor, labels: train_labels})

            [test_loss, test_accuracy, test_summary] = sess.run([cross_entropy, accuracy, summary],
                feed_dict={input_tensor: test_tensor, labels: test_labels})

            train_writer.add_summary(train_summary, epoch)
            test_writer.add_summary(test_summary, epoch)
            print_progress(epoch, train_loss, test_loss, train_accuracy, test_accuracy)

            for step in range(args.steps):
                global_step = step + epoch * args.steps
                lr = calculate_learning_rate(global_step, args.lr_max, args.lr_min, args.lr_decay)
                sess.run([train_step], feed_dict={input_tensor: train_tensor, labels: train_labels, learning_rate: lr})

            if epoch % args.checkpoint is 0:
                saver.save(sess, os.path.join(ckpt_dir, 'model_' + str(epoch) + '.ckpt'))

        ckpt_file = saver.save(sess, os.path.join(ckpt_dir, 'model'), global_step=global_step)
        tf.train.write_graph(sess.graph.as_graph_def(), freeze_dir, 'model.pb', as_text=False)
        freeze_graph.freeze_graph(os.path.join(freeze_dir, 'model.pb'), '', True,
            os.path.join(ckpt_dir, ckpt_file), 'output/softmax', 'save/restore_all',
            'save/Const:0', os.path.join(freeze_dir, 'frozenmodel.pb'), True, '')

if __name__ == '__main__':
    main()

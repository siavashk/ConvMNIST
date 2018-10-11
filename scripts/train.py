import tensorflow as tf
from core import simple_mnist_cnn
from core import MNISTBatch
import os

def shouldReport(step, report_step):
    return step % report_step is 0

def main():
    summary_dir = "/tmp/mnist"
    train_dir = os.path.join(summary_dir, 'train')
    test_dir = os.path.join(summary_dir, 'test')
    n_epochs = 100
    n_steps_per_epoch = 100
    n_report_per_epoch = 10
    report_step = n_steps_per_epoch // n_report_per_epoch
    batch_size = 64
    learning_rate = 1e-6

    tf.reset_default_graph()

    input_tensor, logits, labels = simple_mnist_cnn()

    mnist = MNISTBatch()

    with tf.device('/cpu:0'):
        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
            tf.summary.scalar("cross_entropy", cross_entropy)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(train_dir)
            test_writer = tf.summary.FileWriter(test_dir)
            train_writer.add_graph(sess.graph)
            summary = tf.summary.merge_all()

            for epoch in range(n_epochs):
                train_tensor, train_labels = mnist.train(batch_size=batch_size)
                test_tensor, test_labels = mnist.test(batch_size=batch_size)
                for step in range(n_steps_per_epoch):
                    sess.run([train_step], feed_dict={input_tensor: train_tensor, labels: train_labels})
                    if shouldReport(step, report_step):
                        [train_loss, train_accuracy, train_summary] = sess.run([cross_entropy, accuracy, summary],
                            feed_dict={input_tensor: train_tensor, labels: train_labels})

                        [test_loss, test_accuracy, test_summary] = sess.run([cross_entropy, accuracy, summary],
                            feed_dict={input_tensor: test_tensor, labels: test_labels})

                        train_writer.add_summary(train_summary, step + n_steps_per_epoch * epoch)
                        test_writer.add_summary(test_summary, step + n_steps_per_epoch * epoch)

                        print("epoch %d, step %d, train_loss %g, test_loss %g, train_accuracy %g, test_accuracy %g" % (epoch, step, train_loss, test_loss, train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()

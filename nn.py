import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import pandas as pd
import time

import preprocessing_photos


def metric(prediction, target):
    return roc_auc_score(target, prediction[:, 1])


start = time.time()

dataset = np.random.random_sample(size=(10000, 10))
labels = np.random.randint(low=0, high=1, size=(10000,))
submission_dataset = np.random.random_sample(size=(10000, 10))

test_interaction = pd.DataFrame()
test_interaction['user_id'] = np.arange(submission_dataset.shape[0])
test_interaction['photo_id'] = np.arange(submission_dataset.shape[0])

train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels)
train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(train_dataset, train_labels)

n_dim = train_dataset.shape[1]
num_epochs = 10
batch_size = train_dataset.shape[0] // 1000
num_steps = train_dataset.shape[0] // batch_size
report_interval = num_steps // 10

# hidden layers
f1_depth = n_dim * 10
f2_depth = n_dim * 3

# L2 regularization
lambdas = [0.3]
cost_history = []
cost_epoch_history = []
cost_cv_history = []
loss_history = []

for wl in lambdas:
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=[None, n_dim])
        tf_train_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        global_step = tf.Variable(0)  # count the number of steps taken.
        initial_learning_rate = 0.05
        final_learning_rate = 0.001
        decay_rate = 0.96
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=decay_rate,
                                                   decay_steps=num_steps / (
                                                   np.log(final_learning_rate / initial_learning_rate) / np.log(
                                                       decay_rate)))

        def variable_with_weight_loss(shape, wl, name):
            var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
            if wl is not None:
                weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
                tf.add_to_collection('losses', weight_loss)
            return var

        weights1 = variable_with_weight_loss([n_dim, f1_depth], wl, 'weights1')
        biases1 = tf.Variable(tf.ones([f1_depth]))
        weights2 = variable_with_weight_loss([f1_depth, f2_depth], wl, 'weights2')
        biases2 = tf.Variable(tf.ones([f2_depth]))
        readout_weights = variable_with_weight_loss([f2_depth, 1], wl, 'readout_weights')
        readout_biases = tf.Variable(tf.zeros([1]))

        def model(data):
            f1 = tf.nn.relu(tf.nn.xw_plus_b(data, weights1, biases1))
            f2 = tf.nn.relu(tf.nn.xw_plus_b(f1, weights2, biases2))
            logits = tf.matmul(f2, readout_weights) + readout_biases
            return logits


        logits = model(tf_train_dataset)

        def loss(logits, labels):
            err = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
            tf.add_to_collection('losses', err)
            return tf.add_n(tf.get_collection('losses'), name='total_loss')

        loss = loss(logits, tf_train_labels)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            for epoch in range(num_epochs):
                shuffle = np.random.permutation(train_dataset.shape[0])
                train_dataset = train_dataset[shuffle]
                train_labels = train_labels[shuffle]
                for step in range(num_steps):
                    offset = batch_size * step % (train_labels.shape[0] - batch_size)
                    batch_data = train_dataset[offset:(offset + batch_size), :]
                    batch_labels = train_labels[offset:(offset + batch_size), :]
                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                    _, l, preds = sess.run(fetches=[optimizer, loss, train_prediction], feed_dict=feed_dict)
                    if step % report_interval:
                        print('Minibatch loss at step %d: %.4f' % (step, l))
                        tm = metric(preds, batch_labels)
                        vm = metric(valid_prediction.eval(), valid_labels)
                        print('Minibatch metric: %.4f' % tm)
                        print('Validation metric: %.4f\n' % vm)
                        cost_history.append(tm)
                        loss_history.append(l)
                        cost_cv_history.append(vm)

                epoch_test_metric = metric(test_prediction.eval(), test_labels)
                print('Test metric: %.4f' % epoch_test_metric)
                cost_epoch_history.append(epoch_test_metric)
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3)
            ax1.plot(range(len(loss_history)), loss_history)
            ax1.set_xlim([0, len(loss_history)])
            ax1.set_ylim([0, np.max(loss_history)])
            ax2.plot(range(len(cost_history)), cost_history)
            ax2.plot(range(len(cost_cv_history)), cost_cv_history)
            ax2.set_xlim([0, len(cost_history)])
            ax2.set_ylim([0, np.max(np.max(cost_history), np.max(cost_cv_history))])
            ax3.scatter(range(len(cost_epoch_history)), cost_epoch_history)
            ax3.set_xlim([0, len(cost_epoch_history)])
            ax3.set_ylim([0, np.max(cost_epoch_history)])
            plt.show()

            preds, = sess.run([train_prediction], feed_dict={tf_train_dataset: submission_dataset})
            # generate submission
            submission = pd.DataFrame()
            submission['user_id'] = np.arange(submission_dataset.shape[0])
            submission['photo_id'] = np.arange(submission_dataset.shape[1])
            submission['click_probability'] = preds[:, 1]
            submission.to_csv(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'v1.0.0-no-topic-submission_nn.txt'),
                              sep='\t', index=False, header=False,
                              float_format='%.6f')

            print('Finished.')
            print('Cost time: {} min'.format((time.time() - start) / 60))

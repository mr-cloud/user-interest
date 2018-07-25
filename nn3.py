import pickle

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import pandas as pd
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

import consts


def metric(prediction, target):
    try:
        return roc_auc_score(target, prediction[:, 1])
    except ValueError:
        return 1.0


def stitch_topic_features(data):
    return data[:, :-1]


start_point = time.time()

print('Loading data and models...')
common_words_counter = pd.read_csv(os.path.join(consts.CLEAN_DATA_PATH, consts.COMMON_WORDS_COUNTER),
                                   sep=' ', header=None, index_col=False, names=['word', 'counter'])
word_indexer = dict(zip(common_words_counter['word'], range(common_words_counter.shape[0])))
common_words_counter['word'] = common_words_counter['word'].astype(str)
common_words_counter['counter'] = common_words_counter['counter'].astype(int)

with open(os.path.join(consts.CLEAN_DATA_PATH, consts.EMBEDDINGS), 'rb') as input:
    embeddings = np.load(input)
with open(os.path.join(consts.CLEAN_DATA_PATH, consts.PHOTO_TOPIC_FEATURES), 'rb') as input:
    photo_topic = pickle.load(input)
with open(os.path.join(consts.CLEAN_DATA_PATH, consts.PHOTO_FACE_FEATURES_NORM), 'rb') as input:
    photo_face = pickle.load(input)
with open(os.path.join(consts.CLEAN_DATA_PATH, consts.USER_VECTOR_SPACE), 'rb') as input:
    user_vector_space = pickle.load(input)

columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
train_interaction = pd.read_table(os.path.join(consts.RAW_DATA_PATH, consts.DATASET_TRAIN_INTERACTION),
                                  header=None, names=columns)
test_columns = ['user_id', 'photo_id', 'time', 'duration_time']
test_interaction = pd.read_table(os.path.join(consts.RAW_DATA_PATH, consts.DATASET_TEST_INTERACTION),
                                 header=None, names=test_columns)

print('Data size: #train={}, #test={}'.format(train_interaction.shape[0], test_interaction.shape[0]))

# sample some data for cross-validation and metric evaluation TODO
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels)
train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(train_dataset, train_labels)
test_dataset, test_labels = resample(test_dataset, test_labels, replace=False, n_samples=int(0.1/3 * len(test_labels)))
valid_dataset, valid_labels = resample(valid_dataset, valid_labels, replace=False, n_samples=int(0.01/3 * len(valid_labels)))
test_dataset = stitch_topic_features(test_dataset)
valid_dataset = stitch_topic_features(valid_dataset)
data_pre_time_cost = '\nData preprocessing time: {} min'.format((time.time() - start_point) / 60)
print(data_pre_time_cost)
preprocessing_photo_face_features.logger.write(data_pre_time_cost)

del dataset
del labels


n_label = 2
n_dim = test_dataset.shape[1]
scalers = np.array([1])
batch_base = 1000
batch_size_grid = np.array(batch_base * scalers, dtype=np.int32)
num_steps_grid = train_dataset.shape[0] // batch_size_grid
num_epoch = 1
report_interval_grid = num_steps_grid // 100
initial_learning_rate_grid = [0.05]
final_learning_rate_grid = [0.025]

# hidden layers
f1_depth = n_dim * 30
f2_depth = n_dim * 90
f3_depth = n_dim * 30
f4_depth = n_dim * 10

# L2 regularization
lambdas = [0.0]


for idx, initial_learning_rate in enumerate(initial_learning_rate_grid):
    for batch_size_idx, batch_size in enumerate(batch_size_grid):
        num_steps = num_steps_grid[batch_size_idx]
        report_interval = report_interval_grid[batch_size_idx]
        for wl in lambdas:
            start_point = time.time()
            cost_history = []
            cost_epoch_history = []
            cost_cv_history = []
            loss_history = []

            graph = tf.Graph()
            with graph.as_default():
                tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=[None, n_dim])
                # sparse style
                tf_train_labels = tf.placeholder(dtype=tf.int32, shape=[None])
                tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
                tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)

                global_step = tf.Variable(0)  # count the number of steps taken.
                final_learning_rate = final_learning_rate_grid[idx]
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
                weights3 = variable_with_weight_loss([f2_depth, f3_depth], wl, 'weights3')
                biases3 = tf.Variable(tf.ones([f3_depth]))
                weights4 = variable_with_weight_loss([f3_depth, f4_depth], wl, 'weights4')
                biases4 = tf.Variable(tf.ones([f4_depth]))
                readout_weights = variable_with_weight_loss([f4_depth, n_label], wl, 'readout_weights')
                readout_biases = tf.Variable(tf.zeros([n_label]))

                def model(data):
                    f1 = tf.nn.relu(tf.nn.xw_plus_b(data, weights1, biases1))
                    f2 = tf.nn.relu(tf.nn.xw_plus_b(f1, weights2, biases2))
                    f3 = tf.nn.relu(tf.nn.xw_plus_b(f2, weights3, biases3))
                    f4 = tf.nn.relu(tf.nn.xw_plus_b(f3, weights4, biases4))
                    logits = tf.matmul(f4, readout_weights) + readout_biases
                    return logits


                logits = model(tf_train_dataset)

                def loss(logits, labels):
                    # doc typo-error for ValueException
                    # ValueError: Rank mismatch: Rank of labels should equal rank of logits minus 1.
                    err = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
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
                    for epoch in range(num_epoch):
                        shuffle = np.random.permutation(train_dataset.shape[0])
                        train_dataset = train_dataset[shuffle]
                        train_labels = train_labels[shuffle]
                        for step in range(num_steps):
                            offset = batch_size * step % (train_labels.shape[0] - batch_size)
                            batch_data = train_dataset[offset:(offset + batch_size)]
                            batch_labels = train_labels[offset:(offset + batch_size)]
                            batch_data = stitch_topic_features(batch_data)
                            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                            _, l, preds = sess.run(fetches=[optimizer, loss, train_prediction], feed_dict=feed_dict)
                            if step % max(1, report_interval) == 0:
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
                    ax1.set_ylim([0, np.max(loss_history) * 1.5])
                    ax1.set_title('loss curve')

                    ax2.plot(range(len(cost_history)), cost_history, 'r')
                    ax2.plot(range(len(cost_cv_history)), cost_cv_history, 'g')
                    ax2.set_xlim([0, len(cost_history)])
                    ax2.set_ylim([0, 1])
                    ax2.set_title('training vs. validation metrics')
                    ax2.legend(('train', 'valid'), loc='lower center')

                    ax3.plot(range(len(cost_epoch_history)), cost_epoch_history, marker='o')
                    ax3.set_xlim([0, len(cost_epoch_history)])
                    ax3.set_ylim([0, 1])
                    ax3.set_title('epoch')

                    plt.subplots_adjust(hspace=0.5)
                    # plt.show()
                    time_consume = '\n{}, Cost time: {} min, regularization: {}, learning rate: {}, final learning rate: {}, batch size: {}\n'.format('nn5', (time.time() - start_point) / 60, wl, initial_learning_rate, final_learning_rate, batch_size)
                    print(time_consume)
                    preprocessing_photo_face_features.logger.write(time_consume)
                    topology = 'f1={}-f2={}-f3={}-f4={}'.format(f1_depth, f2_depth, f3_depth, f4_depth )
                    print(topology + '\n')
                    plt.savefig('datahouse/learning-curve-{}-{}-{}-{}-'.format(wl, initial_learning_rate, final_learning_rate, batch_size) + topology + '.png')
                    preprocessing_photo_face_features.logger.write(topology + '\n')
                    metrics = 'valid metric: {}, test metric: {}\n'.format(vm, epoch_test_metric)
                    print(metrics)
                    preprocessing_photo_face_features.logger.write(metrics)

                    print('Predicting...')
                    # comment this when only one model is trained.
                    # del train_dataset
                    # del train_labels
                    # del valid_dataset
                    # del valid_labels
                    # del test_dataset
                    # del test_labels
                    n_submission = submission_dataset.shape[0]
                    preds_rst = np.ndarray(shape=(n_submission), dtype=np.float32)

                    start = 0  # inclusively
                    end = start  # exclusively
                    while end < n_submission:
                        start = end
                        end = min(start + batch_size * 10, n_submission)
                        batch_data = stitch_topic_features(submission_dataset[start: end])
                        feed_dict = {tf_train_dataset: batch_data}
                        preds, = sess.run(fetches=[train_prediction], feed_dict=feed_dict)
                        preds_rst[start: end] = preds[:, 1]
                    # generate submission
                    submission = pd.DataFrame()
                    submission['user_id'] = test_interaction['user_id']
                    submission['photo_id'] = test_interaction['photo_id']
                    submission['click_probability'] = preds_rst
                    submission.to_csv(
                        os.path.join(preprocessing_photo_face_features.DATA_HOUSE_PATH, 'v1.1.0-without-topic-submission_nn5.txt'),
                        sep='\t', index=False, header=False,
                        float_format='%.6f')
                    print('Finished.')

import pickle
import sys

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
from utils import logger
from preprocessing_photo_face_features import NUM_FACE_FEATURE
from preprocessing_text_feature_embedding import EMBEDDING_SIZE


n_ve_err = 0
n_metric_call = 0


def metric(prediction, target):
    global n_metric_call
    n_metric_call += 1
    global n_ve_err
    try:
        return roc_auc_score(target, prediction)
    except ValueError:
        n_ve_err += 1
        return 1.0


start_point = time.time()

print('Loading interaction data...')
columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
train_interaction = pd.read_table(os.path.join(consts.RAW_DATA_PATH, consts.DATASET_TRAIN_INTERACTION),
                                  header=None, names=columns)
test_columns = ['user_id', 'photo_id', 'time', 'duration_time']
test_interaction = pd.read_table(os.path.join(consts.RAW_DATA_PATH, consts.DATASET_TEST_INTERACTION),
                                 header=None, names=test_columns)
print('Data size: #train={}, #submit={}'.format(train_interaction.shape[0], test_interaction.shape[0]))

# data cleaning and normalization
print('normalizing...')
train_interaction['label'] = np.array(np.any(train_interaction[['click', 'like', 'follow']], axis=1), dtype=np.int32)

scaler = MinMaxScaler()
train_interaction[['time', 'duration_time']] = scaler.fit_transform(train_interaction[['time', 'duration_time']])
test_interaction[['time', 'duration_time']] = scaler.transform(test_interaction[['time', 'duration_time']])
test_columns.extend(['label'])
train_interaction = train_interaction[test_columns]
print('Cleaned data size: ', train_interaction.shape)


print('loading models...')
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

n_feature = 1 + NUM_FACE_FEATURE + EMBEDDING_SIZE


def build_photo_feature(pid, duration):
    feature = np.zeros(shape=n_feature, dtype=np.float32)
    feature[-1] = duration
    if pid in photo_face:
        feature[0: NUM_FACE_FEATURE] = photo_face[pid]
    topic_feature = np.zeros(shape=EMBEDDING_SIZE, dtype=np.float32)
    if len(photo_topic[pid]) != 0:
        for word in photo_topic[pid]:
            topic_feature += embeddings[word_indexer[word]]
            topic_feature = topic_feature / len(photo_topic[pid])
    feature[NUM_FACE_FEATURE: NUM_FACE_FEATURE + EMBEDDING_SIZE] = topic_feature
    return feature


def get_user_vector(uid):
    if uid not in user_vector_space:
        vector = np.zeros(shape=n_feature, dtype=np.float32)
        user_vector_space[uid] = vector
    return user_vector_space[uid]


def stitch_topic_features(interacts: pd.DataFrame, score_or_label=0):
    interacts.index = np.arange(interacts.shape[0])
    dataset = np.zeros(shape=(interacts.shape[0], n_feature * 2 + 1))
    for idx in range(dataset.shape[0]):
        dataset[idx, :n_feature] = get_user_vector(interacts.loc[idx, 'user_id'])
        dataset[idx, n_feature: n_feature*2] = build_photo_feature(interacts.loc[idx, 'photo_id'],
                                                                      interacts.loc[idx, 'duration_time'])
        dataset[idx, -1] = interacts.loc[idx, 'time']
    if score_or_label == 0 or score_or_label == 1:
        return dataset, np.array(interacts['label'])
    else:
        return dataset


# sample some data for cross-validation and metric evaluation
data_pre_time_cost = '\nData preprocessing time: {} min'.format((time.time() - start_point) / 60)
print(data_pre_time_cost)
logger.write(data_pre_time_cost)


def ranking(interacts: pd.DataFrame):
    n_examples = interacts.shape[0]
    dists = np.zeros(shape=n_examples)
    for idx in range(n_examples):
        user_vec = get_user_vector(interacts.loc[idx, 'user_id'])
        photo_vec = build_photo_feature(interacts.loc[idx, 'photo_id'],
                                        interacts.loc[idx, 'duration_time'])
        dists[idx] = np.linalg.norm(user_vec - photo_vec)
    sort_args = np.argsort(dists)
    return (n_examples - sort_args) / n_examples


time_consume = '\n{}, Cost time: {} min, \n'.format('vsm', (time.time() - start_point) / 60)
print(time_consume)
logger.write(time_consume)
train_metric = metric(ranking(train_interaction), train_interaction['label'])
print('train metric: {}'.format(train_metric))
logger.write('train metric: {}'.format(train_metric))


# generate submission
submission = pd.DataFrame()
submission['user_id'] = test_interaction['user_id']
submission['photo_id'] = test_interaction['photo_id']
submission['click_probability'] = ranking(test_interaction)
submission.to_csv(
    os.path.join(consts.CLEAN_DATA_PATH, 'v3.0.0-without-image-submission_vsm.txt'),
    sep='\t', index=False, header=False,
    float_format='%.6f')
print('Finished.')

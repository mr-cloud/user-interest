import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import consts
from preprocessing_photo_face_features import NUM_FACE_FEATURE
from preprocessing_text_feature_embedding import EMBEDDING_SIZE

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

# load interaction files
columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
train_interaction = pd.read_table(os.path.join(consts.RAW_DATA_PATH, consts.DATASET_TRAIN_INTERACTION), header=None,
                                  names=columns)
score = np.zeros(shape=(train_interaction.shape[0]), dtype=np.float32)
for ind in range(train_interaction.shape[0]):
    if ind % 10000 == 0:
        print('Score processing #{}...'.format(ind))
    if train_interaction.loc[ind, 'follow'] == 1:
        score[ind] = 3.0/3
    elif train_interaction.loc[ind, 'like'] == 1:
        score[ind] = 2.0/3
    elif train_interaction.loc[ind, 'click'] == 1:
        # weighted by playing time
        if train_interaction.loc[ind, 'duration_time'] == 0:
            w_play = 0
        else:
            w_play = train_interaction.loc[ind, 'playing_time'] / train_interaction.loc[ind, 'duration_time']
        score[ind] = 1.0/3 * w_play
    else:
        pass
train_interaction['label'] = score
scaler = MinMaxScaler()
train_interaction[['time', 'duration_time']] = scaler.fit_transform(train_interaction[['time', 'duration_time']])

user_vector_space = dict()
n_feature = 1 + NUM_FACE_FEATURE + EMBEDDING_SIZE


def build_photo_feature(pid, duration):
    feature = np.ndarray(shape=n_feature, dtype=np.float32)
    feature[-1] = duration
    if pid not in photo_face:
        feature[0: NUM_FACE_FEATURE] = np.zeros(shape=NUM_FACE_FEATURE, dtype=np.float32)
    else:
        feature[0: NUM_FACE_FEATURE] = photo_face[pid]
    topic_feature = np.zeros(shape=EMBEDDING_SIZE, dtype=np.float32)
    for word in photo_topic[pid]:
        topic_feature += embeddings[word_indexer[word]]
    topic_feature /= len(photo_topic[pid])
    feature[NUM_FACE_FEATURE: NUM_FACE_FEATURE + EMBEDDING_SIZE] = topic_feature
    return feature


def get_user_vector(uid):
    if uid not in user_vector_space:
        vector = np.zeros(shape=n_feature, dtype=np.float32)
        user_vector_space[uid] = vector
    return user_vector_space[uid]


user_act_counts = dict()

for ind in range(train_interaction.shape[0]):
    if ind % 10000 == 0:
        print('User vector extracting #{}...'.format(ind))
    # weighted by (score * time)
    weight = train_interaction.loc[ind, 'time'] * train_interaction.loc[ind, 'label']
    if weight < sys.float_info.min:
        continue
    user_id = train_interaction.loc[ind, 'user_id']
    photo_id = train_interaction.loc[ind, 'photo_id']
    photo_feature = build_photo_feature(pid=photo_id, duration=train_interaction.loc[ind, 'duration_time'])
    user_vector = get_user_vector(uid=user_id)
    user_vector += weight * photo_feature
    if user_id in user_act_counts:
        weight += user_act_counts[user_id]
    user_act_counts[user_id] = weight

for uid, feature in user_vector_space.items():
    user_vector_space[uid] = feature / user_act_counts[uid]

del train_interaction
print('#users=', len(user_vector_space))
with open(os.path.join(consts.CLEAN_DATA_PATH, consts.USER_VECTOR_SPACE), 'wb') as output:
    pickle.dump(user_vector_space, output, pickle.HIGHEST_PROTOCOL)

print('finished')

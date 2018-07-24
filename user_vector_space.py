import os
import pickle
import numpy as np
import pandas as pd

import consts
from preprocessing_photo_face_features import NUM_FACE_FEATURE
from preprocessing_text_feature_embedding import EMBEDDING_SIZE

user_vector = dict()

common_words_counter = pd.read_csv(os.path.join(consts.CLEAN_DATA_PATH, consts.COMMON_WORDS_COUNTER),
                                   sep=' ', header=None, index_col=False, names=['word', 'counter'])
common_words_counter['word'] = common_words_counter['word'].astype(str)
common_words_counter['counter'] = common_words_counter['counter'].astype(int)
word_to_embed = dict(zip(common_words_counter['word'], range(common_words_counter.shape[0])))

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
time_sum = train_interaction.groupby('user_id')['time'].sum().to_dict()
# TODO weight: time, playing_time/duration_time? none for like or follow, click, like, follow
train_interaction['weight'] = [train_interaction.loc[ind, 'time'] / time_sum.get(train_interaction.loc[ind, 'user_id'], 1.0)
                               for ind in range(train_interaction.shape[0])]
duration_min = train_interaction.groupby('photo_id')['']
duration_max
for ind in range(train_interaction.shape[0]):
    photo_feature = np.ndarray(shape=(1 + NUM_FACE_FEATURE + EMBEDDING_SIZE), dtype=np.float32)
    photo_id = train_interaction.loc[ind, 'photo_id']
    photo_feature[0] =
    weight =
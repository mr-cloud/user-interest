import os
import pandas as pd
from sklearn.externals import joblib
import math
import numpy as np
import sys
from sklearn.preprocessing import scale

import preprocessing_photos


def build_pop_examples(interact_filename, photo_model, example_filename, K):
    # load model
    kmeans = joblib.load(photo_model)
    n_clusters, n_features = kmeans.cluster_centers_.shape
    examples = dict()
    photos_id_lable = dict()
    for idx, photo_id in enumerate(kmeans.examples_id_):
        photos_id_lable[int(photo_id)] = kmeans.labels_[idx]
    with open(interact_filename, 'r') as interact_file:
        for line in interact_file:
            line = line.strip()
            splits = line.split()
            if len(splits) >= 8:
                user_id = int(splits[0])
                photo_id = int(splits[1])
                click = int(splits[2])
                like = int(splits[3])
                follow = int(splits[4])
                playing_time = int(splits[6])
                duration_time = int(splits[7])

                if user_id in examples.keys():
                    features = examples[user_id]
                else:
                    features = np.ndarray(shape=[1, n_clusters], dtype=np.float32)
                if photo_id not in photos_id_lable.keys():
                    continue
                cate_id = photos_id_lable[photo_id]
                # TODO



def main():
    # TODO
    train_photo_examples_df = pd.read_csv(
        os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'sample_photo_examples.txt'),
        header=None, dtype=np.float32)
    photo = scale(train_photo_examples_df.as_matrix(columns=train_photo_examples_df.columns[1:]))
    K1s = [10, 30, 100]
    for K1 in K1s:
        inf = K1
        sup = math.min(K1, K1 * (K1 - 1) // 2)
        K2s = np.arange(inf, sup, 3)
        for K2 in K2s:
            if not os.path.exists(
                    os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'sample_pop_examples-{}-{}'.format(K1, K2))):
                build_pop_examples(os.path.join(preprocessing_photos.RAW_DATA_PATH, 'sample_interaction.txt'),
                                   os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'photo-{}.pkl'.format(K1)),
                                   os.path.join(preprocessing_photos.CLEAN_DATA_PATH,
                                                'sample_pop_examples-{}-{}'.format(K1, K2)),
                                   K2)


print('Examples building finished.')

if __name__ == '__main__':
    main()

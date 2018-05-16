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
    photo_id_lable = dict()
    for idx, photo_id in enumerate(kmeans.examples_id_):
        photo_id_lable[int(photo_id)] = kmeans.labels_[idx]
    with open(interact_filename, 'r') as interact_file:
        cnt = 0
        for line in interact_file:
            cnt += 1
            if cnt % 10000 == 0:
                print('Processing {} Line: {}'.format(interact_filename, cnt))
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
                    features = np.zeros(shape=[n_clusters], dtype=np.float32)
                if (photo_id not in photo_id_lable.keys()) or duration_time == 0:
                    continue
                cate_id = photo_id_lable[photo_id]
                # weighted behaviors
                if click == 0 and like == 0 and follow == 0:
                    bonus = -1
                else:
                    bonus = (click * playing_time / duration_time + 2 * like + 3 * follow)
                features[cate_id] += bonus
                examples[user_id] = features
        print('#users={}'.format(len(examples)))
        print('Saving to file...')
        with open(example_filename, 'w') as example_file:
            for user_id, features in examples.items():
                line = str(user_id)
                for val in features:
                    line += (',' + str(val))
                example_file.write(line)
                example_file.write('\n')
                example_file.flush()


def main():
    K1s = [10, 30, 100, 300, 1000]
    for K1 in K1s:
        inf = K1
        sup = max(K1 + 1, min(1000, K1 * (K1 - 1) // 2))
        K2s = [inf]
        while inf * 3 < sup:
            inf *= 3
            K2s.append(inf)
        for K2 in K2s:
            build_pop_examples(os.path.join(preprocessing_photos.RAW_DATA_PATH, 'train_interaction.txt'),
                               os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'photo-{}.pkl'.format(K1)),
                               os.path.join(preprocessing_photos.CLEAN_DATA_PATH,
                                            'pop_examples-{}-{}.txt'.format(K1, K2)),
                               K2)

    print('Examples building finished.')


if __name__ == '__main__':
    main()

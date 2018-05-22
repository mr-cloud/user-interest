import os
from sklearn.externals import joblib
import math
import numpy as np
import sys

import preprocessing_photos


def build_pop_examples(interact_filename, photo_model, example_filename):
    # load model
    kmeans = joblib.load(photo_model)
    n_clusters, n_features = kmeans.cluster_centers_.shape
    examples = dict()
    photo_id_label = dict()
    for idx, photo_id in enumerate(kmeans.examples_id_):
        photo_id_label[int(photo_id)] = kmeans.labels_[idx]
    with open(interact_filename, 'r') as interact_file:
        cnt = 0
        num_filter = 0
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
                show_time = int(splits[5])
                playing_time = int(splits[6])
                duration_time = int(splits[7])

                if user_id in examples.keys():
                    features = examples[user_id]
                else:
                    features = np.zeros(shape=[n_clusters], dtype=np.float32)
                # TODO
                if (photo_id not in photo_id_label.keys()) or duration_time == 0:
                    num_filter += 1
                    continue
                cate_id = photo_id_label[photo_id]
                # weighted behaviors
                # Recently priming with timestamp weight. belongs to [1, 2]
                time_weight = min(max(1.0, math.log10(1.0 * show_time / sys.maxsize) + 9), 2.0)
                if click == 0 and like == 0 and follow == 0:
                    bonus = -1
                else:
                    bonus = (click * playing_time / duration_time + 2 * like + 3 * follow)
                features[cate_id] += time_weight * bonus
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
        return cnt, num_filter, len(examples)

def main():
    K1s = [10, 30, 100, 300, 1000]
    num_process_stats = list()
    for K1 in K1s:
        num_process_stats.append(build_pop_examples(os.path.join(preprocessing_photos.RAW_DATA_PATH, 'train_interaction.txt'),
                           os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'photo-{}.pkl'.format(K1)),
                           os.path.join(preprocessing_photos.CLEAN_DATA_PATH,
                                        'pop_examples-{}.txt'.format(K1))))

    print('Examples building finished.')
    for tup in num_process_stats:
        print('interacts #total: {}, #filtered: {}; #users: {}'.format(tup[0], tup[1], tup[2]))


if __name__ == '__main__':
    main()

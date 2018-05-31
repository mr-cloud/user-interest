import os
from operator import attrgetter

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

import preprocessing_photos
import numpy as np
import pandas as pd

from utils import logger


class Magician():

    def __init__(self, photo_kmeans, K1, pop_examples) -> None:
        users = np.array(pop_examples[:, 0], dtype=np.int32)
        prefers = pop_examples[:, 1:]
        self.photo_kmeans = photo_kmeans
        self.K1 = K1
        self.photo_cate_map = dict()
        for idx, photo_id in enumerate(photo_kmeans.examples_id_):
            self.photo_cate_map[int(photo_id)] = photo_kmeans.labels_[idx]

        # normalization: shift and standardization (e.g., MinMax with normalization)
        inf = np.min(prefers, axis=1)
        sup = np.max(prefers, axis=1)
        matrix = (prefers.transpose() - inf) / (sup - inf)
        self.matrix = (matrix / np.sum(matrix, axis=0)).transpose()

        self.user_matrix_map = dict()
        for idx, user_id in enumerate(users):
            self.user_matrix_map[user_id] = idx

        # Cold starting
        fashion = np.mean(photo_kmeans.cluster_centers_, axis=0)
        fashion = (fashion - np.min(fashion)) / (np.max(fashion) - np.min(fashion))
        self.fashion = fashion / np.sum(fashion)

        # sorting key
        self.total_inertia = photo_kmeans.inertia_

        self.name = 'submission-{}.txt'.format(K1)

    def __str__(self) -> str:
        return 'K1={}, total inertia={}'.format(self.K1, self.total_inertia)


def recommend(sub_prefix):
    print('Loading moldes...')
    photo_model_prefix = 'photo-'
    pop_examples_prefix = 'pop_examples-'
    magicians = list()
    for file in os.listdir(preprocessing_photos.DATA_HOUSE_PATH):
        if file.startswith(photo_model_prefix):
            photo_kmeans = joblib.load(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, file))
            first_sep = file.index('-')
            second_sep = file.rindex('.')
            K1 = int(file[first_sep + 1: second_sep])
            pop_examples = np.loadtxt(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, pop_examples_prefix + str(K1) + '.txt'),
                                  delimiter=',')
            if len(pop_examples.shape) == 1:
                pop_examples = pop_examples.reshape(-1, pop_examples.shape[0])
            magicians.append(Magician(photo_kmeans, K1, pop_examples))
    print('{} models loaded.'.format(len(magicians)))
    # sorting models by multiplication of inertia
    magicians.sort(key=attrgetter('total_inertia'))
    for magician in magicians:
        print(str(magician))
        print('#photo_cate_map={}\n'.format(len(magician.photo_cate_map)))

    # normalization
    print('Normalizing dataset...')
    photo_examples = np.load(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'test_photo_examples.npy'))
    train_photo_examples = np.load(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'train_photo_examples.npy'))
    scaler = MinMaxScaler()
    scaler.fit(train_photo_examples[:, 1:])
    data = scaler.transform(photo_examples[:, 1:])
    # TODO convert into int
    photo_idx_map = dict(zip(photo_examples[:, 0], range(photo_examples.shape[0])))

    # inference
    print('Inferring..')
    magician_predicts_map = dict()
    for rank, magician in enumerate(magicians):
        magician_predicts_map[magician.name] = open(os.path.join(preprocessing_photos.DATA_HOUSE_PATH,
                                                                 sub_prefix + '-' + str(rank) + '_' + magician.name), 'w')

    with open(os.path.join(preprocessing_photos.RAW_DATA_PATH, preprocessing_photos.DATASET_TEST_INTERACTION), 'r') as predict_data:
        tot_cnt = 0
        cnt_unk_photo = 0
        cnt_existed_photo = 0
        cnt_predict_photo = 0
        cnt_new_user = 0
        for line in predict_data:
            line = line.strip()
            segs = line.split()
            user_id = int(segs[0])
            photo_id = int(segs[1])
            for magician in magicians:
                tot_cnt += 1
                if user_id not in magician.user_matrix_map:
                    click_probability = magician.fashion[cate_id]
                    cnt_new_user += 1
                else:
                    if photo_id in magician.photo_cate_map.keys():
                        cate_id = magician.photo_cate_map[photo_id]
                        cnt_existed_photo += 1
                    elif photo_id in photo_idx_map.keys():  # Almost examples should hit here.
                        features = data[photo_idx_map[photo_id]]
                        cate_id = magician.photo_kmeans.predict(np.array([features]))[0]
                        cnt_predict_photo += 1
                    else:  # No example should hit here.
                        cate_id = None
                        cnt_unk_photo += 1

                    if cate_id is None:
                        click_probability = 0.0
                    else:
                        matrix_idx = magician.user_matrix_map[user_id]
                        click_probability = magician.matrix[matrix_idx, cate_id]

                magician_predicts_map[magician.name].write('%d\t%d\t%.6f\n' % (user_id, photo_id, click_probability))
                magician_predicts_map[magician.name].flush()
        # print('#new users={}, #existed={}, #predict={}, #unknown={}, #total={}\n'
        #       .format(cnt_new_user, cnt_existed_photo, cnt_predict_photo, cnt_unk_photo, tot_cnt)
        #       )
        logger.write('#new users={}, #existed={}, #predict={}, #unknown={}, #total={}\n'
              .format(cnt_new_user, cnt_existed_photo, cnt_predict_photo, cnt_unk_photo, tot_cnt)
                     + '\n')




def main(sub_prefix):
    recommend(sub_prefix)
    # print('Finished.')
    logger.write('Finished.' + '\n')


if __name__ == '__main__':
    main('v0.9.0')
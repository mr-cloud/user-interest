import os
from operator import attrgetter

from sklearn.externals import joblib

import preprocessing_photos
import numpy as np
import pandas as pd


class Magician():

    def __init__(self, photo_kmeans, pop_kmeans, K1, K2) -> None:
        self.photo_kmeans = photo_kmeans
        self.pop_kmeans = pop_kmeans
        self.K1 = K1
        self.K2 = K2
        self.user_pop_map = dict()
        for idx, user_id in enumerate(pop_kmeans.examples_id_):
            self.user_pop_map[int(user_id)] = pop_kmeans.labels_[idx]
        self.photo_cate_map = dict()
        for idx, photo_id in enumerate(photo_kmeans.examples_id_):
            self.photo_cate_map[int(photo_id)] = photo_kmeans.labels_[idx]

        # normalization: shift and standardization
        inf = np.min(pop_kmeans.cluster_centers_, axis=1)
        sup = np.max(pop_kmeans.cluster_centers_, axis=1)
        matrix = (pop_kmeans.cluster_centers_.transpose() - inf) / (sup - inf)
        self.matrix = (matrix / np.sum(matrix, axis=0)).transpose()

        # Cold starting
        fashion = np.mean(pop_kmeans.cluster_centers_, axis=0)
        fashion = (fashion - np.min(fashion)) / (np.max(fashion) - np.min(fashion))
        self.fashion = fashion / np.sum(fashion)

        # sorting key
        self.total_inertia = photo_kmeans.inertia_ * pop_kmeans.inertia_

        self.name = 'submission-{}-{}.txt'.format(K1, K2)

    def __str__(self) -> str:
        return 'K1={}, K2={}, final inertia={}'.format(self.K1, self.K2, self.total_inertia)


def feature_normalize(dataset, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(dataset, axis=0)
    if sigma is None:
        sigma = np.std(dataset, axis=0) + 0.1  # in case zero division.
    return (dataset - mu) / sigma, mu, sigma


def recommend():
    pop_model_prefix = 'pop-'
    photo_model_prefix = 'photo-'
    magicians = list()
    for file in os.listdir(preprocessing_photos.DATA_HOUSE_PATH):
        if file.startswith(pop_model_prefix):
            pop_kmeans = joblib.load(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, file))
            first_sep = file.index('-')
            second_sep = file.index('-', first_sep + 1)
            third_sep = file.index('-', second_sep + 1)
            fourth_sep = file.rindex('.')
            K2 = int(file[first_sep + 1, second_sep])
            K1 = int(file[third_sep + 1, fourth_sep])
            photo_kmeans = joblib.load(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, photo_model_prefix + str(K1) + '.pkl'))
            magicians.append(Magician(photo_kmeans, pop_kmeans, K1, K2))
    print('{} models loaded.'.format(len(magicians)))
    # sorting models by multiplication of inertia
    magicians.sort(key=attrgetter('comb_inertia'))
    for magician in magicians:
        print(str(magician))

    # normalization
    photo_examples = np.loadtxt(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'sample_photo_examples.txt'),
                                delimiter=',')
    train_photo_examples_df = pd.read_csv(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'sample_photo_examples.txt'),
                                          header=None, dtype=np.float32)
    _, mu, sigma = feature_normalize(train_photo_examples_df.as_matrix(columns=train_photo_examples_df.columns[1:]))
    norm_dataset,_ , _ = feature_normalize(photo_examples[:, 1:], mu, sigma)

    photo_features_map = dict()
    for i in range(photo_examples.shape[0]):
        photo_id = int(photo_examples[i, 0])
        photo_features_map[photo_id] = norm_dataset[i]

    # inference
    magician_predicts_map = dict()
    for magician in magicians:
        magician_predicts_map[magician.name] = open(os.path.join(preprocessing_photos.DATA_HOUSE_PATH,
                                                                 magician.name), 'w')
    with open(os.path.join(preprocessing_photos.RAW_DATA_PATH, 'test_interaction'), 'r') as predict_data:
        for line in predict_data:
            line = line.strip()
            segs = line.split()
            user_id = int(segs[0])
            photo_id = int(segs[1])
            for magician in magicians:
                pop_id = magician.user_pop_map[user_id] if user_id in magician.user_pop_map.keys() else None
                if photo_id in magician.photo_cate_map.keys():
                    cate_id = magician.photo_cate_map[photo_id]
                else:
                    cate_id = magician.photo_kmeans.predict(np.array([photo_features_map[photo_id]]))[0]
                if user_id is None:
                    click_probability = magician.fashion[cate_id]
                else:
                    click_probability = magician.matrix[pop_id, cate_id]
                magician_predicts_map[magician.name].write('%d\t%d\t%.6f\n' % (user_id, photo_id, click_probability))
                magician_predicts_map[magician.name].flush()





def main():
    recommend()


if __name__ == '__main__':
    main()
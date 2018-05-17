import os
from operator import itemgetter

from sklearn.externals import joblib

import preprocessing_photos


def recommend():
    pop_model_prefix = 'pop-'
    photo_model_prefix = 'photo-'
    models = list()
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
            models.append((pop_kmeans, photo_kmeans, K1, K2))
    print('{} models loaded.'.format(len(models)))
    # sorting models by multiplication of inertia
    models.sort(key=itemgetter(0).inertia_ * itemgetter(1).inertia_)
    for tuple in models:
        print('K1={}, K2={}, final inertia={}'.format(tuple[2], tuple[3], tuple[0].inertia_ * tuple[1].inertia_))
    # XXX

def main():
    recommend()


if __name__ == '__main__':
    main()
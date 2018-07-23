import os

from sklearn.preprocessing import MinMaxScaler

import consts
import pickle
import numpy as np
from preprocessing_photo_face_features import NUM_FACE_FEATURE


with open(os.path.join(consts.CLEAN_DATA_PATH, consts.PHOTO_FACE_FEATURES), 'rb') as input:
    photo_face = pickle.load(input)
    X = np.ndarray(shape=(len(photo_face), NUM_FACE_FEATURE), dtype=np.float32)
    ind = 0
    for photo_id, features in photo_face.items():
        X[ind] = features
        ind += 1
    print('before normalization:', X[:3])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print('after normalization:', X[:3])
    for ind in range(len(photo_face)):
        photo_face[ind] = X[ind]
    print('dumping...')
    with open(os.path.join(consts.CLEAN_DATA_PATH, consts.PHOTO_FACE_FEATURES_NORM), 'wb') as output:
        pickle.dump(photo_face, output, pickle.HIGHEST_PROTOCOL)


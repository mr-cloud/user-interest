'''
Feature engineering
'''

import numpy as np
import os
import json
import pickle

from utils import logger

import consts


NUM_FACE_FEATURE = 5


def build_photo_face_features(face_files: tuple, output_filename):
    # photo_id is int
    photo_face = map()
    cnt = 0
    for face_filename in face_files:
        with open(face_filename, 'r') as face_file:
            for line in face_file:
                cnt += 1
                if cnt % 10000 == 0:
                    print('Processing {}: #{}'.format(face_filename, cnt))
                line = line.strip()
                segs = line.split(maxsplit=1)
                if len(segs) == 2:
                    photo_id = int(segs[0])
                    faces_list = json.loads(segs[1])
                    if type(faces_list) is list:
                        faces = np.array(faces_list, dtype=np.float32)
                        num_face = faces.shape[0]
                        face_occu = np.sum(faces[:, 0])
                        gender_pref = np.mean(faces[:, 1])
                        age = np.mean(faces[:, 2])
                        looking = np.mean(faces[:, 3])
                        photo_face[photo_id] = [num_face, face_occu, gender_pref, age, looking]
    logger.write('#photos with face info = {}'.format(len(photo_face)) + '\n')
    logger.write('dumping...\n')
    with open(output_filename, 'wb') as output:
        pickle.dump(photo_face, output, pickle.HIGHEST_PROTOCOL)


def main():
    build_photo_face_features((consts.DATASET_TRAIN_FACE, consts.DATASET_TEST_FACE), os.path.join(consts.CLEAN_DATA_PATH, 'photo_face_features.pkl'))
    logger.write('Finished.' + '\n')


if __name__ == '__main__':
    main()

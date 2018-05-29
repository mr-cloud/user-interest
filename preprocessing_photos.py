'''
Feature engineering
'''

import numpy as np
import os
import json



DATA_HOUSE_PATH = 'datahouse'
CLEAN_DATA_PATH = DATA_HOUSE_PATH + '/clean-data'
RAW_DATA_PATH = DATA_HOUSE_PATH + '/raw-data'
NUM_FACE_FEATURE = 5
# TODO
NUM_TEXT_FEATURE = 0


def build_photo_examples(face_filename, text_filename, example_filename):
    print()
    photos_id = set()
    face_info = dict()  # {photo_id: string, features: []}
    cnt = 0
    with open(face_filename, 'r') as face_file:
        for line in face_file:
            cnt += 1
            if cnt % 1000 == 0:
                print('Processing {}: {}'.format(face_filename, cnt))
            line = line.strip()
            segs = line.split(maxsplit=1)
            if len(segs) == 2:
                photo_id = segs[0]
                faces_list = json.loads(segs[1])
                if type(faces_list) is list:
                    faces = np.array(faces_list, dtype=np.float32)
                    num_face = faces.shape[0]
                    face_occu = np.sum(faces[:, 0])
                    gender_pref = np.mean(faces[:, 1])
                    age = np.mean(faces[:, 2])
                    looking = np.mean(faces[:, 3])
                    face_info[photo_id] = [num_face, face_occu, gender_pref, age, looking]
                    photos_id.add(photo_id)
    print('#photos with face info = {}'.format(len(face_info)))

    text_info = dict()  # {photo_id: string, features: []}
    if text_filename is not None:
        cnt = 0
        with open(text_filename, 'r') as text_file:
            for line in text_file:
                cnt += 1
                if cnt % 1000 == 0:
                    print('Processing {}: {}'.format(text_filename, cnt))
                line = line.strip()
                segs = line.split()
                if len(segs) == 2:
                    photo_id = segs[0]
                    text_info[photo_id] = json.loads(segs[1])
                    photos_id.add(photo_id)
    print('#photos with text info = {}'.format(len(text_info)))
    print('#photos with info = {}'.format(len(photos_id)))

    with open(example_filename, 'w') as example_file:
        cnt = 0
        for photo_id in photos_id:
            cnt += 1
            if cnt % 1000 == 0:
                print('Generating {}: {}'.format(example_filename, cnt))
            if photo_id in face_info.keys():
                face_features = face_info[photo_id]
            else:
                face_features = [0] * NUM_FACE_FEATURE
            if photo_id in text_info:
                text_features = text_info[photo_id]
            else:
                text_features = [0] * NUM_TEXT_FEATURE
            line = str(photo_id)
            for ele in face_features:
                line += ',' + str(ele)
            for ele in text_features:
                line += ',' + str(ele)
            example_file.write(line)
            example_file.write('\n')
            example_file.flush()


def main():
    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)
    # XXX Add text features
    if not os.path.exists(os.path.join(CLEAN_DATA_PATH, 'train_photo_examples.txt')):
        build_photo_examples(os.path.join(RAW_DATA_PATH, 'train_face.txt'),
                             None,
                             os.path.join(CLEAN_DATA_PATH, 'train_photo_examples.txt'))
    if not os.path.exists(os.path.join(CLEAN_DATA_PATH, 'test_photo_examples.txt')):
        build_photo_examples(os.path.join(RAW_DATA_PATH, 'test_face.txt'),
                             None,
                             os.path.join(CLEAN_DATA_PATH, 'test_photo_examples.txt'))
    print('Examples building finished.')


if __name__ == '__main__':
    main()

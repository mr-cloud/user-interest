'''
Feature engineering
'''

import numpy as np
import os
import json
import pandas as pd

from utils import logger

DATA_HOUSE_PATH = 'datahouse'
CLEAN_DATA_PATH = DATA_HOUSE_PATH + '/clean-data'
RAW_DATA_PATH = DATA_HOUSE_PATH + '/raw-data'
NUM_FACE_FEATURE = 5

COMMON_WORDS_COUNTER = os.path.join(CLEAN_DATA_PATH, 'common-words-counter.txt')
EMBEDDINGS = os.path.join(CLEAN_DATA_PATH, 'embeddings.npy')
PHOTO_TOPIC = os.path.join(CLEAN_DATA_PATH, 'photo_topic.txt')

common_words_counter = pd.read_csv(COMMON_WORDS_COUNTER, header=None, sep=' ')
embeddings = np.load(EMBEDDINGS)
photo_topic = pd.read_csv(PHOTO_TOPIC, header=None, sep=' ', dtype=int)
common_word_idx_map = dict(zip(common_words_counter.iloc[:, 0], range(common_words_counter.shape[0])))
photo_topic_map = dict(zip(photo_topic.iloc[:, 0], photo_topic.iloc[:, 1]))


def store(example_filename, NUM_TEXT_FEATURE, photos_id, face_info, text_info_photos):
    with open(example_filename, 'w') as example_file:
        cnt = 0
        num_unfound_photo = 0
        for photo_id in photos_id:
            cnt += 1
            if cnt % 10000 == 0:
                print('Generating {}: {}'.format(example_filename, cnt))
            if photo_id in face_info.keys():
                face_features = face_info[photo_id]
            else:
                face_features = [0] * NUM_FACE_FEATURE
            if photo_id in text_info_photos:
                topic = photo_topic_map[photo_id]
                if NUM_TEXT_FEATURE == 1:
                    text_features = [topic]
                else:
                    idx = common_word_idx_map[topic] if topic in common_word_idx_map.keys() else 0
                    text_features = embeddings[idx]
            else:
                text_features = [0] * NUM_TEXT_FEATURE
                num_unfound_photo += 1
            line = str(photo_id)
            for ele in face_features:
                line += ',' + str(ele)
            for ele in text_features:
                line += ',' + str(ele)
            example_file.write(line)
            example_file.write('\n')
            example_file.flush()
        logger.write('#Unfound photo: {}'.format(num_unfound_photo) + '\n')


def build_photo_examples(face_filename, text_filename, example_filename_prefix):
    print()
    photos_id = set()  # integers
    face_info = dict()  # {photo_id: integer, features: []}
    cnt = 0
    with open(face_filename, 'r') as face_file:
        for line in face_file:
            cnt += 1
            if cnt % 10000 == 0:
                print('Processing {}: {}'.format(face_filename, cnt))
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
                    face_info[photo_id] = [num_face, face_occu, gender_pref, age, looking]
                    photos_id.add(photo_id)
    # print('#photos with face info = {}'.format(len(face_info)))
    logger.write('#photos with face info = {}'.format(len(face_info)) + '\n')
    text_info_photos = set()  # integers
    if text_filename is not None:
        cnt = 0
        with open(text_filename, 'r') as text_file:
            for line in text_file:
                cnt += 1
                if cnt % 10000 == 0:
                    print('Processing {}: {}'.format(text_filename, cnt))
                line = line.strip()
                segs = line.split(maxsplit=1)
                if len(segs) == 2:
                    photo_id = int(segs[0])
                    text_info_photos.add(photo_id)
                    photos_id.add(photo_id)
    # print('#photos with text info = {}'.format(len(text_info_photos)))
    logger.write('#photos with text info = {}'.format(len(text_info_photos)) + '\n')
    # print('#photos in total = {}'.format(len(photos_id)))
    logger.write('#photos in total = {}'.format(len(photos_id)) + '\n')

    store(example_filename_prefix + '.txt', embeddings.shape[1], photos_id, face_info, text_info_photos, logger)
    store(example_filename_prefix + '-topic.txt', 1, photos_id, face_info, text_info_photos, logger)


def main():
    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)
    build_photo_examples(os.path.join(RAW_DATA_PATH, 'train_face.txt'),
                         os.path.join(RAW_DATA_PATH, 'train_text.txt'),
                         os.path.join(CLEAN_DATA_PATH, 'train_photo_examples'),
                         logger)
    build_photo_examples(os.path.join(RAW_DATA_PATH, 'test_face.txt'),
                         os.path.join(RAW_DATA_PATH, 'test_text.txt'),
                         os.path.join(CLEAN_DATA_PATH, 'test_photo_examples'),
                         logger)
    # print('Finished.')
    logger.write('Finished.' + '\n')


if __name__ == '__main__':
    main()

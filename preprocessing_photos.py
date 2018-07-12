'''
Feature engineering
'''

import numpy as np
import os
import json
import pandas as pd

from utils import logger

DATA_HOUSE_PATH = '/media/leo/working/SoftwarePackage/datahouse/user-interest/datahouse'
CLEAN_DATA_PATH = DATA_HOUSE_PATH + '/clean-data'
RAW_DATA_PATH = DATA_HOUSE_PATH + '/raw-data'
DATASET_TEST_FACE = 'sample_test_face.txt'
DATASET_TEST_INTERACTION = 'sample_test_interaction.txt'
DATASET_TEST_TEXT = 'sample_test_text.txt'
DATASET_TRAIN_FACE = 'sample_train_face.txt'
DATASET_TRAIN_INTERACTION = 'sample_train_interaction.txt'
DATASET_TRAIN_TEXT = 'sample_train_text.txt'
DATASET_TEST_FACE = 'test_face.txt'
DATASET_TEST_INTERACTION = 'test_interaction.txt'
DATASET_TEST_TEXT = 'test_text.txt'
DATASET_TRAIN_FACE = 'train_face.txt'
DATASET_TRAIN_INTERACTION = 'train_interaction.txt'
DATASET_TRAIN_TEXT = 'train_text.txt'

TRAIN_PHOTO_EXAMPLE_WITH_TOPIC = 'train_photo_examples-topic.npy'
TEST_PHOTO_EXAMPLE_WITH_TOPIC = 'test_photo_examples-topic.npy'

NUM_FACE_FEATURE = 5

COMMON_WORDS_COUNTER = os.path.join(CLEAN_DATA_PATH, 'common-words-counter.txt')
EMBEDDINGS = os.path.join(CLEAN_DATA_PATH, 'embeddings.npy')
PHOTO_TOPIC = os.path.join(CLEAN_DATA_PATH, 'photo_topic.txt')

common_words_counter = pd.read_csv(COMMON_WORDS_COUNTER, header=None, sep=' ')
embeddings = np.load(EMBEDDINGS)
photo_topic = pd.read_csv(PHOTO_TOPIC, header=None, sep=' ', dtype=int)
# <key: str, value: int>
common_word_idx_map = dict(zip(common_words_counter.iloc[:, 0], range(common_words_counter.shape[0])))
photo_topic_map = dict(zip(photo_topic.iloc[:, 0], photo_topic.iloc[:, 1]))


def store(example_filename, NUM_TEXT_FEATURE, photos_id, face_info, text_info_photos):
    cnt = 0
    num_unfound_photo = 0
    examples = np.zeros(shape=(len(photos_id), 1 + NUM_FACE_FEATURE + NUM_TEXT_FEATURE), dtype=np.float32)
    examples[:, 0] = list(photos_id)
    for exam_idx, photo_id in enumerate(photos_id):
        if cnt % 10000 == 0:
            print('Generating {}: {}'.format(example_filename, cnt))
        if photo_id in face_info.keys():
            examples[exam_idx, 1: NUM_FACE_FEATURE + 1] = face_info[photo_id]
        if photo_id in text_info_photos:
            topic = photo_topic_map[photo_id]
            if NUM_TEXT_FEATURE == 1:
                examples[exam_idx, NUM_FACE_FEATURE + 1:] = [topic]
            else:
                idx = common_word_idx_map[topic] if topic in common_word_idx_map.keys() else 0
                examples[exam_idx, NUM_FACE_FEATURE + 1:] = embeddings[idx]
        else:
            num_unfound_photo += 1
        cnt += 1
    np.save(example_filename, examples)
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

    store(example_filename_prefix + '-topic.npy', 1, photos_id, face_info, text_info_photos)
    store(example_filename_prefix + '.npy', embeddings.shape[1], photos_id, face_info, text_info_photos)


def main():
    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)
    build_photo_examples(os.path.join(RAW_DATA_PATH, DATASET_TRAIN_FACE),
                         os.path.join(RAW_DATA_PATH, DATASET_TRAIN_TEXT),
                         os.path.join(CLEAN_DATA_PATH, 'train_photo_examples'))
    build_photo_examples(os.path.join(RAW_DATA_PATH, DATASET_TEST_FACE),
                         os.path.join(RAW_DATA_PATH, DATASET_TEST_TEXT),
                         os.path.join(CLEAN_DATA_PATH, 'test_photo_examples'))
    # print('Finished.')
    logger.write('Finished.' + '\n')


if __name__ == '__main__':
    main()

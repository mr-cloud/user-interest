'''
Feature engineering
'''

import numpy as np
import pandas as pd
import os

CLEAN_DATA_PATH = 'datahouse/clean-data'
RAW_DATA_PATH = 'datahouse/raw-data'


def build_photo_examples(face_filename, text_filename, examples_filename):
    # XXX
    pass


def build_pop_examples(interact_filename, examples_filename):
    pass


def main():
    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)
    build_photo_examples(os.path.join(RAW_DATA_PATH, 'train_face.txt'),
                         os.path.join(RAW_DATA_PATH, 'train_text.txt'),
                         os.path.join(CLEAN_DATA_PATH, 'train_photo_examples.txt'))
    build_pop_examples(os.path.join(RAW_DATA_PATH, 'train_interaction.txt'),
                       os.path.join(CLEAN_DATA_PATH, 'train_pop_examples.txt'))
    print('Examples building finished.')



if __name__ == '__main__':
    main()

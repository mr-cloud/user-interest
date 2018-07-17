import os

DATA_HOUSE_PATH = '/media/leo/working/SoftwarePackage/datahouse/user-interest/datahouse'
CLEAN_DATA_PATH = DATA_HOUSE_PATH + '/clean-data'
RAW_DATA_PATH = DATA_HOUSE_PATH + '/raw-data'

if not os.path.exists(CLEAN_DATA_PATH):
    os.makedirs(CLEAN_DATA_PATH)
if not os.path.exists(RAW_DATA_PATH):
    os.makedirs(RAW_DATA_PATH)

is_debuggable = True

DATASET_TEST_FACE = 'test_face.txt'
DATASET_TEST_INTERACTION = 'test_interaction.txt'
DATASET_TEST_TEXT = 'test_text.txt'
DATASET_TRAIN_FACE = 'train_face.txt'
DATASET_TRAIN_INTERACTION = 'train_interaction.txt'
DATASET_TRAIN_TEXT = 'train_text.txt'

if is_debuggable:
    DATASET_TEST_FACE = 'sample_test_face.txt'
    DATASET_TEST_INTERACTION = 'sample_test_interaction.txt'
    DATASET_TEST_TEXT = 'sample_test_text.txt'
    DATASET_TRAIN_FACE = 'sample_train_face.txt'
    DATASET_TRAIN_INTERACTION = 'sample_train_interaction.txt'
    DATASET_TRAIN_TEXT = 'sample_train_text.txt'


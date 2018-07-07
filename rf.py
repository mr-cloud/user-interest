import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import sparse

import preprocessing_photos
import numpy as np
import time


start = time.time()
print('Loading data and models...')
path = preprocessing_photos.RAW_DATA_PATH
columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
train_interaction = pd.read_table(os.path.join(path, preprocessing_photos.DATASET_TRAIN_INTERACTION), header=None)
train_interaction.columns = columns
test_columns = ['user_id', 'photo_id', 'time', 'duration_time']
test_interaction = pd.read_table(os.path.join(path, preprocessing_photos.DATASET_TEST_INTERACTION), header=None)
test_interaction.columns = test_columns

train_photo_topic = np.load(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, preprocessing_photos.TRAIN_PHOTO_EXAMPLE_WITH_TOPIC))
test_photo_topic = np.load(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, preprocessing_photos.TEST_PHOTO_EXAMPLE_WITH_TOPIC))

train_photo_features_idx_map = dict(zip(train_photo_topic[:, 0], range(train_photo_topic.shape[0])))
test_photo_features_idx_map = dict(zip(test_photo_topic[:, 0], range(test_photo_topic.shape[0])))

print('Adding photo features')
train_interaction['num_face'] = train_interaction['photo_id'].apply(
    lambda x: train_photo_topic[train_photo_features_idx_map[x], 1])
train_interaction['face_occu'] = train_interaction['photo_id'].apply(
    lambda x: train_photo_topic[train_photo_features_idx_map[x], 2])
train_interaction['gender_pref'] = train_interaction['photo_id'].apply(
    lambda x: train_photo_topic[train_photo_features_idx_map[x], 3])
train_interaction['age'] = train_interaction['photo_id'].apply(
    lambda x: train_photo_topic[train_photo_features_idx_map[x], 4])
train_interaction['looking'] = train_interaction['photo_id'].apply(
    lambda x: train_photo_topic[train_photo_features_idx_map[x], 5])
# Replace real topic with index in embeddings.
train_interaction['topic'] = train_interaction['photo_id'].apply(
    lambda x: preprocessing_photos.common_word_idx_map.get(str(int(train_photo_topic[train_photo_features_idx_map[x], 6])), 0))

test_interaction['num_face'] = test_interaction['photo_id'].apply(
    lambda x: test_photo_topic[test_photo_features_idx_map[x], 1])
test_interaction['face_occu'] = test_interaction['photo_id'].apply(
    lambda x: test_photo_topic[test_photo_features_idx_map[x], 2])
test_interaction['gender_pref'] = test_interaction['photo_id'].apply(
    lambda x: test_photo_topic[test_photo_features_idx_map[x], 3])
test_interaction['age'] = test_interaction['photo_id'].apply(
    lambda x: test_photo_topic[test_photo_features_idx_map[x], 4])
test_interaction['looking'] = test_interaction['photo_id'].apply(
    lambda x: test_photo_topic[test_photo_features_idx_map[x], 5])
test_interaction['topic'] = test_interaction['photo_id'].apply(
    lambda x: preprocessing_photos.common_word_idx_map.get(str(int(test_photo_topic[test_photo_features_idx_map[x], 6])), 0))


print('Adding user features')
rst = train_interaction.groupby('user_id')['click'].mean().to_dict()
train_interaction['user_click_oof'] = train_interaction['user_id'].apply(
    lambda x: rst.get(x, 0))
test_interaction['user_click_oof'] = test_interaction['user_id'].apply(
    lambda x: rst.get(x, 0)
)
rst = train_interaction.groupby('user_id')['playing_time'].mean().to_dict()
train_interaction['user_play_time_oof'] = train_interaction['user_id'].apply(
    lambda x: rst.get(x, 0))
test_interaction['user_play_time_oof'] = test_interaction['user_id'].apply(
    lambda x: rst.get(x, 0)
)

print('Normalizing...')
## No topic
features = ['user_click_oof', 'user_play_time_oof', 'duration_time', 'time', 'num_face', 'face_occu', 'gender_pref', 'age', 'looking']
scaler = MinMaxScaler()
dataset = scaler.fit_transform(train_interaction[features])
submission_dataset = scaler.transform(test_interaction[features])
labels = np.array(np.any(train_interaction[['click', 'like', 'follow']], axis=1), dtype=int)

## With topic
# num_features = ['user_click_oof', 'user_play_time_oof', 'duration_time', 'time', 'num_face', 'face_occu', 'gender_pref', 'age', 'looking']
# cat_features = ['topic']
# scaler = MinMaxScaler()
# dataset = scaler.fit_transform(train_interaction[num_features])
# submission_dataset = scaler.transform(test_interaction[num_features])
# enc = OneHotEncoder(handle_unknown='ignore')
# X_cat = enc.fit_transform(train_interaction[cat_features])
# X_t_cat = enc.transform(test_interaction[cat_features])
# dataset = sparse.hstack([X_cat, dataset]).tocsr()
# submission_dataset = sparse.hstack([X_t_cat, submission_dataset]).tocsr()
# labels = np.array(np.any(train_interaction[['click', 'like', 'follow']], axis=1), dtype=int)


del train_interaction
dataset, labels = resample(dataset, labels, replace=False, n_samples=int(len(labels) * 0.7))
print('Data size: ', dataset.shape)
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels)

reg = RandomForestClassifier()

# RF
tuned_parameters_rf = [{
    'n_estimators': [10],
    'min_samples_split': [18, 24],
    'min_samples_leaf': [9, 12]
}]


reg = GridSearchCV(reg, tuned_parameters_rf, n_jobs=4, cv=5, verbose=1)

print('training and tuning ...')
reg.fit(train_dataset, train_labels)

print("Best parameters set found on development set:")
print()
print(reg.best_params_)
print()
print("Grid scores on development set:")
print()
report = pd.DataFrame(reg.cv_results_)
print(report[['mean_fit_time', 'mean_test_score', 'mean_train_score']])
print()


def metric(prediction, target):
    try:
        return roc_auc_score(target, prediction[:, 1])
    except ValueError:
        return 1.0


print('Training metric %.6f' % metric(reg.predict_proba(train_dataset), train_labels))
print('Test metric %.6f' % metric(reg.predict_proba(test_dataset), test_labels))

preds = reg.predict_proba(submission_dataset)
# generate submission
submission = pd.DataFrame()
submission['user_id'] = test_interaction['user_id']
submission['photo_id'] = test_interaction['photo_id']
submission['click_probability'] = preds[:, 1]
submission.to_csv(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'v1.0.3-no-topic-submission_rf.txt'), sep='\t', index=False, header=False,
                  float_format='%.6f')

print('Finished.')
print('Cost time: {} min'.format((time.time() - start) / 60))

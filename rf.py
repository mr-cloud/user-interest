import numpy as np
from scipy import sparse as ssp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

import preprocessing_photos

# from config import path
path = preprocessing_photos.RAW_DATA_PATH

out = open(os.path.join('datahouse/rst.txt'), 'w')

columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
train_interaction = pd.read_table(os.path.join(path, preprocessing_photos.DATASET_TRAIN_INTERACTION), header=None)
train_interaction.columns = columns
test_columns = ['user_id', 'photo_id', 'time', 'duration_time']

test_interaction = pd.read_table(os.path.join(path, preprocessing_photos.DATASET_TEST_INTERACTION), header=None)
test_interaction.columns = test_columns

cat_features = ['user_id']
num_features = ['time', 'duration_time']


# calculate the average actions for each feature and add it into example.
def oof_features(train_interaction, test_interaction, agg_col, target_col='click', use_mean=True, use_min=False,
                 use_max=False, use_std=False, use_median=False, n_split=5, seed=1, split_col='user_id'):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed).split(train_interaction[split_col],
                                                                                   train_interaction[target_col])
    train_oof = np.zeros(train_interaction.shape[0])

    test_oof = np.zeros(test_interaction.shape[0])

    for ind_tr, ind_te in skf:
        data_tr = train_interaction.iloc[ind_tr]
        data_te = train_interaction.iloc[ind_te]
        d = data_tr.groupby(agg_col)[target_col].mean().to_dict()
        train_oof[ind_te] = data_te[agg_col].apply(lambda x: d.get(x, 0))

    d = train_interaction.groupby(agg_col)[target_col].mean().to_dict()
    test_oof = test_interaction[agg_col].apply(lambda x: d.get(x, 0))

    return train_oof, test_oof


train_user_click, test_user_click = oof_features(train_interaction, test_interaction, agg_col='user_id',
                                                 target_col='click', split_col='user_id')

train_interaction['user_click_oof'] = train_user_click
test_interaction['user_click_oof'] = test_user_click

num_features += ['user_click_oof']

train_user_playing_time, test_user_playing_time = oof_features(train_interaction, test_interaction, agg_col='user_id',
                                                               target_col='playing_time', split_col='user_id')
train_interaction['user_playing_time_oof'] = train_user_playing_time
test_interaction['user_playing_time_oof'] = test_user_playing_time
num_features += ['user_playing_time_oof']


cat_count_features = []
for c in cat_features:
    d = train_interaction[c].value_counts().to_dict()
    train_interaction['%s_count' % c] = train_interaction[c].apply(lambda x: d.get(x, 0))
    test_interaction['%s_count' % c] = test_interaction[c].apply(lambda x: d.get(x, 0))
    cat_count_features.append('%s_count' % c)

num_features += cat_count_features

train_photo_topic = np.load(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, preprocessing_photos.TRAIN_PHOTO_EXAMPLE_WITH_TOPIC))
test_photo_topic = np.load(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, preprocessing_photos.TEST_PHOTO_EXAMPLE_WITH_TOPIC))

train_photo_features_idx_map = dict(zip(train_photo_topic[:, 0], range(train_photo_topic.shape[0])))
test_photo_features_idx_map = dict(zip(test_photo_topic[:, 0], range(test_photo_topic.shape[0])))

train_interaction['num_face'] =  train_interaction['photo_id'].apply(lambda x: train_photo_topic[train_photo_features_idx_map[x], 1])
train_interaction['face_occu'] =  train_interaction['photo_id'].apply(lambda x: train_photo_topic[train_photo_features_idx_map[x], 2])
train_interaction['gender_pref'] =  train_interaction['photo_id'].apply(lambda x: train_photo_topic[train_photo_features_idx_map[x], 3])
train_interaction['age'] =  train_interaction['photo_id'].apply(lambda x: train_photo_topic[train_photo_features_idx_map[x], 4])
train_interaction['looking'] =  train_interaction['photo_id'].apply(lambda x: train_photo_topic[train_photo_features_idx_map[x], 5])
train_interaction['topic'] =  train_interaction['photo_id'].apply(lambda x: train_photo_topic[train_photo_features_idx_map[x], 6])

test_interaction['num_face'] =  test_interaction['photo_id'].apply(lambda x: test_photo_topic[test_photo_features_idx_map[x], 1])
test_interaction['face_occu'] =  test_interaction['photo_id'].apply(lambda x: test_photo_topic[test_photo_features_idx_map[x], 2])
test_interaction['gender_pref'] =  test_interaction['photo_id'].apply(lambda x: test_photo_topic[test_photo_features_idx_map[x], 3])
test_interaction['age'] =  test_interaction['photo_id'].apply(lambda x: test_photo_topic[test_photo_features_idx_map[x], 4])
test_interaction['looking'] =  test_interaction['photo_id'].apply(lambda x: test_photo_topic[test_photo_features_idx_map[x], 5])
test_interaction['topic'] =  test_interaction['photo_id'].apply(lambda x: test_photo_topic[test_photo_features_idx_map[x], 6])

num_features.extend(['num_face', 'face_occu', 'gender_pref', 'age', 'looking'])
print('num_features', num_features)
out.write('Built features.')
out.flush()

scaler = MinMaxScaler()
enc = OneHotEncoder(handle_unknown='ignore')  # default sparse matrix returned
cat_features = ['topic']
X_cat = enc.fit_transform(train_interaction[cat_features])
X_num = scaler.fit_transform(train_interaction[num_features])
X = ssp.hstack([X_cat, X_num]).tocsr()

X_t_cat = enc.transform(test_interaction[cat_features])
X_t_num = scaler.transform(test_interaction[num_features])
X_t = ssp.hstack([X_t_cat, X_t_num]).tocsr()

y = np.array(np.any(train_interaction[['click', 'like', 'follow']], axis=1), dtype=int)

del X_cat
del X_num
del X_t_cat
del X_t_num
import gc

gc.collect()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(train_interaction['user_id'], y)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_valid = X[ind_te]
    y_train = y[ind_tr]
    y_valid = y[ind_te]
    break
del X
gc.collect()

out.write('Normalization ready.')
out.flush()

clf = LogisticRegression(C=10, random_state=1, verbose=1)
clf.fit(X_train, y_train)

out.write('Trained model.')
out.flush()

y_pred = clf.predict_proba(X_valid)[:, 1]
score = roc_auc_score(y_valid, y_pred)
print('score:%s' % score)

y_sub = clf.predict_proba(X_t)[:, 1]

submission = pd.DataFrame()
submission['user_id'] = test_interaction['user_id']
submission['photo_id'] = test_interaction['photo_id']
submission['click_probability'] = y_sub
submission.to_csv(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'v1.0.0-submission_lr.txt'), sep='\t', index=False, header=False,
                  float_format='%.6f')

out.write('Finished.')
out.flush()
out.close()

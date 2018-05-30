# coding: utf-8

# In[1]:

import numpy as np
import scipy as sp
from scipy import sparse as ssp
from scipy.stats import spearmanr
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# from config import path
path = 'datahouse/raw-data/'

# In[2]:

out = open('datahouse/rst.txt', 'w')

columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
train_interaction = pd.read_table(path + 'train_interaction.txt', header=None)
train_interaction.columns = columns
test_columns = ['user_id', 'photo_id', 'time', 'duration_time']

test_interaction = pd.read_table(path + 'test_interaction.txt', header=None)
test_interaction.columns = test_columns

# In[3]:

cat_features = ['user_id']
num_features = ['time', 'duration_time']


# In[4]:
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

# In[5]:

# corrscore = spearmanr(train_interaction['click'], train_user_click)
# print(corrscore)

train_interaction['user_click_oof'] = train_user_click
test_interaction['user_click_oof'] = test_user_click

num_features += ['user_click_oof']

# In[6]:

# train_user_follow,test_user_follow = oov_features(train_interaction,None,agg_col = 'user_id',target_col='follow',split_col='user_id')
# corrscore = spearmanr(train_interaction['click'],train_user_follow)
# print(corrscore)
# train_interaction['user_follow_oov'] = train_user_follow
# num_features+=['user_follow_oov']


# In[7]:

train_user_playing_time, test_user_playing_time = oof_features(train_interaction, test_interaction, agg_col='user_id',
                                                               target_col='playing_time', split_col='user_id')
# corrscore = spearmanr(train_interaction['click'], train_user_playing_time)
# print(corrscore)
train_interaction['user_playing_time_oof'] = train_user_playing_time
test_interaction['user_playing_time_oof'] = test_user_playing_time
num_features += ['user_playing_time_oof']

# In[8]:

# train_user_duration_time,test_user_duration_time = oov_features(train_interaction,None,agg_col = 'user_id',target_col='duration_time',split_col='user_id')
# corrscore = spearmanr(train_interaction['click'],train_user_duration_time)
# print(corrscore)
# train_interaction['user_duration_time_oov'] = train_user_duration_time
# num_features+=['user_duration_time_oov']


# In[4]:

print('num_features', num_features)

# In[10]:

cat_count_features = []
for c in cat_features:
    d = train_interaction[c].value_counts().to_dict()
    train_interaction['%s_count' % c] = train_interaction[c].apply(lambda x: d.get(x, 0))
    test_interaction['%s_count' % c] = test_interaction[c].apply(lambda x: d.get(x, 0))
    cat_count_features.append('%s_count' % c)

# corrscore = spearmanr(train_interaction['click'], train_interaction['%s_count' % c])
# print(corrscore)
num_features += cat_count_features

out.write('Built features.')
out.flush()

# In[6]:

scaler = MinMaxScaler()
# one column with sparse discrete values to multiple column with 0/1 values.
# enc = OneHotEncoder()
# X_cat = enc.fit_transform(train_interaction[cat_features])
X_num = scaler.fit_transform(train_interaction[num_features])
# X = ssp.hstack([X_cat, X_num]).tocsr()
X = ssp.coo_matrix(X_num).tocsr()


# In[ ]:

# X_t_cat = enc.transform(test_interaction[cat_features])
X_t_num = scaler.transform(test_interaction[num_features])

# In[7]:


# X_t = ssp.hstack([X_t_cat, X_t_num]).tocsr()
X_t = ssp.coo_matrix(X_t_num).tocsr()


# In[9]:

y = train_interaction['click'].values

# In[10]:

# del X_cat
del X_num
import gc

gc.collect()

# In[11]:

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(train_interaction['user_id'], y)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]
    y_train = y[ind_tr]
    y_test = y[ind_te]
    break
del X
gc.collect()

out.write('Normalization ready.')
out.flush()

# In[12]:

clf = LogisticRegression(C=10, random_state=1)
clf.fit(X_train, y_train)

# In[13]:


out.write('Trained model.')
out.flush()

y_pred = clf.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, y_pred)
print('score:%s' % score)

y_sub = clf.predict_proba(X_t)[:, 1]

submission = pd.DataFrame()
submission['user_id'] = test_interaction['user_id']
submission['photo_id'] = test_interaction['photo_id']
submission['click_probability'] = y_sub
submission['click_probability'].apply(lambda x: float('%.6f' % x))
submission.to_csv('datahouse/submission_lr.txt', sep='\t', index=False, header=False)

out.write('Finished.')
out.flush()
out.close()

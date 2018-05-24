import os
from operator import attrgetter

from sklearn.externals import joblib

import preprocessing_photos
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

train_interaction = pd.DataFrame([[1, 0], [2, 1], [3, 0], [1, 1]], columns=['user_id', 'click'])
test_interaction = pd.DataFrame([[1, 0], [2, 1], [3, 0], [1, 1]], columns=['user_id', 'click'])

def oof_features(train_interaction, test_interaction, agg_col, target_col='click', use_mean=True, use_min=False,
                 use_max=False, use_std=False, use_median=False, n_split=2, seed=1, split_col='user_id'):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed).split(train_interaction[split_col],
    train_interaction[target_col])
    print(skf)

    train_oof = np.zeros(train_interaction.shape[0])

    test_oof = np.zeros(test_interaction.shape[0])

    for ind_tr, ind_te in skf:
        data_tr = train_interaction.iloc[ind_tr]
        data_te = train_interaction.iloc[ind_te]
        print(ind_tr, ind_te, data_tr, data_te)
        d = data_tr.groupby(agg_col)[target_col].mean().to_dict()
        print(d)
        train_oof[ind_te] = data_te[agg_col].apply(lambda x: d.get(x, 0))

    d = train_interaction.groupby(agg_col)[target_col].mean().to_dict()
    test_oof = test_interaction[agg_col].apply(lambda x: d.get(x, 0))
    print(d)
    print(test_oof)
    return train_oof, test_oof

train_user_click, test_user_click = oof_features(train_interaction, test_interaction, agg_col='user_id',
                                                 target_col='click', split_col='user_id')
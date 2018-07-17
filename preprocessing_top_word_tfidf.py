from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import numpy as np
import pickle

import consts


def load_data(filenames, prefix):
    photos_id = list()
    corpus = list()
    for filename in filenames:
        with open(os.path.join(prefix, filename)) as input:
            for line in input:
                line = line.strip()
                segs = line.split()
                if len(segs) == 2:
                    photos_id.append(int(segs[0]))
                    corpus.append(segs[1])
    return corpus, photos_id


filenames = [consts.DATASET_TRAIN_TEXT, consts.DATASET_TEST_TEXT]
prefix = consts.RAW_DATA_PATH
# photo_id is int
corpus, photos_id = load_data(filenames, prefix)

vectorizer = CountVectorizer(analyzer=lambda x: x.split(','))
counts = vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(counts)
print('Trained TF-IDF, size: ', tfidf.shape)
# XXX
arg_sort = np.argsort(tfidf, axis=1)
# top 10
tfidf = tfidf[:, arg_sort[int(tfidf.shape[1] * 0.9):]]
print('top20 size: ', tfidf.shape)
n_photos = tfidf.shape[0]
print('#photos={}'.format(n_photos))
features = vectorizer.get_feature_names()
print('#features={}'.format(len(features)))
photo_topic = map()
with open(os.path.join(consts.CLEAN_DATA_PATH, 'photo_topic.pkl'), 'wb') as output:
    for ind in range(tfidf.shape[0]):
        if ind % 10000 == 0:
            print('Writing #{}'.format(ind))
        photo_topic[photos_id[ind]] = features[tfidf[ind]]
    print('dumping...')
    pickle.dump(photo_topic, output, pickle.HIGHEST_PROTOCOL)
print('Finished')

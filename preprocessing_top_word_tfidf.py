from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import numpy as np

import preprocessing_photos


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


filenames = [preprocessing_photos.DATASET_TRAIN_TEXT, preprocessing_photos.DATASET_TEST_TEXT]
# filenames = ['sample_train_text.txt']
prefix = preprocessing_photos.RAW_DATA_PATH
corpus, photos_id = load_data(filenames, prefix)

vectorizer = CountVectorizer(analyzer=lambda x: x.split(','))
counts = vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(counts)
print('Trained TF-IDF, size: ', tfidf.shape)
maxargs = np.argmax(tfidf, axis=1)
print('maxargs size: ', maxargs.shape)
n_photos = maxargs.shape[0]
print('#photos={}'.format(n_photos))
features = vectorizer.get_feature_names()
print('#features={}'.format(len(features)))
with open(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'photo_topic.txt'), 'w') as output:
    for ind in range(maxargs.shape[0]):
        if ind % 10000 == 0:
            print('Writing #{}'.format(ind))
        output.write('{} {}\n'.format(photos_id[ind], features[maxargs[ind, 0]]))
        output.flush()
print('Finished')
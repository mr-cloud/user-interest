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


filenames = ['train_text.txt', 'test_text.txt']
# filenames = ['sample_train_text.txt']
prefix = preprocessing_photos.RAW_DATA_PATH
corpus, photos_id = load_data(filenames, prefix)

vectorizer = CountVectorizer(analyzer=lambda x: x.split(','))
counts = vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(counts)
print(tfidf[:5, :5])
maxargs = np.argmax(tfidf, axis=1)
with open(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'photo_term.txt'), 'w') as output:
    for ind in range(maxargs.shape[0]):
        output.write('{} {}'.format(photos_id[ind], vectorizer.get_feature_names()[maxargs[ind, 0]]))
        output.write('\n')
print('Finished')
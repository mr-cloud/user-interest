from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import numpy as np
import pickle
import pandas as pd

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


def main():
    common_words_counter = pd.read_csv(os.path.join(consts.CLEAN_DATA_PATH, consts.COMMON_WORDS_COUNTER),
                                       sep=' ', header=None, index_col=False, names=['word', 'counter'])
    common_words_counter['word'] = common_words_counter['word'].astype(str)
    common_words_counter['counter'] = common_words_counter['counter'].astype(int)
    print(common_words_counter.dtypes)
    print(common_words_counter['word'][:5])
    common_words = set(common_words_counter['word'])
    print('size of vocabulary: ', len(common_words))
    filenames = [consts.DATASET_TRAIN_TEXT, consts.DATASET_TEST_TEXT]
    prefix = consts.RAW_DATA_PATH
    # photo_id is int
    corpus, photos_id = load_data(filenames, prefix)

    vectorizer = CountVectorizer(analyzer=lambda x: [word for word in x.split(',') if word in common_words])
    counts = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(counts)
    print('Trained TF-IDF, size: ', tfidf.shape)
    features = vectorizer.get_feature_names()
    features = np.array(features)
    print('#features={}'.format(len(features)))

    n_words = 0
    n_photos = len(photos_id)
    photo_topic = dict()
    with open(os.path.join(consts.CLEAN_DATA_PATH, 'photo_topic.pkl'), 'wb') as output:
        for ind, row in enumerate(tfidf):
            if ind % 10000 == 0:
                print('Top-processing #{}'.format(ind))
            row = row.toarray().flatten()
            arg_sort = np.argsort(-row)
            # top 20%
            words = [word for word in corpus[ind].split(',') if word in common_words]
            # at least three word if content supports(主谓宾)
            len_desc = min(max(3, int(len(words) * 0.2)), len(words))
            photo_topic[photos_id[ind]] = features[arg_sort[:len_desc]]
            n_words += len_desc

        print('dumping...')
        pickle.dump(photo_topic, output, pickle.HIGHEST_PROTOCOL)

    print('#photos={}, #words in total: {}, #words in avg: {}'.format(n_photos, n_words, n_words/n_photos))
    print('Finished')


if __name__ == '__main__':
    main()
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
    top_tfidf = dict()
    n_words = 0
    for ind, row in enumerate(tfidf):
        row = row.toarray().flatten()
        arg_sort = np.argsort(-row)
        # top 30%
        words = [word for word in corpus[ind].split(',') if word in common_words]
        len_desc = int(len(words) * 0.3)
        top_tfidf[ind] = arg_sort[:len_desc]
        n_words += len_desc
    tfidf = top_tfidf
    n_photos = len(tfidf)
    print('#photos={}, #words in total: {}, #words in avg: {}'.format(n_photos, n_words, n_words/n_photos))
    features = vectorizer.get_feature_names()
    features = np.array(features)
    print('#features={}'.format(len(features)))
    photo_topic = dict()
    with open(os.path.join(consts.CLEAN_DATA_PATH, 'photo_topic.pkl'), 'wb') as output:
        for ind in range(n_photos):
            if ind % 10000 == 0:
                print('Writing #{}'.format(ind))
            photo_topic[photos_id[ind]] = features[tfidf[ind]]
        print('dumping...')
        pickle.dump(photo_topic, output, pickle.HIGHEST_PROTOCOL)
    print('Finished')


if __name__ == '__main__':
    main()
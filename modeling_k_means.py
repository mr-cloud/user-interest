from time import time
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, MinMaxScaler

import pandas as pd
import os
from sklearn.externals import joblib
import preprocessing_photos
import datetime

from utils import logger

now = datetime.datetime.now()
np.random.seed(now.second)


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    return name, (time() - t0), estimator.inertia_


def viz_data(data, K, pic_name):
    ### Viz in PCA
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=K, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the test dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(pic_name)


def build_model(data, K, name='k-means++', num_epoch=10, num_iter=300, tol=1e-4):
    print('init\t\ttime\tinertia')
    estimator = KMeans(init=name, n_clusters=K, n_init=num_epoch, max_iter=num_iter, tol=tol, verbose=1)
    name, time_cost, loss = bench_k_means(estimator, name, data)
    print('%-9s\t%.2fs\t%i' % (name, time_cost, loss))
    return estimator


def build_model_with_batch(data, K, name='k-means++', batch_size = 100, num_iter=100, init_size=None, n_init=3, max_no_improvement=10):
    print('init\t\ttime\tinertia')
    estimator = MiniBatchKMeans(n_clusters=K, init=name, max_iter=num_iter, batch_size=batch_size, verbose=1,
                                compute_labels=True, random_state=None, tol=0.0, max_no_improvement=max_no_improvement, init_size=init_size,
                                n_init=n_init, reassignment_ratio=0.01)
    name, time_cost, loss = bench_k_means(estimator, name, data)
    print('%-9s\t%.2fs\t%i' % (name, time_cost, loss))
    return estimator


def modeling(pca_pic, pca_file, train_examples, Ks, model_name, batch_style_threshold=sys.maxsize, batch_size=1000, num_iter=1000, init_size=10000, n_init=3, max_no_improvement=30):
    if not os.path.exists(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, pca_pic)):
        test_photo_examples_df = pd.read_csv(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, pca_file),
                                             header=None, dtype=np.float32)
        data = scale(test_photo_examples_df.as_matrix(columns=test_photo_examples_df.columns[1:]))
        K_viz = 10
        print('Started PCA exploring...')
        viz_data(data, K_viz, os.path.join(preprocessing_photos.DATA_HOUSE_PATH, pca_pic))
        print('Finished PCA exploring.')
    else:
        print('Viz Model has been built!')

    print('Building basic features...')
    train_photo_examples_df = pd.read_csv(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, train_examples),
                                          header=None)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(train_photo_examples_df.as_matrix(columns=train_photo_examples_df.columns[1:]))
    print('Data size: ', data.shape)

    for K in Ks:
        print('\n' + '_' * 82)
        print('Modeling K={}...'.format(K))
        if K > batch_style_threshold:
            estimator = build_model_with_batch(data, K, batch_size=batch_size, num_iter=num_iter, init_size=init_size, n_init=n_init, max_no_improvement=max_no_improvement)
        else:
            estimator = build_model(data, K)
        estimator.examples_id_ = np.reshape(train_photo_examples_df.as_matrix(columns=train_photo_examples_df.columns[0:1]), newshape=[-1])
        print('Saving model K={}...'.format(K))
        joblib.dump(estimator, os.path.join(preprocessing_photos.DATA_HOUSE_PATH, model_name.format(K)))
        print('_' * 82 + '\n')


def modeling_photos():
    modeling('pca-reduced.jpg', 'test_photo_examples.txt', 'train_photo_examples.txt', [10, 30, 100, 300, 1000], 'photo-{}.pkl', 1)


def modeling_users():
    K1s = [10, 30, 100, 300, 1000]
    pop_examples_prefix = 'pop_examples-'
    for K1 in K1s:
        train_examples = pop_examples_prefix + str(K1) + '.txt'
        inf = K1
        sup = max(K1 + 1, min(1000, K1 * (K1 - 1) // 2))
        K2s = [inf]
        while inf * 3 < sup:
            inf *= 3
            K2s.append(inf)
        modeling('pca-reduced-users.jpg',
                 'pop_examples-10.txt',
                 train_examples,
                 K2s,
                 'pop-{}-' + 'photo-' + str(K1) + '.pkl'
                 )


def main():
    # modeling photos
    modeling_photos()

    # modeling users
    # modeling_users()

    # print('Finished.')
    logger.write('Finished.' + '\n')

if __name__ == '__main__':
    main()
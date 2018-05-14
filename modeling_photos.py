from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import pandas as pd
import os
from sklearn.externals import joblib
import preprocessing_photos


np.random.seed(42)


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


def build_model(data, K, name='k-means++', num_epoch=10, num_iter=300, tol = 1e-4):
    print('init\t\ttime\tinertia')
    estimator = KMeans(init=name, n_clusters=K, n_init=num_epoch, max_iter=num_iter, tol=tol)
    name, time_cost, loss = bench_k_means(estimator, name, data)
    print('%-9s\t%.2fs\t%i' % (name, time_cost, loss))
    return estimator


def main():
    if not os.path.exists(os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'pca-reduced.jpg')):
        test_photo_examples_df = pd.read_csv(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'test_photo_examples.txt'),
                                             header=None, dtype=np.float32)
        data = scale(test_photo_examples_df.as_matrix(columns=test_photo_examples_df.columns[1:]))
        K = 10
        print('Started PCA exploring...')
        viz_data(data, K, os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'pca-reduced.jpg'))
        print('Finished PCA exploring.')

    train_photo_examples_df = pd.read_csv(os.path.join(preprocessing_photos.CLEAN_DATA_PATH, 'train_photo_examples.txt'),
                                          header=None, dtype=np.float32)
    data = scale(train_photo_examples_df.as_matrix(columns=train_photo_examples_df.columns[1:]))
    print('Data size: ', data.shape)
    for K in [10, 30, 100]:
        print('\n' + '_' * 82)
        print('Modeling K={}...'.format(K))
        estimator = build_model(data, K)
        estimator.examples_id_ = np.reshape(train_photo_examples_df.as_matrix(columns=train_photo_examples_df.columns[0:1]), newshape=[-1])
        print('Saving model K={}...'.format(K))
        joblib.dump(estimator, os.path.join(preprocessing_photos.DATA_HOUSE_PATH, 'photo-{}.pkl'.format(K)))
        print('_' * 82 + '\n')
    print('Finished.')


if __name__ == '__main__':
    main()
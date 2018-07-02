# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Cluster latents representations in AEs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import all_aes
import tensorflow as tf
from lib import data, utils
import numpy as np
import numpy.linalg as la
import sklearn
from sklearn.cluster import KMeans
from munkres import Munkres


FLAGS = flags.FLAGS
flags.DEFINE_string('ae_dir', '', 'Folder containing AE to use for DA.')
flags.DEFINE_integer('use_svd', 1, 'Whether to normalize singular values.')
flags.DEFINE_integer('n_init', 1000, 'Number of inits for k-means.')
flags.DEFINE_integer('n_jobs', 8, 'Number of jobs for k-means.')
flags.DEFINE_integer('n_try', 1, 'Number of experiments.')


def error(cluster, target_cluster, k):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param target_cluster: target cluster
    :return: error
    """
    n = np.shape(target_cluster)[0]
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i][j] = np.sum(np.logical_and(cluster == i, target_cluster == j))
    m = Munkres()
    indexes = m.compute(-M)
    corresp = []
    for i in range(k):
        corresp.append(indexes[i][1])
    pred_corresp = [corresp[int(predicted)] for predicted in cluster]
    acc = np.sum(pred_corresp == target_cluster) / float(len(target_cluster))
    return acc


def cluster(train_latents, train_labels, test_latents, test_labels):
    num_classes = np.shape(train_labels)[-1]
    labels_hot = np.argmax(test_labels, axis=-1)
    train_latents = np.reshape(train_latents,
                               newshape=[train_latents.shape[0], -1])
    test_latents = np.reshape(test_latents,
                              newshape=[test_latents.shape[0], -1])
    kmeans = KMeans(init='random', n_clusters=num_classes,
                    random_state=0, max_iter=1000, n_init=FLAGS.n_init,
                    n_jobs=FLAGS.n_jobs)
    kmeans.fit(train_latents)
    print(kmeans.cluster_centers_)
    print('Train/Test k-means objective = %.4f / %.4f' %
          (-kmeans.score(train_latents), -kmeans.score(test_latents)))
    print('Train/Test accuracy %.4f / %.3f' %
          (error(np.argmax(train_labels, axis=-1), kmeans.predict(train_latents), k=num_classes),
           error(np.argmax(test_labels, axis=-1), kmeans.predict(test_latents), k=num_classes)))
    return error(labels_hot, kmeans.predict(test_latents), k=num_classes)


def get_latents_and_labels(sess, ops, dataset, batches=None):
    batch = FLAGS.batch
    with tf.Graph().as_default():
        data_in = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess_new:
            images = []
            labels = []
            while True:
                try:
                    payload = sess_new.run(data_in)
                    images.append(payload['x'])
                    assert images[-1].shape[0] == 1 or batches is not None
                    labels.append(payload['label'])
                    if len(images) == batches:
                        break
                except tf.errors.OutOfRangeError:
                    break
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    latents = [sess.run(ops.encode,
                        feed_dict={ops.x: images[p:p + batch]})
               for p in range(0, images.shape[0], FLAGS.batch)]
    latents = np.concatenate(latents, axis=0)
    latents = latents.reshape([latents.shape[0], -1])
    return latents, labels


def main(argv):
    del argv  # Unused.
    ae, ds = utils.load_ae(FLAGS.ae_dir, FLAGS.dataset, FLAGS.batch,
                           all_aes.ALL_AES, return_dataset=True)
    with utils.HookReport.disable():
        ae.eval_mode()

    # Convert all test samples to latents and get the labels
    test_latents, test_labels = get_latents_and_labels(ae.eval_sess,
                                                       ae.eval_ops,
                                                       ds.test)
    print('Shape of test_labels = {}'.format(np.shape(test_labels)))
    print('Shape of test_latents = {}'.format(np.shape(test_latents)))
    train_latents, train_labels = get_latents_and_labels(ae.eval_sess,
                                                         ae.eval_ops,
                                                         ds.train_once,
                                                         60000)
    print('Shape of train_labels = {}'.format(np.shape(train_labels)))
    print('Shape of train_latents = {}'.format(np.shape(train_latents)))
    if not FLAGS.use_svd:
        acc = cluster(train_latents, train_labels, test_latents, test_labels)
        print('classification acc = {}'.format(acc))
        return
    if 0:  # use PCA
        print('PCA')
        pca = sklearn.decomposition.PCA(train_latents.shape[1], whiten=True)
        pca.fit(train_latents)
        train_latents = pca.transform(train_latents)
        test_latents = pca.transform(test_latents)
    else:
        print('SVD')
        mean = train_latents.mean(axis=0)
        train_latents -= mean
        test_latents -= mean
        s, vt = la.svd(train_latents, full_matrices=False)[-2:]
        print('SVD Sigma', s)
        train_latents = (train_latents.dot(vt.T) / (s + 1e-5))
        test_latents = (test_latents.dot(vt.T) / (s + 1e-5))
    rank = train_latents.shape[1]
    for x in range(FLAGS.n_try):
        acc = cluster(train_latents[:, :rank].copy(),
                      train_labels[:, :rank].copy(),
                      test_latents[:, :rank].copy(),
                      test_labels[:, :rank].copy())
        print('Rank %3d Inits %4d Accuracy %.2f' % (rank, FLAGS.n_init, 100 * acc))


if __name__ == '__main__':
    app.run(main)

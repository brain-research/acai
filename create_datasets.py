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
"""Script to download all datasets and create .tfrecord files.
"""

import collections
import gzip
import os
import tarfile
import tempfile
import urllib
import zipfile

from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import scipy.io
import tensorflow as tf
from lib.data import DATA_DIR
from tqdm import trange, tqdm

URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'celeba': '0B7EVK8r0v71pZjFTYXZWM3FlRnM',
    'mnist': 'https://storage.googleapis.com/cvdf-datasets/mnist/{}.gz',
}


def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw


def _load_svhn():
    splits = collections.OrderedDict()
    for split in ['train', 'test', 'extra']:
        with tempfile.NamedTemporaryFile() as f:
            urllib.urlretrieve(URLS['svhn'].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset['labels'] -= 1
        splits[split] = dataset
    return splits


def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        urllib.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile(
                'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())
        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_celeba():
    with tempfile.NamedTemporaryFile() as f:
        gdd.download_file_from_google_drive(
            file_id=URLS['celeba'], dest_path=f.name, overwrite=True)
        zip_f = zipfile.ZipFile(f)
        images = []
        for image_file in tqdm(zip_f.namelist(), 'Decompressing', leave=False):
            if os.path.splitext(image_file)[1] == '.jpg':
                with zip_f.open(image_file) as image_f:
                    images.append(image_f.read())
    train_set = {'images': images, 'labels': np.zeros(len(images), int)}
    return dict(train=train_set)


def _load_mnist():
    def _read32(data):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(data.read(4), dtype=dt)[0]

    image_filename = '{}-images-idx3-ubyte'
    label_filename = '{}-labels-idx1-ubyte'
    split_files = collections.OrderedDict(
        [('train', 'train'), ('test', 't10k')])
    splits = collections.OrderedDict()
    for split, split_file in split_files.items():
        with tempfile.NamedTemporaryFile() as f:
            urllib.urlretrieve(
                URLS['mnist'].format(image_filename.format(split_file)),
                f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2051
                n_images = _read32(data)
                row = _read32(data)
                col = _read32(data)
                images = np.frombuffer(
                    data.read(n_images * row * col), dtype=np.uint8)
                images = images.reshape((n_images, row, col, 1))
        with tempfile.NamedTemporaryFile() as f:
            urllib.urlretrieve(
                URLS['mnist'].format(label_filename.format(split_file)),
                f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2049
                n_labels = _read32(data)
                labels = np.frombuffer(data.read(n_labels), dtype=np.uint8)
        splits[split] = {'images': _encode_png(images), 'labels': labels}
    return splits


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(DATA_DIR, filename + '.tfrecord')
    print 'Saving dataset:', filename
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(label=_int64_feature(data['labels'][x]),
                        image=_bytes_feature(data['images'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())


LOADERS = [
    ('mnist', _load_mnist),
    ('cifar10', _load_cifar10),
    ('svhn', _load_svhn),
    ('celeba', _load_celeba)
]

if __name__ == '__main__':
    try:
        os.makedirs(DATA_DIR)
    except OSError:
        pass
    for name, loader in LOADERS:
        print 'Preparing', name
        datas = loader()
        for sub_name, data in datas.items():
            _save_as_tfrecord(data, '%s-%s' % (name, sub_name))

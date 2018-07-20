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

"""Input data for GANs.

This module provides the input images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import glob
import math
import os
import random

import numpy as np
import tensorflow as tf

_DATA_CACHE = None

DATA_DIR = os.environ['AE_DATA']


class DataSet(object):

    def __init__(self, name, train, test, train_once, height, width, colors,
                 nclass):
        self.name = name
        self.train = train
        self.train_once = train_once
        self.test = test
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass


def get_dataset(dataset_name, params):
    train, height, width, colors = _DATASETS[dataset_name + '_train'](
        batch_size=params['batch_size'])
    test = _DATASETS[dataset_name + '_test'](batch_size=1)[0]
    train = train.map(lambda v: dict(x=v['x'],
                                     label=tf.one_hot(v['label'],
                                                      _NCLASS[dataset_name])))
    test = test.map(lambda v: dict(x=v['x'],
                                   label=tf.one_hot(v['label'],
                                                    _NCLASS[dataset_name])))
    if dataset_name + '_train_once' in _DATASETS:
        train_once = _DATASETS[dataset_name + '_train_once'](batch_size=1)[0]
        train_once = train_once.map(lambda v: dict(
            x=v['x'], label=tf.one_hot(v['label'], _NCLASS[dataset_name])))
    else:
        train_once = None
    return DataSet(dataset_name, train, test, train_once, height, width,
                   colors, _NCLASS[dataset_name])


def draw_line(angle, height, width, w=2.):
    m = np.zeros((height, width, 1))
    x0 = height*0.5
    y0 = width*0.5
    x1 = x0 + (x0 - 1) * math.cos(-angle)
    y1 = y0 + (y0 - 1) * math.sin(-angle)
    flip = False
    if abs(y0 - y1) < abs(x0 - x1):
        x0, x1, y0, y1 = y0, y1, x0, x1
        flip = True
    if y1 < y0:
        x0, x1, y0, y1 = x1, x0, y1, y0
    x0, x1 = x0 - w / 2, x1 - w / 2
    dx = x1 - x0
    dy = y1 - y0
    ds = dx / dy if dy != 0 else 0
    yi = int(math.ceil(y0)), int(y1)
    points = []
    for y in range(int(y0), int(math.ceil(y1))):
        if y < yi[0]:
            weight = yi[0] - y0
        elif y > yi[1]:
            weight = y1 - yi[1]
        else:
            weight = 1
        xs = x0 + (y - y0 - .5) * ds
        xe = xs + w
        xi = int(math.ceil(xs)), int(xe)
        if xi[0] != xi[1]:
            points.append((y, slice(xi[0], xi[1]), weight))
        if xi[0] != xs:
            points.append((y, int(xs), weight * (xi[0] - xs)))
        if xi[1] != xe:
            points.append((y, xi[1], weight * (xe - xi[1])))
    if flip:
        points = [(x, y, z) for y, x, z in points]
    for y, x, z in points:
        m[y, x] += 2 * z
    m -= 1
    m = m.clip(-1, 1)
    return m


def input_lines(batch_size, size=(32, 32, 1), limit=None):
    h, w, c = size

    def gen():
        count = 0
        while limit is None or count < limit:
            angle = 2 * random.random() * math.pi
            m = draw_line(angle, h, w)
            label = int(10 * angle / (2 * math.pi - 1e-6))
            count += 1
            yield m, label

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int64),
                                        (size, tuple()))
    ds = ds.map(lambda x, y: dict(x=x, label=y))
    ds = ds.batch(batch_size)
    return ds, size[0], size[1], size[2]


def _parser_all(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    label = features['label']
    return image, label


def input_fn_record(record_parse_fn,
                    filenames,
                    batch_size,
                    size=(32, 32, 3),
                    pad=(0, 0),
                    crop=(0, 0),
                    resize=(32, 32),
                    shuffle=1024,
                    repeat=True,
                    random_flip_x=False,
                    random_shift_x=0,
                    random_shift_y=0,
                    limit=None):
    """Creates a Dataset pipeline for tfrecord files.

    Args:
    record_parse_fn: function, used to parse a record entry.
    filenames: list of filenames of the tfrecords.
    batch_size: int, batch size.
    size: tuple (HWC) containing the expected image shape.
    pad: tuple (HW) containing how much to pad y and x axis on each size.
    crop: tuple (HW) containing how much to crop y and x axis.
    resize: tuple (HW) containing the desired image shape.
    shuffle: int, the size of the shuffle buffer.
    repeat: bool, whether the dataset repeats itself.
    random_flip_x: bool, whether to random flip the x-axis.
    random_shift_x: int, amount of random horizontal shift.
    random_shift_y: int, amount of random vertical shift.
    limit: int, the number of samples to drop (<0) or to take (>0)..

    Returns:
    Dataset iterator and 3 ints (height, width, colors).
    """

    def random_shift(v):
        if random_shift_y:
            v = tf.concat([v[-random_shift_y:], v, v[:random_shift_y]], 0)
        if random_shift_x:
            v = tf.concat([v[:, -random_shift_x:], v, v[:, :random_shift_x]],
                          1)
        return tf.random_crop(v, [resize[0], resize[1], size[2]])

    filenames = sum([glob.glob(x) for x in filenames], [])
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(record_parse_fn, max(4, batch_size // 4))
    if limit is not None:
        if limit > 0:
            dataset = dataset.take(limit)
        elif limit < 0:
            dataset = dataset.skip(-limit)
    if repeat:
        dataset = dataset.repeat()
    delta = [0, 0]
    if sum(crop):
        dataset = dataset.map(
            lambda x, y: (x[crop[0]:-crop[0], crop[1]:-crop[1]], y))
        delta[0] -= 2 * crop[0]
        delta[1] -= 2 * crop[1]
    if sum(pad):
        padding = [[pad[0]] * 2, [pad[1]] * 2, [0] * 2]
        dataset = dataset.map(
            lambda x, y: (tf.pad(x, padding, constant_values=-1.), y))
        delta[0] += 2 * crop[0]
        delta[1] += 2 * crop[1]
    if resize[0] - delta[0] != size[0] or resize[1] - delta[1] != size[1]:
        dataset = dataset.map(
            lambda x, y: (tf.image.resize_bicubic([x], list(resize))[0], y), 4)
    if shuffle:
        dataset = dataset.shuffle(shuffle)
    if random_flip_x:
        dataset = dataset.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y), 4)
    if random_shift_x or random_shift_y:
        dataset = dataset.map(lambda x, y: (random_shift(x), y), 4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y: dict(
            x=tf.reshape(x, [batch_size] + list(resize) + list(size[-1:])),
            label=y))
    dataset = dataset.prefetch(4)  # Prefetch a few batches.
    return dataset, resize[0] or size[0], resize[1] or size[1], size[2]


_NCLASS = {
    'celeba32': 1,
    'cifar10': 10,
    'lines32': 10,
    'mnist32': 10,
    'svhn32': 10,
}

_DATASETS = {
    'celeba32_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'celeba-train.tfrecord')],
            size=(218, 178, 3),
            crop=(36, 16),
            resize=(32, 32)),
    'celeba32_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'celeba-train.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(218, 178, 3),
            crop=(36, 16),
            resize=(32, 32)),
    'cifar10_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'cifar10-train.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(32, 32, 3),
            resize=(32, 32)),
    'cifar10_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'cifar10-train.tfrecord')],
            size=(32, 32, 3),
            resize=(32, 32)),
    'cifar10_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'cifar10-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(32, 32, 3),
            resize=(32, 32)),
    'mnist32_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'mnist-train.tfrecord')],
            repeat=False,
            shuffle=False,
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'mnist32_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'mnist-train.tfrecord')],
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'mnist32_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'mnist-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'svhn32_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'svhn-train.tfrecord'),
             os.path.join(DATA_DIR, 'svhn-extra.tfrecord')],
            repeat=False,
            shuffle=False,
            size=(32, 32, 3)),
    'svhn32_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'svhn-train.tfrecord'),
             os.path.join(DATA_DIR, 'svhn-extra.tfrecord')],
            size=(32, 32, 3)),
    'svhn32_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'svhn-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(32, 32, 3)),
    'lines32_train': functools.partial(input_lines, size=(32, 32, 1)),
    'lines32_test': functools.partial(input_lines, limit=5000,
                                         size=(32, 32, 1)),
}

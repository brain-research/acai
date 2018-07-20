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

"""Custom neural network layers.

Low-level primitives such as custom convolution with custom initialization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


def downscale2d(x, n):
    """Box downscaling.

    Args:
    x: 4D tensor in NHWC format.
    n: integer scale.

    Returns:
    4D tensor down scaled by a factor n.
    """
    if n <= 1:
        return x
    if n % 2 == 0:
        x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        return downscale2d(x, n // 2)
    return tf.nn.avg_pool(x, [1, n, n, 1], [1, n, n, 1], 'VALID')


def upscale2d(x, n):
    """Box upscaling (also called nearest neighbors).

    Args:
    x: 4D tensor in NHWC format.
    n: integer scale (must be a power of 2).

    Returns:
    4D tensor up scaled by a factor n.
    """
    if n == 1:
        return x
    return tf.batch_to_space(tf.tile(x, [n**2, 1, 1, 1]), [[0, 0], [0, 0]], n)


class HeModifiedNormalInitializer(tf.initializers.random_normal):

    def __init__(self, slope):
        self.slope = slope

    def get_config(self):
        return dict(slope=self.slope)

    def __call__(self, shape, dtype=None, partition_info=None):
        del partition_info
        dtype = dtype or tf.float32
        std = tf.rsqrt((1. + self.slope**2) *
                       tf.cast(tf.reduce_prod(shape[:-1]), tf.float32))
        return tf.random_normal(shape, stddev=std, dtype=dtype)


def encoder(x, scales, depth, latent, scope):
    activation = tf.nn.leaky_relu
    conv_op = functools.partial(
        tf.layers.conv2d, padding='same',
        kernel_initializer=HeModifiedNormalInitializer(0.2))
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y = conv_op(x, depth, 1)
        for scale in range(scales):
            y = conv_op(y, depth << scale, 3, activation=activation)
            y = conv_op(y, depth << scale, 3, activation=activation)
            y = downscale2d(y, 2)
        y = conv_op(y, depth << scales, 3, activation=activation)
        y = conv_op(y, latent, 3)
        return y


def decoder(x, scales, depth, colors, scope):
    activation = tf.nn.leaky_relu
    conv_op = functools.partial(
        tf.layers.conv2d, padding='same',
        kernel_initializer=HeModifiedNormalInitializer(0.2))
    y = x
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for scale in range(scales - 1, -1, -1):
            y = conv_op(y, depth << scale, 3, activation=activation)
            y = conv_op(y, depth << scale, 3, activation=activation)
            y = upscale2d(y, 2)
        y = conv_op(y, depth, 3, activation=activation)
        y = conv_op(y, colors, 3)
        return y

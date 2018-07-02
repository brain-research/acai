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
"""Baseline auto-encoder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

import tensorflow as tf
from lib import data, layers, train, utils, classifiers

FLAGS = flags.FLAGS


class AEDropout(train.AE):

    def model(self, latent, depth, scales, dropout):
        x = tf.placeholder(tf.float32,
                           [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        h = tf.placeholder(
            tf.float32,
            [None, self.height >> scales, self.width >> scales, latent], 'h')

        encode = layers.encoder(x, scales, depth, latent, 'ae_encoder')
        encode_train = tf.layers.flatten(encode)
        encode_train = tf.nn.dropout(encode_train, dropout)
        encode_train = tf.reshape(encode_train, tf.shape(encode))
        decode = layers.decoder(h, scales, depth, self.colors, 'ae_decoder')
        ae = layers.decoder(encode, scales, depth, self.colors, 'ae_decoder')
        ae_train = layers.decoder(encode_train, scales, depth, self.colors,
                                  'ae_decoder')
        loss = tf.losses.mean_squared_error(x, ae_train)

        utils.HookReport.log_tensor(loss, 'loss')
        utils.HookReport.log_tensor(tf.sqrt(loss) * 127.5, 'rmse')

        xops = classifiers.single_layer_classifier(
            tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(FLAGS.lr)
            train_op = train_op.minimize(loss + xloss,
                                         tf.train.get_global_step())
        ops = train.AEOps(x, h, l, encode, decode, ae, train_op,
                          classify_latent=xops.output)

        def gen_images():
            return self.make_sample_grid_and_save(ops)

        recon, inter, slerp, samples = tf.py_func(
            gen_images, [], [tf.float32]*4)
        tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        tf.summary.image('samples', tf.expand_dims(samples, 0))

        return ops


def main(argv):
    del argv  # Unused.
    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    model = AEDropout(
        dataset,
        FLAGS.train_dir,
        dropout=FLAGS.dropout,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales)
    model.train()


if __name__ == '__main__':
    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer(
        'latent', 16,
        'Latent space depth, the total latent size is the depth multiplied by '
        'latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('dropout', 0.5, 'Probability to keep value.')
    app.run(main)

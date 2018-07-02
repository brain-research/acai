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
from lib import data, layers, train, utils, classifiers, eval
from lib.discretization import DiscreteBottleneck
from lib.utils import ClassDict

FLAGS = flags.FLAGS
flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
flags.DEFINE_integer(
    'latent', 16,
    'Latent space depth, the total latent size is the depth multiplied by '
    'latent_width ** 2.')
flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
flags.DEFINE_integer('z_log_size', 14, 'Log size of codebook.')
flags.DEFINE_float('beta', 10, 'Beta term for vqvae')
flags.DEFINE_integer('num_latents', 10, 'Number of Discrete latents')

class AEVQVAE(train.AE):
    hparams = ClassDict(decay=0.999,
                        random_top_k=1,
                        filter_size=512,
                        soft_em=False,
                        num_samples=1,
                        epsilon=1e-5)

    def model(self, latent, depth, scales, z_log_size, beta, num_latents):
        tf.set_random_seed(123)
        x = tf.placeholder(tf.float32,
                           [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        h = tf.placeholder(
            tf.float32,
            [None, self.height >> scales, self.width >> scales, latent], 'h')

        def decode_fn(h):
            with tf.variable_scope('vqvae', reuse=tf.AUTO_REUSE):
                h2 = tf.expand_dims(tf.layers.flatten(h), axis=1)
                h2 = tf.layers.dense(h2, self.hparams.hidden_size * num_latents)
                d = bneck.discrete_bottleneck(h2)
                y = layers.decoder(tf.reshape(d['dense'], tf.shape(h)),
                                   scales, depth, self.colors, 'ae_decoder')
                return y, d

        self.hparams.hidden_size = (
                    (self.height >> scales) * (self.width >> scales) * latent)
        self.hparams.z_size = z_log_size
        self.hparams.num_residuals = 1
        self.hparams.num_blocks = 1
        self.hparams.beta = beta
        self.hparams.ema = True
        bneck = DiscreteBottleneck(self.hparams)
        encode = layers.encoder(x, scales, depth, latent, 'ae_encoder')
        decode = decode_fn(h)[0]
        ae, d = decode_fn(encode)
        loss_ae = tf.losses.mean_squared_error(x, ae)

        utils.HookReport.log_tensor(tf.sqrt(loss_ae) * 127.5, 'rmse')
        utils.HookReport.log_tensor(loss_ae, 'loss_ae')
        utils.HookReport.log_tensor(d['loss'], 'vqvae_loss')

        xops = classifiers.single_layer_classifier(
            tf.stop_gradient(d['dense']), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops + [d['discrete']]):
            train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_ae + xloss + d['loss'], tf.train.get_global_step())
        ops = train.AEOps(x, h, l, encode, decode, ae, train_op,
                          classify_latent=xops.output)

        n_interpolations = 16
        n_images_per_interpolation = 16

        def gen_images():
            return self.make_sample_grid_and_save(
                ops, interpolation=n_interpolations,
                height=n_images_per_interpolation)

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
    model = AEVQVAE(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales,
        z_log_size=FLAGS.z_log_size,
        beta=FLAGS.beta,
        num_latents=FLAGS.num_latents)
    model.train()


if __name__ == '__main__':
    app.run(main)

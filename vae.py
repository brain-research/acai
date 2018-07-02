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
"""Variational autoencoder.
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


class VAE(train.AE):

    def model(self, latent, depth, scales, beta):
        x = tf.placeholder(tf.float32,
                           [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        h = tf.placeholder(
            tf.float32,
            [None, self.height >> scales, self.width >> scales, latent], 'h')

        def encoder(x):
            return layers.encoder(x, scales, depth, latent, 'ae_enc')

        def decoder(h):
            return layers.decoder(h, scales, depth, self.colors, 'ae_dec')

        encode = encoder(x)
        with tf.variable_scope('ae_latent'):
            encode_shape = tf.shape(encode)
            encode_flat = tf.layers.flatten(encode)
            latent_dim = encode_flat.get_shape()[-1]
            q_mu = tf.layers.dense(encode_flat, latent_dim)
            log_q_sigma_sq = tf.layers.dense(encode_flat, latent_dim)
        q_sigma = tf.sqrt(tf.exp(log_q_sigma_sq))
        q_z = tf.distributions.Normal(loc=q_mu, scale=q_sigma)
        q_z_sample = q_z.sample()
        q_z_sample_reshaped = tf.reshape(q_z_sample, encode_shape)
        p_x_given_z_logits = decoder(q_z_sample_reshaped)
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
        ae = 2*tf.nn.sigmoid(p_x_given_z_logits) - 1
        decode = 2*tf.nn.sigmoid(decoder(h)) - 1
        loss_kl = 0.5*tf.reduce_sum(
            -log_q_sigma_sq - 1 + tf.exp(log_q_sigma_sq) + q_mu**2)
        loss_kl = loss_kl/tf.to_float(tf.shape(x)[0])
        x_bernoulli = 0.5*(x + 1)
        loss_ll = tf.reduce_sum(p_x_given_z.log_prob(x_bernoulli))
        loss_ll = loss_ll/tf.to_float(tf.shape(x)[0])
        elbo = loss_ll - beta*loss_kl

        utils.HookReport.log_tensor(loss_kl, 'loss_kl')
        utils.HookReport.log_tensor(loss_ll, 'loss_ll')
        utils.HookReport.log_tensor(elbo, 'elbo')

        xops = classifiers.single_layer_classifier(
            tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ae_vars = tf.global_variables('ae_')
        xl_vars = tf.global_variables('single_layer_classifier')
        with tf.control_dependencies(update_ops):
            train_ae = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                -elbo, var_list=ae_vars)
            train_xl = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                xloss, tf.train.get_global_step(), var_list=xl_vars)
        ops = train.AEOps(x, h, l, q_z_sample_reshaped, decode, ae,
                          tf.group(train_ae, train_xl),
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
    model = VAE(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales,
        beta=FLAGS.beta)
    model.train()


if __name__ == '__main__':
    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer(
        'latent', 16,
        'Latent space depth, the total latent size is the depth multiplied by '
        'latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('beta', 1.0, 'ELBO KL term scale.')
    app.run(main)

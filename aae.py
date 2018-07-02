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
"""Adversarial autoencoder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

import tensorflow as tf
from lib import data, layers, train, utils, classifiers, eval

FLAGS = flags.FLAGS


class AAE(train.AE):

    def model(self, latent, depth, scales, adversary_lr, disc_layer_sizes):
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

        def discriminator(h):
            with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
                h = tf.layers.flatten(h)
                for size in [int(s) for s in disc_layer_sizes.split(',')]:
                    h = tf.layers.dense(h, size, tf.nn.leaky_relu)
                return tf.layers.dense(h, 1)

        encode = encoder(x)
        decode = decoder(h)
        ae = decoder(encode)
        loss_ae = tf.losses.mean_squared_error(x, ae)

        prior_samples = tf.random_normal(tf.shape(encode), dtype=encode.dtype)
        adversary_logit_latent = discriminator(encode)
        adversary_logit_prior = discriminator(prior_samples)
        adversary_loss_latents = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=adversary_logit_latent,
                labels=tf.zeros_like(adversary_logit_latent)))
        adversary_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=adversary_logit_prior,
                labels=tf.ones_like(adversary_logit_prior)))
        autoencoder_loss_latents = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=adversary_logit_latent,
                labels=tf.ones_like(adversary_logit_latent)))

        def _accuracy(logits, label):
            labels = tf.logical_and(label, tf.ones_like(logits, dtype=bool))
            correct = tf.equal(tf.greater(logits, 0), labels)
            return tf.reduce_mean(tf.to_float(correct))
        latent_accuracy = _accuracy(adversary_logit_latent, False)
        prior_accuracy = _accuracy(adversary_logit_prior, True)
        adversary_accuracy = (latent_accuracy + prior_accuracy)/2

        utils.HookReport.log_tensor(loss_ae, 'loss_ae')
        utils.HookReport.log_tensor(adversary_loss_latents, 'loss_adv_latent')
        utils.HookReport.log_tensor(adversary_loss_prior, 'loss_adv_prior')
        utils.HookReport.log_tensor(autoencoder_loss_latents, 'loss_ae_latent')
        utils.HookReport.log_tensor(adversary_accuracy, 'adversary_accuracy')

        xops = classifiers.single_layer_classifier(
            tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ae_vars = tf.global_variables('ae_')
        disc_vars = tf.global_variables('disc')
        xl_vars = tf.global_variables('single_layer_classifier')
        with tf.control_dependencies(update_ops):
            train_ae = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_ae + autoencoder_loss_latents, var_list=ae_vars)
            train_disc = tf.train.AdamOptimizer(adversary_lr).minimize(
                adversary_loss_prior + adversary_loss_latents,
                var_list=disc_vars)
            train_xl = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                xloss, tf.train.get_global_step(), var_list=xl_vars)
        ops = train.AEOps(x, h, l, encode, decode, ae,
                          tf.group(train_ae, train_disc, train_xl),
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

        if FLAGS.dataset == 'lines32':
            batched = (n_interpolations, 32, n_images_per_interpolation, 32, 1)
            batched_interp = tf.transpose(
                tf.reshape(inter, batched), [0, 2, 1, 3, 4])
            mean_distance, mean_smoothness = tf.py_func(
                eval.line_eval, [batched_interp], [tf.float32, tf.float32])
            tf.summary.scalar('mean_distance', mean_distance)
            tf.summary.scalar('mean_smoothness', mean_smoothness)

        return ops


def main(argv):
    del argv  # Unused.
    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    model = AAE(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales,
        adversary_lr=FLAGS.adversary_lr,
        disc_layer_sizes=FLAGS.disc_layer_sizes)
    model.train()


if __name__ == '__main__':
    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer(
        'latent', 16,
        'Latent space depth, the total latent size is the depth multiplied by '
        'latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('adversary_lr', 1e-4,
                       'Learning rate for discriminator.')
    flags.DEFINE_string('disc_layer_sizes', '100,100',
                        'Comma-separated list of discriminator layer sizes.')
    app.run(main)

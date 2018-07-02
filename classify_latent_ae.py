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
"""Classifier on the latents representations in AEs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import all_aes
import tensorflow as tf
from lib import data, utils, train, classifiers

FLAGS = flags.FLAGS
flags.DEFINE_string('ae_dir', '', 'Folder containing AE to use for DA.')


class XLatentAE(train.Classify):

    def process(self, x, label):
        h = self.ae.eval_sess.run(self.ae.eval_ops.encode,
                                  feed_dict={self.ae.eval_ops.x: x})
        return h, label

    def train_step(self, data, ops):
        x = self.tf_sess.run(data)
        x, label = x['x'], x['label']
        x, label = self.process(x, label)
        self.sess.run(ops.train_op, feed_dict={ops.x: x, ops.label: label})

    def model(self):
        x = tf.placeholder(tf.float32,
                           [None,
                            self.height >> self.ae.params['scales'],
                            self.width >> self.ae.params['scales'],
                            self.ae.params['latent']], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label_onehot')

        ops = classifiers.single_layer_classifier(x, l, self.nclass)
        ops.x = x
        ops.label = l
        loss = tf.reduce_mean(ops.loss)
        halfway = ((FLAGS.total_kimg << 10) // FLAGS.batch) // 2
        lr = tf.train.exponential_decay(FLAGS.lr, tf.train.get_global_step(),
                                        decay_steps=halfway,
                                        decay_rate=0.1)

        utils.HookReport.log_tensor(loss, 'xe')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(lr)
            ops.train_op = opt.minimize(loss, tf.train.get_global_step())

        return ops


def main(argv):
    del argv  # Unused.
    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    ae, ae_model = utils.load_ae(FLAGS.ae_dir, FLAGS.dataset, FLAGS.batch,
                                 all_aes.ALL_AES)
    with utils.HookReport.disable():
        ae.eval_mode()
    model = XLatentAE(
        dataset,
        FLAGS.train_dir)
    model.train_dir = os.path.join(model.train_dir, ae_model)
    model.ae = ae
    model.train()


if __name__ == '__main__':
    app.run(main)

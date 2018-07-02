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
"""Single layer fully connected classifier.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

import tensorflow as tf
from lib import data, train, utils, classifiers

FLAGS = flags.FLAGS
flags.DEFINE_float('smoothing', -0.1, 'Label smoothing amount.')


class XFullyConnected(train.Classify):

    def model(self, smoothing):
        x = tf.placeholder(tf.float32,
                           [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label_onehot')

        ops = classifiers.single_layer_classifier(x, l, self.nclass,
                                                  smoothing=smoothing)
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
    model = XFullyConnected(
        dataset,
        FLAGS.train_dir,
        smoothing=FLAGS.smoothing)
    model.train()


if __name__ == '__main__':
    app.run(main)

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

"""Classifiers."""

import tensorflow as tf


class ClassifierOps(object):
    def __init__(self, loss, output, extra_loss=None):
        self.loss = loss
        self.output = output
        self.extra_loss = extra_loss
        # Inputs, train op (used for data augmentation).
        self.x = None
        self.label = None
        self.train_op = None


def single_layer_classifier(h, l, nclass, scope='single_layer_classifier',
                            smoothing=None):
    with tf.variable_scope(scope):
        h0 = tf.layers.flatten(h)
        logits = tf.layers.dense(h0, nclass)
        output = tf.argmax(logits, 1)
        if smoothing:
            l -= abs(smoothing) * (l - 1. / nclass)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                          labels=l)
    return ClassifierOps(loss=loss, output=output)

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

#!/usr/bin/env bash
# Celeba runs
vqvae.py --dataset=celeba32 --latent=16 --latent_width=4 --depth=64 --train_dir=TRAIN/celeba32/
vqvae.py --dataset=celeba32 --latent=2 --latent_width=4 --depth=64 --train_dir=TRAIN/celeba32

# CIFAR10 runs
vqvae.py --dataset=cifar10 --latent=16 --latent_width=4 --depth=64 --train_dir=TRAIN/cifar10/

vqvae.py --dataset=cifar10 --latent=64 --latent_width=4 --depth=64 --train_dir=TRAIN/cifar10/


# SVHN runs
vqvae.py --dataset=svhn32 --latent=16 --latent_width=4 --depth=64 --train_dir=TRAIN/svhn32/ 

vqvae.py --dataset=svhn32 --latent=2 --latent_width=4 --depth=64 --train_dir=TRAIN/svhn32/ 

# MNIST runs
vqvae.py --dataset=mnist32 --latent=16 --latent_width=4 --depth=16 --train_dir=TRAIN/mnist32/ 

vqvae.py --dataset=mnist32 --latent=2 --latent_width=4 --depth=16 --train_dir=TRAIN/mnist32/

# Watchmin32 runs
vqvae.py --dataset=lines32 --latent=16 --latent_width=4 --depth=16 --train_dir=TRAIN/lines32/


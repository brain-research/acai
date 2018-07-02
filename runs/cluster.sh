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

export N_CPUS=8

cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/AEBaseline_depth16_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/AEBaseline_depth16_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/AEBaseline_depth16_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/AEBaseline_depth16_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/AEBaseline_depth16_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/AEBaseline_depth16_latent64_scales3

cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/AAE_adversary_lr0.0001_depth16_disc_layer_sizes100,100_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/AAE_adversary_lr0.0001_depth16_disc_layer_sizes100,100_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/AAE_adversary_lr0.0001_depth16_disc_layer_sizes100,100_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/AAE_adversary_lr0.0001_depth16_disc_layer_sizes100,100_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/AAE_adversary_lr0.0001_depth16_disc_layer_sizes100,100_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/AAE_adversary_lr0.0001_depth16_disc_layer_sizes100,100_latent64_scales3

cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/ACAI_advdepth16_advweight0.5_depth16_latent2_reg0.2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/ACAI_advdepth16_advweight0.5_depth16_latent16_reg0.2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/ACAI_advdepth16_advweight0.5_depth16_latent2_reg0.2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/ACAI_advdepth16_advweight0.5_depth16_latent16_reg0.2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/ACAI_advdepth16_advweight0.5_depth16_latent16_reg0.2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/ACAI_advdepth16_advweight0.5_depth16_latent64_reg0.2_scales3

cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/AEDenoising_depth16_noise1.0_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/AEDenoising_depth16_noise1.0_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/AEDenoising_depth16_noise1.0_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/AEDenoising_depth16_noise1.0_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/AEDenoising_depth16_noise1.0_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/AEDenoising_depth16_noise1.0_latent64_scales3

cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/AEDropout_depth16_dropout0.5_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/AEDropout_depth16_dropout0.5_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/AEDropout_depth16_dropout0.5_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/AEDropout_depth16_dropout0.5_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/AEDropout_depth16_dropout0.5_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/AEDropout_depth16_dropout0.5_latent64_scales3

cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/VAE_beta1.0_depth16_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/mnist32/VAE_beta1.0_depth16_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/VAE_beta1.0_depth16_latent2_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/svhn32/VAE_beta1.0_depth16_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/VAE_beta1.0_depth16_latent16_scales3
cluster.py --n_jobs=$N_CPUS --ae_dir TRAIN/cifar10/VAE_beta1.0_depth16_latent64_scales3

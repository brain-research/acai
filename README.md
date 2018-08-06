# Adversarially Constrained Autoencoder Interpolations (ACAI)

Code for the paper "[Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](http://arxiv.org/abs/1807.07543)" by David Berthelot, Colin Raffel, Aurko Roy, and Ian Goodfellow.

This is not an officially supported Google product.

## Setup

### Config with virtualenv

```bash
sudo apt install virtualenv

cd <path_to_code>
virtualenv --system-site-packages env2
. env2/bin/activate
pip install -r requirements.txt
```

### Config environment variables

Choose a folder where to save the datasets, for example ~/Data
```bash
export AE_DATA=~/Data
```

### Installing datasets

```bash
python create_datasets.py
```

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python acai.py \
--train_dir=TEMP \
--latent=16 --latent_width=2 --depth=16 --dataset=celeba32
```

All training from the paper can be found in folder `runs`.

## Models

These are the maintained models:
- aae.py
- acai.py
- baseline.py
- denoising.py
- dropout.py
- vae.py
- vqvae.py

## Classifiers / clustering

- classifier_fc.py: fully connected single layer from raw pixels, see
 `runs/classify.sh` for examples.
- Auto-encoder classification is trained at the same as the auto-encoder.
- cluster.py: K-means clustering, see `runs/cluster.sh` for examples.

## Utilities

- create_datasets.py: see Installing datsets for more info.

## Unofficial implementations

- Kyle McDonald created a Pytorch version of ACAI [here](https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0).

![Travis CI](https://travis-ci.com/krasserm/super-resolution.svg?branch=master)

# Single Image Super-Resolution with EDSR, WDSR and SRGAN

A [Tensorflow 2.0](https://www.tensorflow.org/beta) based implementation of

- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (EDSR), winner
  of the [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) super-resolution challenge.
- [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718) (WDSR), winner
  of the [NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/) super-resolution challenge (realistic tracks).
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (SRGAN).



A `DIV2K` [data provider](#div2k-dataset) automatically downloads [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
training and validation images of given scale (2, 3, 4 or 8) and downgrade operator ("bicubic", "unknown", "mild" or
"difficult").

## Environment setup

Create a new [conda](https://conda.io) environment with

    conda env create -f environment.yml

and activate it with

    conda activate sisr

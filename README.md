# RetinaNet

A DeepLearning Architecture for image detection.

## Description

This repository contains the code necessary to run a [RetinaNet](https://arxiv.org/abs/1708.02002) based on the [DeepForest](https://github.com/weecology/DeepForest) implementation. 
The implementation uses the PyTorch DeepLearning framework. RetinaNet is used to detect objects within an image.
The repository contains all code necessary to preprocess large tif-images, run training and validation, and perform predictions using the trained models.

## Getting Started

### Dependencies

* DeepForest, GDAL, Pytorch-Lightning, Scipy ... (see Installation)
* Cuda-capable GPU ([overview here](https://developer.nvidia.com/cuda-gpus))
* Anaconda ([download here](https://www.anaconda.com/products/distribution))
* developed on Windows 10

### Installation

* clone the Stable RetinaNet repository
* cd ../RetinaNet/environment
* conda env create -f retinanet.yml

### Executing program

* set parameters in params.py
* run main_retinanet.py

## Help/Known Issues

* None yet

## Authors

* Benjamin Stöckigt
* Malik-Manel Hashim

## Version History

* 0.1
    * Initial Release

## License

Not licensed

## Acknowledgments

* [DeepForest](https://github.com/weecology/DeepForest)
* [DeepForest Documentation](https://deepforest.readthedocs.io/en/latest/)
* [RetinaNet paper](https://arxiv.org/abs/1708.02002)

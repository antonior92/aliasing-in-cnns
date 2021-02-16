# How Convolutional Neural Networks Deal with Aliasing

Python scripts for reproducing the results from the paper: ["How Convolutional Neural Networks Deal with Aliasing"](https://arxiv.org/abs/2102.07757).

```
Antônio H. Ribeiro and Thomas B. Schön "How Convolutional Neural Networks Deal with Aliasing". IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021.
```
```
@inproceedings{ribeiro_how_2021,
author={Ant\^onio H. Ribeiro and Thomas B. Sch\"on},
title={How Convolutional Neural Networks Deal with
Aliasing},
year={2021},
publisher={IEEE},
booktitle={2021 {IEEE} International Conference on Acoustics, Speech and Signal Processing, {ICASSP}}
```

Preprint: https://arxiv.org/abs/2102.07757


------

The folders in this repository contain two experiments:

1. [classifying-oscillations](./classifying-oscillations): toy example designed to assess the ability
    of convolutional neural networks to resolve between different frequencies at its input.
1. [quantifying-aliasing](./quantifying-aliasing): Scripts for quantifying to what extent aliasing takes 
    place in the intermediate layers of the neural network
    
Requirements
-----------

The file `requirements.txt` contains the python modules required. 
The versions specified are the ones the code has been tested on. Nonetheless,
I believe lower versions of most packages should also work. One exception is matplotlib where I observed that using versions different than
the 3.2.1 might yield minor changes (namely, different axis ticks).

Finally, some experiments also require ImageNet validation set. I include basic
instructions for applying for the license, downloading the dataset and extracting it [here](./quantifying-aliasing/README.md).

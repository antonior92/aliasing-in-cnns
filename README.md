# How Convolutional Neural Networks Deal with Aliasing

Python scripts for reproducing the up-coming paper: "How Convolutional Neural Networks Deal with Aliasing".

```
Ribeiro, A.H. and Schon T.B. "How Convolutional Neural Networks Deal with Aliasing". Under Review.
```


The folders in this repository contain two experiments:

1. [classifying-oscillations](./classifying-oscillations/README.md): toy example designed to assess the ability
    of convolutional neural networks to resolve between different frequencies at its input.
1. [quantifying-aliasing](./quantifying-aliasing/README.md): Scripts for quantifying to what extent aliasing takes 
    place in the intermediate layers of the neural network
    
Requirements
-----------

The file `requirements.txt` contains the python modules required. 
The versions specified are the ones the code has been tested on. Nonetheless,
I believe lower versions of most packages should also work. One exception is matplotlib where I observed that using versions different than
the 3.2.1 might yield minor changes (namely, different axis ticks).

Finally, some experiments also require ImageNet validation set. I include basic
instructions for applying for the license, downloading the dataset and extracting it [here](./quantifying-aliasing/README.md).
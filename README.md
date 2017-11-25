# MNIST Digit Recognizer Kaggle

## Overview:
Implemented deep convolutional neural network using TensorFlow to correctly identify digits from handwritten images.

Improved accuracy using Xavier initialization, batch normalization, data augmentations like random rotation, shift, etc.

Achieved 99.4% accuracy on test set. 

Ranked in top 13% in Kaggle.

## Architecture:
The architecture is quite simple. It consists of 3 convolutional layers with 32, 64, 128 filters respectively. All of the convolutional layers have a kernel size of 3x3, activation as ReLU. Every convolutional layer is followed by a max pooling of window size 2x2 and strides (1,1). Finally, two fully connected layers are attached. Variables are initialized using Xavier Initialization. 

Loss Function: Softmax cross entropy.

Optimizer: Adam

### Dataset:
The dataset was obtained from [Digit Recognizer from Kaggle](https://www.kaggle.com/c/digit-recognizer). It is first normalized it between 0 and 1, and is further augmented during every training epoch.

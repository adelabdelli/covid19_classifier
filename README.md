# CoViD-19 Classifier


In this repository, i will show you how to automatically detect COVID-19 in X-ray images dataset using Deep Learning (Convolutional Neural Network).
The dataset contains X-ray images, 25 normal and 25 with covid-19 case. The used dataset is from a tutorial of  [yImagesSearch](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)
The used CNN architecture is the [SqueezeNet](https://arxiv.org/abs/1602.07360).

## Requirements
1- Python 
2- Tensorflow
3- Numpy

# Classification Report

              precision    recall  f1-score   support

       covid     1.0000    1.0000    1.0000         8
      normal     1.0000    1.0000    1.0000         8

   micro avg     1.0000    1.0000    1.0000        16
   macro avg     1.0000    1.0000    1.0000        16
weighted avg     1.0000    1.0000    1.0000        16

# P.S This repository is not for professional use, it needs more examples, more than 1000 to make it somehow sure.

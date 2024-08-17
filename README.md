# Handwritten Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) similar to the TinyVGG architecture to recognize handwritten digits. The model was trained on the MNIST dataset and tested on custom handwritten digits.

## Model Overview

The CNN architecture is inspired by TinyVGG and consists of multiple convolutional layers followed by ReLU activation, max-pooling, and fully connected layers. The model achieved **98.12% accuracy (on random seed and accuracy much be better/ worse on other seeds)** on both the MNIST test data and custom handwritten digits created using GIMP.

## Dataset

- **Training Data:** MNIST dataset of handwritten digits (60,000 training images)
- **Testing Data:** MNIST test set (10,000 images) and custom images created in GIMP

## Results

The model achieved:
- **98.12% accuracy (on random seed and accuracy much be better/ worse on other seeds)** on custom handwritten digits

## Requirements

To run this project, you will need the following dependencies:

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib

NOTE: The images in MNIST dataset, on which the model are greyscale and 28x28 pixels in size. So, ensure to test on images of same size and color channels.

## Contributor
B M Manohara @Manohara-Ai

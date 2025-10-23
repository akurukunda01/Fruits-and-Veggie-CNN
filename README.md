# Fruits-and-Veggie-CNN
## Welcome to this Fruit and Vegetable Image Classification Project!

This notebook demonstrates the process of building and evaluating convolutional neural networks (CNNs) for classifying images of fruits and vegetables.

## Project Overview
The goal of this project is to train a model that can accurately identify different types of fruits and vegetables from images. We will explore the use of pre-trained models and transfer learning to achieve high accuracy on a diverse dataset.

## Dataset
We will be using the "moltean/fruits" dataset from Kaggle, which contains a large collection of images of various fruits and vegetables. The dataset is split into training and testing sets, and has over 223 different classes.

## Approach
1.  **Data Loading and Preparation**: Download the dataset and prepare the image data for input into our models using PyTorch's data loading and augmentation capabilities.
2.  **Model Selection and Setup**: Set up two different CNN architectures, ResNet50 and EfficientNetB0, leveraging pre-trained weights for transfer learning.
3.  **Training**: Train both models on the prepared training dataset, monitoring their performance over epochs.
4.  **Testing**: Evaluate the trained models on the unseen test dataset to assess their generalization ability and compare their performance.
5.  **Prediction**: Use the trained model to predict the class of a single image.

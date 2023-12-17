# BAMS - CIFAR10 Convolutional Neural Network

## About

This is an implementation in Python with PyTorch + CUDA of a CNN capable of identifying images from the CIFAR10 Dataset.

## Table of Contents

- [Project Architecture](#project-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)

## Project Architecture

- **/data**
  - **/train**
    - Training Images
  - **/test**
    - Test Images

- **./**
  - **Cifar10CnnModel.py**
    - Definition and Implementation of the CIFAR10 CNN model
  - **DeviceDataLoader.py**
    - Handler for utilizing CUDA
  - **ImageClassificationBase.py**
    - Contains helper methods for training & validation
  - **training.py**
    - Script for training the model
  - **tests.py**
    - Script for loading and testing the model

## Dependencies

- Python 3.x [https://www.python.org/]
- PyTorch [https://pytorch.org/]
- Matplotlib [https://matplotlib.org/]
- CUDA [https://developer.nvidia.com/cuda-toolkit]
- CIFAR10 [https://www.cs.toronto.edu/~kriz/cifar.html]

## Installation

1. Clone this repository
2. Download the CIFAR10 and follow the project architecture
3. Install CUDA devkit
4. Install PyTorch `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. Install Matplotlib `pip3 install matplotlib`

## Usage

- For training and generating the model (if needed), navigate to the project directory and run:
```bash
python training.py
```
- For testing with random images from the set, navigate to the project directory and run:
```bash
python tests.py 
```

## Results

- The model in the actual state v 1.0 could achive a 81% of accuracy within the train of 20 epochs
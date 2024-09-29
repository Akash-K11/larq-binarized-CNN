# Binarized CNN for MNIST Classification

## Introduction

This project implements a Binarized Convolutional Neural Network (BCNN) using Larq and TensorFlow to classify MNIST digits. The primary goal is to explore the effectiveness of binarized neural networks in image classification tasks while significantly reducing model complexity and computational requirements.

### Purpose and Insights

1. **Model Efficiency**: The binarized CNN achieves an impressive around 97% accuracy on the MNIST test set while using only 93,790 parameters, of which 93,322 are trainable. This demonstrates the power of binarized networks in maintaining high performance with a reduced parameter count.

2. **Binary Operations**: Most layers in the network use binary weights and activations, which can lead to faster inference times and lower power consumption, making this model potentially suitable for deployment on edge devices or in resource-constrained environments.

3. **Training Stability**: The training output shows that the model converges quickly, reaching high accuracy within just 6 epochs. This suggests that binarized CNNs can be trained efficiently, potentially reducing the time and computational resources required for model development.

4. **Generalization**: The small gap between training and validation accuracy indicates good generalization, suggesting that binarized CNNs can avoid overfitting even with their reduced complexity.

By exploring binarized CNNs, this project aims to contribute to the ongoing research in efficient deep learning models, which are crucial for deploying AI in resource-constrained environments such as mobile devices, IoT, and edge computing scenarios.

## Project Structure

- `larq-bcnn-script.py`: Main script for training and evaluating the BCNN model
- `larq_bcnn.ipynb`: Jupyter notebook for testing and experimentation
- `requirements.txt`: List of required Python packages

## Requirements

To run this project, you need Python 3.8 and the following packages:

```
larq==0.13.3
matplotlib==3.5.0
numpy==1.18.5
pandas==1.0.5
protobuf==3.20.3
scikit-learn==0.22.2
scipy==1.4.1
seaborn==0.10.1
tensorflow==2.3.0
tensorflow-estimator==2.3.0
```

You can install these requirements using pip:

```
pip install -r requirements.txt
```

## Model Architecture

The binarized CNN model consists of the following layers:

1. Input Layer: 28x28x1 (MNIST image size)
2. Quantized Convolutional Layer: 32 filters, 3x3 kernel
3. Max Pooling Layer: 2x2
4. Batch Normalization
5. Quantized Convolutional Layer: 64 filters, 3x3 kernel
6. Max Pooling Layer: 2x2
7. Batch Normalization
8. Quantized Convolutional Layer: 64 filters, 3x3 kernel
9. Batch Normalization
10. Flatten Layer
11. Quantized Dense Layer: 64 units
12. Batch Normalization
13. Quantized Dense Layer: 10 units (output layer)
14. Batch Normalization
15. Softmax Activation

The model uses the Straight-Through Estimator (STE) for binarization of weights and activations.

## Usage

To train and evaluate the model, run the `larq-bcnn-script.py` file:

```
python larq-bcnn-script.py
```

This script will:
1. Load and preprocess the MNIST dataset
2. Create and compile the binarized CNN model
3. Train the model for 6 epochs
4. Evaluate the model on the test set
5. Generate a confusion matrix
6. Plot training history (accuracy and loss)

## Results

The model typically achieves around 97% accuracy on the MNIST test set. The exact results may vary slightly due to the random nature of neural network training.

## Visualizations

The script generates two main visualizations:

1. Confusion Matrix: Shows the model's classification performance across all digits.
2. Training History: Displays the accuracy and loss curves for both training and validation sets.

These visualizations help in understanding the model's performance and training progression.
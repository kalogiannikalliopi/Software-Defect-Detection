# Software Defect Detection

## Introduction

Software defect detection plays a crucial role in ensuring the reliability and quality of software systems. By leveraging machine learning techniques, it is possible to automatically identify defective components in programs, reducing maintenance costs and improving overall software performance.

This project explores the application of multiple classifiers across different datasets to evaluate their effectiveness in software defect detection. The primary goal is to compare and measure the performance of several well-established machine learning algorithms in identifying software defects.

### Classifiers

The classifiers to be studied are:
* $C_1$: Logistic Regression
* $C_2$: Perceptron
* $C_3$: Support Vector Machines (with linear &amp; RBF kernel)
* $C_4$: Decision Tree
* $C_5$: Random Forests
* $C_6$: Feed-forward Neural Network

### Datasets

The classification performance of the abovementioned algorithms will be studied in a collection of 3 datasets on software defect detection. The datasets are:

* `jm1` (Description [here](https://www.openml.org/search?type=data&status=active&id=1053))
* `mc1` (Description [here](https://www.openml.org/search?type=data&status=active&id=1056))
* `pc3` (Description [here](https://www.openml.org/search?type=data&status=active&id=1050))

## Experiments

To ensure a fair and robust evaluation, the experiments follow these steps:
* Data Splitting: Each dataset is divided into 80% training and 20% testing.
* Validation: A 5-fold cross-validation technique is applied to assess generalization performance.
* Evaluation Metrics: The algorithms are compared using four key metrics:
    * M1: Accuracy
    * M2: F1-score
    * M3: G-Mean score
    * M4: Fit time

## Feature Normalization

The experiments are repeated using different feature normalization methods to analyze their impact on model performance:
* N1: No normalization (raw features)
* N2: Min-Max normalization
* N3: Feature standardization
# Introduction to Convolutional Neural Networks and Image Classification

## Introduction

This is Assignment 2 of the course Fundamentals of Deep Learning (DA6401) at IITM. The assignment is based on the Nature_12k dataset which contains 12,000 images of 10 different classes. The dataset is available at [https://www.kaggle.com/prasunroy/nature-12k-dataset](https://www.kaggle.com/prasunroy/nature-12k-dataset).

## Instructions

The code is written in Python 3.10.9 and the following libraries were used:
 - pytorch 2.0
 - pytorch-lightning 2.0.0
 - matplotlib
 - wandb
 - torchmetrics
 - torchvision
 - numpy

It is advised to run the code in a virtual environment. The following commands can be used to create a virtual environment and install the required libraries:
```

pip install -r requirements.txt

```

The code can be run using the following command after navigating to the appropriate directory:
```
python main.py
```

The default set of parameters are not the best parameters for the model that is trained from scratch whereas the finetuned model is trained using the best parameters.

The parameters and the help messages can be viewed using the following command:
```
python main.py --help
```

The results are discussed in a detailed manner in the report. You may find them in the report.pdf file.

## References

- Lecture Slides
- Wandb documentation.
- Pytorch documentation.
- Lightning AI documentation.
- Sebastian Raschka's linkedin post chain on finetuning models.

## Acknowledgements

I sincerely thank my course instructor Prof. Mitesh Khapra for the in-depth lectures and the vast amount of concepts that he covered.
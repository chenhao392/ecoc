---
layout: default
---

# Introduction
We work on gene function prediction problems by focusing on a typical scenario in functional or disease-associated gene screening projects. **Given a set of cooperating genes, predict their additional members.**

>In principle, these cooperating genes must co-exist at the same time and location. Following this "Guilt By Association" principle, these genes may physically and genetically interact with each other,  co-evolve across multiple species, and co-express in different tissues. In the genomic scale, these gene-gene associations form biological association networks. Started by assigning a label to these genes and related labels to other sets, this model propagates these labels on the networks. It then integrates the propagated values from different biological perspectives and models label relationships. After training the parameters, this model can predict labels for genes.

As a user-friendly toolset, this package implements the functions for not only label propagation, model training, and prediction, but also some ancillary services for fast correlation calculation and an algorithm that enhance local structures in networks (*Wang  et al.*). The model uses an " Error correction of code" framework that models multiple labels together and reduces errors. If interested, please read the [model description and implementation details](./model.html).  The implementation is done in Golang and organized into ***the following sub-commands.***. 
 - **neten**
	 - ***network enhancement***. A GO implementation for the network enhancement algorithm mentioned with all default parameters. The algorithm reads in a network and outputs another one with enhanced local structures. 
 - **pcc**
	 - ***fast correlation calculation***. An ultrafast, scalable, and memory-efficient implementation for *Pearson* correlation coefficient calculation. It is optimized for large-scale row/column-wise correlation calculation, such as a 20,000 by 20,000 tab-delimited matrix. 
 -  **prop**
	 - ***multi-label propagation***.  A GO implementation to a modified PRINCE label propagation algorithm. It propagates a set of labels on a list of input networks and integrates the results into a gene by label matrix. 
 - **tune**
	 - ***hyperparameter tuning and automatic prediction***.  Run of the model with automatical hyperparameters tuning and gene label predictions.
  -  **pred**
	 - ***prediction with specific hyperparameters***.  With user-defined hyperparameters and some initially labeled gene sets, predict labels for genes. 
 - **report**
	 - ***calculate benchmark scores***. With predicted label probabilities and ground truth label matrix, calculated accuracy, microF1, micro/macro area under the Precision-Recall curve as benchmark scores.

# Features

## Multi-label modeling
The model learns the multi-label relationships and trains an error-correcting decoder to decode network propagated labels values to labels.  As stated, please read the [model description and implementation](./model.html) for details.

## Second-order iterative stratification 
To preserve better the label relationship in a training subset, we implemented a second-order iterative stratification (SOIS) procedure for stratifying datasets. It is essential for the gene function prediction problems, as the labels are usually imbalanced. This SOIS procedure iteratively distributes a most demanding multi-label gene to a subset that preserves the label dependencies. 

## Platt's scaling for probability calibration
Transferring the output scores to reliable probabilities is also not a trivial task, as machine learning methods tend to produce skewed probability distributions. We implemented a modified version of Platt’s scaling to calibrate the probabilities, which use logistic regression with known positive and negative labels in training.
## Tolerenting missing positive labels
Missing positive labels in the training dataset may confuse the training process. In this package, we added a further modification to estimate and remove a fraction of top scores before probability scaling, which alleviates the effect of high probabilities assigned to false-negative genes in calibration.

## Automatical threshold estimation
We estimate thresholds for each of the predicted labels, based on matching training and testing label probability distributions and the harmonic mean of precision and recall. 

## Goroutines for scalable computing
The package supports multi goroutines/threads for both the multi-label multi-network propagations and the expensive mean-field approximation step in decoding the network propagated values back to gene labels. 

# Installation
For most users, please download the pre-compiled binary files from the [release](https://github.com/chenhao392/ecoc/releases) and install the [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) package. If you'd like to build from source, you can download the source files and compile in a [GO](https://golang.org/doc/install) environment. Please read this [page](./installation.html)  for exemplary installations on ubuntu and mac machines and error massages from not correctly installed liblinear packages. 

# Quick Start
 This package is organized as a suite of commands using Cobra. Help information for input data description, sub-commands, and parameters are printed to the standard output. For instance, a call for ecoc will print the following to the terminal.  In addition, A vignette using this tool for predicting piRNA pathway genes in *D. melanogaster* is available in this [demo](./demo1.html).
```

  ______ _____ ____   _____ 
 |  ____/ ____/ __ \ / ____|
 | |__ | |   | |  | | |     
 |  __|| |   | |  | | |     
 | |___| |___| |__| | |____ 
 |______\_____\____/ \_____|

 Encoding and decoding label set with error correction.

Usage:
  ecoc [command]

Available Commands:
  help        Help about any command
  neten       network enhancement
  pcc         fast correlation calculation
  pred        prediction with specific hyperparameters
  prop        multi-label propagation
  report      calculate benchmark scores
  tune        hyperparameter tuning and automatic prediction

Flags:
      --config string   config file (default is $HOME/.ecoc.yaml)
  -h, --help            help for ecoc

Use "ecoc [command] --help" for more information about a command.

``` 
 
# Reference
 - Wang, B., Pourshafeie, A., Zitnik, M. _et al._ Network enhancement as a general method to denoise weighted biological networks. _Nat Commun_  9, 3108 (2018). 



# Introduction
We work on gene function prediction problems by focusing on a typical scenario in functional or disease-associated gene screening projects. **Given a set of associated genes, predict their additional members.**

> Following the “Guilt By Association” principle, these genes may be physically and genetically interact with each other,  co-evolving across multiple species and co-express at the same tissue type and etc. These gene-gene associations are considered as biological association networks. This project uses the label propagation algorithm and ***error correction of code*** ( a multi-label learning framework) to model a focused function and its relationship with other labels. After training,  the model will assign these functional labels to new genes.

This package is implemented using Golang and Cobra, aiming for a user-friendly toolset for gene function prediction problem. Though it is mainly built for model tuning and gene function prediction, it also supports functionalities that are useful for other purposes, such as network enhancement algorithm in Wang *et al.* and fast large-scale Pearson correlation coefficient calculation. With Cobra, they are organized into ***the following sub-commands.***

 - **neten**     
	 - ***network enhancement***. A GO implementation for the network enhancement algorithm in Wang *et al.*, all default parameters. The algorithm enhances a network's local structure and removes noise. 
 - **pcc**
	 - ***fast correlation calculation***. An ultrafast, scalable and memory efficient implementation for *Pearson* correlation coefficient calculation. Optimized for large-scale row/column-wise correlation calculation, such as a 20,000 by 20,000 tab delimited matrix. 
 -  **prop**
	 - ***multi-label propagation***.  A GO implementation and modification to the PRINCE algorithm. It propagates a set of labels on a list of input networks and summarizes the results into a gene by label matrix, where the columns stack the propagated labels from different networks. 
 - **tune**
	 - ***hyperparameter tuning and automatic prediction***. Automatical run of the model, which is consist of multi-label propagation, hyperparameters tuning ( k CCA dimentions and lamda for balancing potentials), and gene label predictions.
 -  **pred**
	 - ***prediction with specific hyperparameters***.  With user-defined hyperparameters, predicting gene label with a set of genes with known labels. 
 - **report**
	 - ***calculate benchmark scores***. With predicted label probabilities and ground truth label matrix, calculated accuracy, microF1, micro/macro area under the Precision-Recall curve as benchmark scores.

# Installation
For most of the users, please download the pre-compiled binary files from the [release](https://github.com/chenhao392/ecoc/releases). If you'd like to build from source, you can download the source files and compile in a [GO](https://golang.org/doc/install) environment. Please note that the [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) package from Lin's group must also be installed.

<details> <summary>Click here for guide for compiling liblinear and config.</summary>

```
# instll liblinear
tar -xf liblinear-2.30.tar.gz
cd liblinear-2.30
make lib
ln -s liblinear.so.3 liblinear.so

# config for osx
# please add the following to your ~/.bash_profile
export LD_LIBRARY_PATH="/path/to/liblinear-2.30:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/path/to/liblinear-2.30:$DYLD_LIBRARY_PATH"
export C_INCLUDE_PATH="/path/to/liblinear-2.30:$C_INCLUDE_PATH"

# config for ubuntu
# please add the following to your ~/.bashrc
export LD_LIBRARY_PATH="/path/to/liblinear-2.30:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/path/to/liblinear-2.30:$LIBRARY_PATH"
export C_INCLUDE_PATH="/path/to/liblinear-2.30:$C_INCLUDE_PATH"
```

</details>
 
```
# compile ecoc from source
git init
git pull https://github.com/chenhao392/ecoc
go build
```
# Features

### Multi-label modeling
The model learns a latent structure that maximizes the correlations between a set of associated labels and their propagated scores on networks. If a structure is successfully learned,  it can be used to decode the propagated scores back to labels for genes, assigning labels to genes that were not annotated.  Please read the case study for predicting piRNA pathway genes in D. melanogaster for demo.
  
### Second-order iterative stratification 
To better preserve the label relationship in a subset,  especially for the imbalanced gene function labels, a second-order iterative stratification (SOIS) procedure is implemented in this model for stratifying datasets (Szymański and Kajdanowicz, 2017; Sechidis et al., 2011). This SOIS procedure iteratively distributes the most demanding multi-label genes each time to a subset. 

### Platt's scaling for probability calibration
Transferring the output scores of a classifier to reliable probabilities is also not a trivial task, as machine learning methods tend to produce skewed probability distributions. A modified version of Platt’s scaling is implemented, according to the pseudocode from Hsuan-Tien Lin et al. Platt’s scaling is a simple probability calibration method that uses logistic regression to calibrate the probabilities, using known positive and negative labels in training (Platt, 1999).

### Tolerenting missing positive labels
Missing positive labels in the training dataset may confuse the training process. In this package, a further modification to estimate and remove a fraction of top scores before Platt’s scaling is used, which alleviates the effect of high probabilities assigned to false-negative genes in calibration (Rüping, 2006).

### Goroutines for scalable computing
The package supports multi goroutines/threads for both the multi-label multi-network propagations and the expensive mean-field approximation step in decoding the network propagated values back to gene labels. 

# Common problems


 ***Missing liblinear shared library.*** 
I downloaded the pre-compiled binary file from the release. When I try to exeucute, it complains that liblinear.so.3 can not be found, such as the following. 
```
# example error msg from Ubuntu
error while loading shared libraries: liblinear.so.3: cannot open shared object file: No such file or directory
```
```
# example error msg from osx
dyld: Library not loaded: liblinear.so.3
Referenced from: /Users/chen/work/ecoc/./ecoc
Reason: image not found
Abort trap: 6
```
***Solution***: the liblinear package is either not installed or not properly configured. Please see installation for examplary configrations in Ubuntu and Mac machines. 

 ***Missing liblinear shared library and head files for compiling from sounrce.*** 
 I downloaded the source files and tried to compile it. But it complains that it cannot find -llinear or linear.h. 
```
# missing shared library
/usr/local/go/pkg/tool/linux_amd64/link: running gcc failed: exit status 1
/usr/bin/ld: cannot find -llinear
collect2: error: ld returned 1 exit status

# missing head file
src/ml_linear.go:5:10: fatal error: 'linear.h' file not found
#include <linear.h>
^~~~~~~~~~
1 error generated.
```
***Solution***: the liblinear package is either not installed or not properly configured. Please see installation for examplary configrations in Ubuntu and Mac machines. 

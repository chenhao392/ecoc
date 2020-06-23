

# Introduction
We work on gene function prediction problems by focusing on a typical scenario in functional or disease-associated gene screening projects. **Given a set of co-functioning genes, predict their additional functional members.**

> Following the “Guilt By Association” principle, these genes may be physically and genetically interact with each other,  co-evolving across multiple species and co-express at the same tissue type and etc. These gene-gene associations are considered as biological association networks. This project uses the label propagation algorithm and ***error correction of code*** ( a multi-label learning framework) to model a focused function and its relationship with other labels. After training,  the model will assign these functional labels to new genes.

This package is implemented using Golang and Cobra, aiming for a user friendly toolset for gene function prediction problem. Though it is mainly built for model tuning and gene function prediction, it also supports functionalities that are useful for other purposes, such as network enhancement algorithm in Wang *et al.* and fast large-scale Pearson correlation coefficient calculation. With Cobra, they are organized into ***the following sub-commands.***

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
For most of the users, please download the pre-compiled binary files from the [release](https://github.com/chenhao392/ecoc/releases). If you'd like to build from source, you can download the source files and compile in a [GO](https://golang.org/doc/install) environment. Please note that the [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) package from Lin's group must also be installed.<details> <summary>Click here for guide for compile and config.</summary>
 
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

 ### Multi-label framework modeling netowork data
 ### Second-order iterative stratification 
 ### Platt's scaling for probability distribution
 ### Tolerenting missing positive labels
 ### Goroutines for scalable computing

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

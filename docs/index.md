## Introduction
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

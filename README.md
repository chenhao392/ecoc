
**Given a set of co-functioning genes, predict their additional functional members.**

> Following the “Guilt By Association” principle, co-fucntioning genes may be co-evolving across multiple species and co-express at the same tissue type and etc. These gene-gene associations are considered as biological association networks. This project uses label propagation algorithm and ***error correction of code*** ( a multi-label learning framework) to model the focused functional label and its relationship with other labels, and then assign function label to genes.

This packages is implemented using Golang and Cobra, aiming for a straightforwards toolset for gene function prediction with custimized datasets. Though it is mainly for model tuning and predict additional genes of a certaion function, it also contains functionalities that may be used for other purpose, such as network enhancement algorithm in Wang et al. and fast large-scale Pearson correlation coefficient calculation. Specifically, **It contains the following sub-commands.**

 - neten     
	 - network enhancement
 - pcc
	 - fast Pearson correlation coefficient calculation
 -  pred
	 - prediction with specific hyperparameters
 -  prop
	 - multi-label propagation
 - report
	 - calculate per label benchmark scores
 - tune
	 - hyperparameter tuning and benchmarking

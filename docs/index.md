ECOC following "Guilt by association principle"

This is an semi-supervised multi-label learning toolbox using "Error Correction of Code" ensemble framework.
Following "Guilt by association" principle. Many prior information can be integrated, such as the followings.

* phylogenetic profiles across species
* expression profiles across tissue types
* GO semantic similarities in one species corpus
* protein-protein interactions
* genetic interactions
* positive selection pressure across closely related species

The toolbox is written in Golang and Cobra with the following sub-commands.

* cals
  * fast pearson correlation coefficient calculation for large matrix.
* tune
  * automatic hyper-parameter tuning and testing.
* pred
  * predict labels for a feature dataset with given hyper-parameters.
* report
  * print prediction result in detail.

---
layout: default
---

##  piRNA pathway gene prediction
Biological networks, known function genes, and a set of related labels are required to train an ECOC model. Knowing the piRNA pathway functions in germline cells, we'll build up context-specific networks for this tissue type with the help of a recently published single-cell dataset and find a set of related labels using gene set enrichment analysis. You can find all demo scripts and data in this [github repository](https://github.com/chenhao392/ecocDemo). 

### Building up context-specific networks
 With a recently published [single-cell RNAseq dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4363298)  for *D. melanogaster*'s ovary, we can prune the generally purposed [stringDb](https://string-db.org/cgi/input.pl) association networks into context-specific ones. This [single cell study](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000538) also classified the ovary cells into 32 groups, including germline clusters 1. The following R and Perl scripts download the data, map the gene IDs, and prepare the gene list for germline.
```
# cat step1.0.ovaryGenes.sh 
# download data
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4363nnn/GSM4363298/suppl/GSM4363298_7053cells_highquality_ovary.loom.gz
gunzip GSM4363298_7053cells_highquality_ovary.loom.gz

# process scRNAseq data 
Rscript scripts/demo.scRNAseqData.R

# mapping gene IDs to stringDb version 11
perl scripts/idMap.pl \
     data/matrix.ovary.txt \
     data/7227.protein.info.v11.0.txt \
     data/dm_id_mapper_all_mar_2017.txt \
     14,18 >data/matrix.ovary.idMapped.txt 2>/dev/null
perl scripts/idMap.pl \
     data/matrix.cellByGermGene.txt \
     data/7227.protein.info.v11.0.txt \
     data/dm_id_mapper_all_mar_2017.txt \
     14,18 >data/matrix.cellByGermGene.idMapped.txt 2>/dev/null
     
# genes expressed in germline cluster 1
cut -f1,2 data/matrix.cellByGermGene.idMapped.txt | perl -lane 'print if $F[1] >0' >data/ovary.germlineClust1.gene.txt
```
This study also provides gene expression profiles across cell sub-types in fly's ovary. To fully take advantage of this, we build an additional ovary-specific co-expression network, using the provided fast correlation calculation functionality in ECOC. A 0.5 correlation cut-off is used to remove less significant correlations in this network.
```
# cat step1.1.ovaryCoExpressionNetwork.sh
tail -n +2 data/matrix.cellByGermGene.idMapped.txt >tmp.txt
ecoc pcc --i tmp.txt --t 8 --o data/pcc.cellByGermGene.idMapped.txt
rm -f tmp.txt
perl scripts/matrix2pair.pl \
    data/pcc.cellByGermGene.idMapped.txt \
    0.5 >data/pairs.cellByGermGene.txt
```
 As a sanity check, a boxplot for the known piRNA pathway genes indeed indicates that they are mostly enriched in germline cluster 1. ![](/assets/images/top10celltypeExpresionPiRNAGenes.jpg) It also suggests that we can remove noises while preserving most piRNA pathway genes with a 0.1 average expression threshold. The following script generates context-specific networks with this cut-off. 
 
 ```
# cat sh step1.2. ovarySpecificNetworks.sh
perl  scripts/subsetPPI.pl  data/matrix.ovary.idMapped.txt  nets/dm_coe_net.txt  0.1  1 >nets/dm_coe_net.ovary.1.txt
perl  scripts/subsetPPI.pl  data/matrix.ovary.idMapped.txt  nets/dm_db_net.txt   0.1  1 >nets/dm_db_net.ovary.1.txt
perl  scripts/subsetPPI.pl  data/matrix.ovary.idMapped.txt  nets/dm_exp_net.txt  0.1  1 >nets/dm_exp_net.ovary.1.txt
perl  scripts/subsetPPI.pl  data/matrix.ovary.idMapped.txt  nets/dm_fus_net.txt  0.1  1 >nets/dm_fus_net.ovary.1.txt
perl  scripts/subsetPPI.pl  data/matrix.ovary.idMapped.txt  nets/dm_nej_net.txt  0.1  1 >nets/dm_nej_net.ovary.1.txt
perl  scripts/subsetPPI.pl  data/matrix.ovary.idMapped.txt  nets/dm_pp_net.txt   0.1  1 >nets/dm_pp_net.ovary.1.txt
perl  scripts/subsetPPI.pl  data/matrix.ovary.idMapped.txt	data/pairs.cellByGermGene.txt 0.1 1 >nets/pcc.cellByGermGene.ovary.1.txt
```
### Building up a multi-label matrix 
We can find the set of labels associated with the piRNA pathway using [g: profiler](https://biit.cs.ut.ee/gprofiler/gost)'s gene set enrichment analysis. Please note that both positively and negatively associated labels are informative. While selecting positively associated labels are straight forwarding, we can also find the negatively associated ones using the same enrichment analysis, with genes that are not annotated with the positively associated labels.  Then, with all selected labels, we can find all genes annotated by them and build the multi-label matrix. The R script in this step selects labels and retrieve genes with these labels. And the following Perl script maps gene IDs and convert them into matrix form. 
```
# cat step2.0.multiLabelMatrix.sh
Rscript scripts/demo.GOtermAsLabels.R
perl scripts/piRNAGeneMatrixForECOC.pl \
	data/gProfiler.piRNA_hc.ovaryGene.txt \
	data/dm.piRNA.nick.hc.2019.txt \
	data/7227.protein.info.v11.0.txt \
	data/gProfiler.piRNA_hc.ovaryGene.negCtrl.txt \
	data/fly.goTermOvary.piRNApathway.hc.withNeg.txt \
	data/fly.goTermOvary.piRNApathway.pseudoTsmatrix.hc.txt
```

### Predict piRNA pathway genes
Finally, we predict piRNA pathway genes using the prepared networks and the multi-label matrix, with a pseudo testing matrix generated in step 2. Please note that ECOC can predict labels for any genes with network propagated values, including genes used in training. 
```
# cat step3.0.ecocPrediction.sh
ecoc tune --n nets/dm_db_net.ovary.1.txt,nets/dm_exp_net.ovary.1.txt,nets/dm_fus_net.ovary.1.txt,nets/dm_nej_net.ovary.1.txt,nets/dm_pp_net.ovary.1.txt,nets/pcc.cellByGermGene.ovary.1.txt \
          --res result \
          --trY data/fly.goTermOvary.piRNApathway.hc.withNeg.txt \
          --tsY data/fly.goTermOvary.piRNApathway.pseudoTsmatrix.hc.txt \
          --t 55 --nFold 5 --k 10 --isFirstLabel --isCali --v 
```

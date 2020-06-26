piRNA pathway is known to function in germline cells. In this demo, we will predict its additional functional components in *D. melanogaster*'s ovary, by integrating [stringDb](https://string-db.org/cgi/input.pl) association networks with a recently published [single-cell RNAseq dataset](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000538). To build a suitable multi-label matrix for this demo,  We'll use [g: profiler](https://biit.cs.ut.ee/gprofiler/gost) for piRNA pathway genes' GO term enrichment analysis. All scripts and data are summarized in this github repository.  

### Building up context-specific networks
With a recently published single-cell level analysis on the ovary, we can prune the generally purposed stringDb networks into context-specific ones. The data is from NCBI ([GSM4363298](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4363298)), where the ovary cells are clustered into 32 groups, including germline clusters 1 and 2. The following R and Perl scripts process the dataset and generate ID mapped genes by group matrix for ovary and a gene list for germline cluster 1.
```
sh step1.0.ovaryGenes.sh
```
 As a sanity check, a boxplot for the known piRNA pathway genes also indicates that they are mostly enriched in germline cluster 1. Using a 0.1 average expression cut-off, we can remove the network genes that are not expressed in this group, generating context-specific networks for piRNA gene prediction. 
```
sh step1.1.ovarySpecificNetworks.sh
```
To fully take advantage of the single-cell RNAseq dataset, we will also utilize the provided fast correlation calculation functionality in ECOC to build an additional ovary-specific co-expression network. A 0.5 correlation cut-off is applied to this network.
```
# cat step1.2.ovaryCoExpressionNetwork.sh
ecoc pcc --i matrix.cellByGermGene.txt --t 55 --o pcc.cellByGermGene.txt
perl scripts/matrix2pair.pl pcc.cellByGermGene.txt 0.5 >pairs.cellByGermGene.txt
```

### Building up a multi-label matrix 
We build the multi-label matrix with the piRNA pathway's associated GO terms, both positively and negatively. The most positively associated labels are selected using g:profile's gene set enrichment analysis. After retrieving all ovary genes that are annotated with these GO terms, the negatively associated labels are also selected via the same enrichment analysis, with the remaining genes that are not annotated with the positively associated labels. 
```
sh step2.0.multiLabelMatrix.sh
```  

### Predict piRNA pathway genes
We predict addtional piRNA pathway genes using the prepared networks and the multi-label matrix. 
```
# cat step3.0.ecocPrediction.sh
ecoc tune --n nets/dm_db_net.ovary.1.txt,nets/dm_exp_net.ovary.1.txt,nets/dm_fus_net.ovary.1.txt,nets/dm_nej_net.ovary.1.txt,nets/dm_pp_net.ovary.1.txt,nets/pcc.cellByGermGene.ovary.1.txt \
          --res result \
          --trY data/fly.goTermOvary.piRNApathway.hc.withNeg.txt \
          --tsY data/fly.goTermOvary.piRNApathway.pseudoTsmatrix.hc.txt \
          --t 55 --nFold 5 --k 10 --isFirstLabel --isCali --v 
```

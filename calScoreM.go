package main

//#include <stdlib.h>
//import "C"
import (
	//"bufio"
	"flag"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"os"
	//"strconv"
	//"unsafe"
)

func main() {
	//input argv
	var tsY *string = flag.String("tsY", "data/yeast.level1.set1.tsMatrix.txt", "testLabelSet")
	var trY *string = flag.String("trY", "data/yeast.level1.set1.trMatrix.txt", "testLabelSet")
	var tsYh *string = flag.String("tsYh", "data/tsY.txt", "testLabelSet")
	//var resFolder *string = flag.String("res", "resultECOC", "resultFolder")
	var rankCut *int = flag.Int("r", 3, "rank cut for positive/negative classification label")
	flag.Parse()
	//read data
	tsYdata, _, _, _ := readFile(*tsY, true, true)
	trYdata, _, _, _ := readFile(*trY, true, true)
	tsYhat, _, _, _ := readFile(*tsYh, false, false)

	//microF1 := mat64.NewDense(1, 3, nil)
	//meanAupr := mat64.NewDense(1, 3, nil)

	//wg.Add(1)
	microF1, accuracy, macroAupr, microAupr := single_compute2(tsYdata, tsYhat, trYdata, *rankCut)
	fmt.Println(microF1, accuracy, macroAupr, microAupr)
	//oFile = "./" + *resFolder + "/m.macroF1.txt"
	//writeFile(oFile, macroF1)
	//oFile = "./" + *resFolder + "/m.meanAupr.txt"
	//writeFile(oFile, meanAupr)
	os.Exit(0)
}

func single_compute2(tsYdata *mat64.Dense, tsYhat *mat64.Dense, trYdata *mat64.Dense, rankCut int) (microF1 float64, accuracy float64, macroAupr float64, microAupr float64) {
	//F1 score
	_, nLabel := tsYdata.Caps()
	sumAupr := 0.0
	sumF1 := 0.0
	sumTp := 0
	sumFp := 0
	sumFn := 0
	sumTn := 0
	macroAuprSet := make([]float64, 0)
	accuracySet := make([]float64, 0)
	microF1Set := make([]float64, 0)
	for i := 0; i < nLabel; i++ {
		aupr := computeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
		macroAuprSet = append(macroAuprSet, aupr)
		sumAupr += aupr
		//os.Exit(0)
	}
	macroAupr = sumAupr / float64(nLabel)
	tsYhat = binPredByAlpha(tsYhat, rankCut)

	for i := 0; i < nLabel; i++ {
		f1, tp, fp, fn, tn := computeF1_3(tsYdata.ColView(i), tsYhat.ColView(i), rankCut)
		microF1Set = append(microF1Set, f1)
		accuracySet = append(accuracySet, (float64(tp)+float64(tn))/(float64(tp)+float64(fp)+float64(fn)+float64(tn)))
		sumF1 += f1
		sumTp += tp
		sumFp += fp
		sumFn += fn
		sumTn += tn
	}
	p := float64(sumTp) / (float64(sumTp) + float64(sumFp))
	r := float64(sumTp) / (float64(sumTp) + float64(sumFn))
	microF1 = 2.0 * p * r / (p + r)
	accuracy = (float64(sumTp) + float64(sumTn)) / (float64(sumTp) + float64(sumFp) + float64(sumFn) + float64(sumTn))
	//y-flat
	tsYdataVec := flat(tsYdata)
	tsYhatVec := flat(tsYhat)
	microAupr = computeAupr(tsYdataVec, tsYhatVec)
	for i := 0; i < nLabel; i++ {
		fmt.Printf("%d\t%.3f\t%.3f\t%.3f\n", i, accuracySet[i], microF1Set[i], macroAuprSet[i])
	}

	return microF1, accuracy, macroAupr, microAupr
}

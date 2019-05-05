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
	var tsY *string = flag.String("tsY", "data/tsY.txt", "testLabelSet")
	var tsYh *string = flag.String("tsYh", "data/tsY.txt", "testLabelSet")
	//var resFolder *string = flag.String("res", "resultECOC", "resultFolder")
	var rankCut *int = flag.Int("r", 3, "rank cut for positive/negative classification label")
	flag.Parse()
	//read data
	tsYdata, _, _, _ := readFile(*tsY, true, true)
	tsYhat, _, _, _ := readFile(*tsYh, false, false)

	//microF1 := mat64.NewDense(1, 3, nil)
	//meanAupr := mat64.NewDense(1, 3, nil)

	//wg.Add(1)
	microF1, accuracy, macroAupr, microAupr := single_compute(tsYdata, tsYhat, *rankCut)
	fmt.Println(microF1, accuracy, macroAupr, microAupr)
	//oFile = "./" + *resFolder + "/m.macroF1.txt"
	//writeFile(oFile, macroF1)
	//oFile = "./" + *resFolder + "/m.meanAupr.txt"
	//writeFile(oFile, meanAupr)
	os.Exit(0)
}

func single_compute2(tsYdata *mat64.Dense, tsYhat *mat64.Dense, rankCut int) (microF1 float64, accuracy float64, macroAupr float64, microAupr float64) {
	//F1 score
	_, nLabel := tsYdata.Caps()
	sumAupr := 0.0
	sumF1 := 0.0
	sumTp := 0
	sumFp := 0
	sumFn := 0
	sumTn := 0
	for i := 0; i < nLabel; i++ {
		aupr := computeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
		//fmt.Println(f1)
		sumAupr += aupr
	}
	tsYhat = binPredByAlpha(tsYhat, rankCut)

	for i := 0; i < nLabel; i++ {
		f1, tp, fp, fn, tn := computeF1_3(tsYdata.ColView(i), tsYhat.ColView(i), rankCut)
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
	macroAupr = sumAupr / float64(nLabel)

	//y-flat
	tsYdataVec := flat(tsYdata)
	tsYhatVec := flat(tsYhat)
	microAupr = computeAupr(tsYdataVec, tsYhatVec)
	return microF1, accuracy, macroAupr, microAupr
}

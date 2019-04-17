package main

//#include <stdlib.h>
//import "C"
import (
	//"bufio"
	"flag"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"os"
	"runtime"
	"strconv"
	"sync"
	//"unsafe"
)

var wg sync.WaitGroup
var mutex sync.Mutex

func main() {
	//input argv
	var tsY *string = flag.String("tsY", "data/tsY.txt", "testLabelSet")
	var resFolder *string = flag.String("res", "resultECOC", "resultFolder")
	var rankCut *int = flag.Int("r", 3, "rank cut for positive/negative classification label")
	flag.Parse()
	runtime.GOMAXPROCS(5)
	kSet := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	sigmaFctsSet := []float64{0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 16.0, 20.0}
	//read data
	tsYdata, _, _ := readFile(*tsY, false)
	//vars
	_, nLabel := tsYdata.Caps()
	//CCA dims
	minDims := nLabel
	//nK
	nK := 0
	for k := 0; k < len(kSet); k++ {
		if kSet[k] < minDims {
			nK += 1
		}
	}
	nL := nK * len(sigmaFctsSet)
	c := 0
	sumResF1 := mat64.NewDense(nL, nLabel, nil)
	sumResAupr := mat64.NewDense(nL, nLabel, nil)
	macroF1 := mat64.NewDense(nL, 3, nil)
	meanAupr := mat64.NewDense(nL, 3, nil)
	//decoding and step 4
	err := os.MkdirAll("./"+*resFolder, 0755)
	if err != nil {
		fmt.Println(err)
		return
	}

	wg.Add(nK * len(sigmaFctsSet))
	//wg.Add(1)
	for k := 0; k < nK; k++ {
		//for k := 0; k < 1; k++ {
		for s := 0; s < len(sigmaFctsSet); s++ {
			//for s := 0; s < 1; s++ {
			//go function
			//kSet[k]
			//sigmaFctsSet[s]
			go single_compute(tsYdata, kSet[k], sigmaFctsSet[s], sumResF1, macroF1, sumResAupr, meanAupr, nLabel, *rankCut, *resFolder, c)
			c += 1
		}
	}
	wg.Wait()
	oFile := "./" + *resFolder + "/sumRes.F1.txt"
	writeFile(oFile, sumResF1)
	oFile = "./" + *resFolder + "/sumRes.AUPR.txt"
	writeFile(oFile, sumResAupr)
	oFile = "./" + *resFolder + "/sumRes.macroF1.txt"
	writeFile(oFile, macroF1)
	oFile = "./" + *resFolder + "/sumRes.meanAupr.txt"
	writeFile(oFile, meanAupr)
	os.Exit(0)
}

func single_compute(tsYdata *mat64.Dense, k int, sigmaFcts float64, sumResF1 *mat64.Dense, macroF1 *mat64.Dense, sumResAupr *mat64.Dense, meanAupr *mat64.Dense, nLabel int, rankCut int, resFolder string, c int) {
	defer wg.Done()
	sFctStr := strconv.FormatFloat(sigmaFcts, 'f', 3, 64)
	kStr := strconv.FormatInt(int64(k), 16)
	inFile := "./" + resFolder + "/k" + kStr + "sFct" + sFctStr + ".txt"
	tsYhat, _, _ := readFile(inFile, false)
	//F1 score
	mutex.Lock()
	sumF1 := 0.0
	sumAupr := 0.0
	for i := 0; i < nLabel; i++ {
		f1 := computeF1_3(tsYdata.ColView(i), tsYhat.ColView(i), rankCut)
		aupr := computeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
		sumResF1.Set(c, i, f1)
		sumResAupr.Set(c, i, aupr)
		sumF1 += f1
		sumAupr += aupr
		//fmt.Println(f1)
	}
	macroF1.Set(c, 0, float64(k))
	macroF1.Set(c, 1, sigmaFcts)
	macroF1.Set(c, 2, sumF1/float64(nLabel))
	meanAupr.Set(c, 0, float64(k))
	meanAupr.Set(c, 1, sigmaFcts)
	meanAupr.Set(c, 2, sumAupr/float64(nLabel))
	mutex.Unlock()
}

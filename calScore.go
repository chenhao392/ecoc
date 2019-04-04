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
	var thres *float64 = flag.Float64("t", 0.001, "threshold for positive/negative classification for each label")
	flag.Parse()
	runtime.GOMAXPROCS(5)
	kSet := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	sigmaFctsSet := []float64{0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5}
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
	sumRes := mat64.NewDense(nL, nLabel, nil)
	macroF1 := mat64.NewDense(nL, 3, nil)
	//decoding and step 4
	err := os.MkdirAll("./"+*resFolder, 0755)
	if err != nil {
		fmt.Println(err)
		return
	}

	wg.Add(nK * len(sigmaFctsSet))
	for k := 0; k < nK; k++ {
		for s := 0; s < len(sigmaFctsSet); s++ {
			//go function
			//kSet[k]
			//sigmaFctsSet[s]
			go single_computeF1(tsYdata, kSet[k], sigmaFctsSet[s], sumRes, macroF1, nLabel, *thres, *resFolder, c)
			c += 1
		}
	}
	wg.Wait()
	oFile := "./" + *resFolder + "/sumRes.F1.txt"
	writeFile(oFile, sumRes)
	oFile = "./" + *resFolder + "/sumRes.macroF1.txt"
	writeFile(oFile, macroF1)
	os.Exit(0)
}

func single_computeF1(tsYdata *mat64.Dense, k int, sigmaFcts float64, sumRes *mat64.Dense, macroF1 *mat64.Dense, nLabel int, thres float64, resFolder string, c int) {
	defer wg.Done()
	sFctStr := strconv.FormatFloat(sigmaFcts, 'f', 3, 64)
	kStr := strconv.FormatInt(int64(k), 16)
	inFile := "./" + resFolder + "/k" + kStr + "sFct" + sFctStr + ".txt"
	tsYhat, _, _ := readFile(inFile, false)
	//F1 score
	mutex.Lock()
	sum := 0.0
	for i := 0; i < nLabel; i++ {
		f1 := computeF1_2(tsYdata.ColView(i), tsYhat.ColView(i), thres)
		sumRes.Set(c, i, f1)
		sum += f1
		//fmt.Println(f1)
	}
	macroF1.Set(c, 0, float64(k))
	macroF1.Set(c, 1, sigmaFcts)
	macroF1.Set(c, 2, sum/float64(nLabel))
	mutex.Unlock()
}

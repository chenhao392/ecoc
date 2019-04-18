package main

//#include <stdlib.h>
//import "C"
import (
	//"bufio"
	"flag"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	//"strconv"
	"sync"
	//"unsafe"
)

var wg sync.WaitGroup
var mutex sync.Mutex

func main() {
	//input argv
	var tsX *string = flag.String("tsX", "data/tsX.emo.txt", "test FeatureSet")
	var tsY *string = flag.String("tsY", "data/tsY.emo.txt", "test LabelSet")
	var trX *string = flag.String("trX", "data/trX.emo.txt", "train FeatureSet")
	var trY *string = flag.String("trY", "data/trY.emo.txt", "train LabelSet")
	var resFolder *string = flag.String("res", "resultEmo", "resultFolder")
	var inThreads *int = flag.Int("p", 48, "number of threads")
	var rankCut *int = flag.Int("c", 3, "rank cut (alpha) for F1 calculation")
	var reg *bool = flag.Bool("r", false, "regularize CCA, default false")
	kSet := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	//0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 and rev
	sigmaFctsSet := []float64{0.0001, 0.0025, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1, 1.23, 1.56, 2.04, 2.78, 4.0, 6.25, 11.11, 25.0, 100.0, 400.0, 10000.0}
	nFold := 5
	flag.Parse()
	rand.Seed(1)
	runtime.GOMAXPROCS(*inThreads)
	//read data
	tsXdata, _, _ := readFile(*tsX, false)
	tsYdata, _, _ := readFile(*tsY, false)
	trXdata, _, _ := readFile(*trX, false)
	trYdata, _, _ := readFile(*trY, false)

	_, nFea := trXdata.Caps()
	//nTs, _ := tsXdata.Caps()
	_, nLabel := trYdata.Caps()
	//nRowTsY, _ := tsYdata.Caps()
	//min dims
	//potential bug when cv set's minDims is smaller
	minDims := int(math.Min(float64(nFea), float64(nLabel)))
	//nK
	nK := 0
	for k := 0; k < len(kSet); k++ {
		if kSet[k] < minDims {
			nK += 1
		}
	}
	//measures
	nL := nK * len(sigmaFctsSet)
	sumResF1 := mat64.NewDense(nL, nLabel, nil)
	sumResAupr := mat64.NewDense(nL, nLabel, nil)
	macroF1 := mat64.NewDense(nL, 3, nil)
	meanAupr := mat64.NewDense(nL, 3, nil)
	//out dir
	err := os.MkdirAll("./"+*resFolder, 0755)
	if err != nil {
		fmt.Println(err)
		return
	}

	ecocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, sigmaFctsSet, sumResF1, sumResAupr, macroF1, meanAupr, nFold, nK, true)

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
func single_IOC_MFADecoding_and_result(outPerLabel bool, nTs int, k int, c int, tsY_Prob *mat64.Dense, tsY_C *mat64.Dense, sigma *mat64.Dense, Bsub *mat64.Dense, sigmaFcts float64, nLabel int, sumResF1 *mat64.Dense, macroF1 *mat64.Dense, sumResAupr *mat64.Dense, meanAupr *mat64.Dense, tsYdata *mat64.Dense, rankCut int) {
	defer wg.Done()
	tsYhat := mat64.NewDense(nTs, nLabel, nil)
	for i := 0; i < nTs; i++ {
		//the doc seems to be old, (0,x] seems to be correct
		//dim checked to be correct
		tsY_Prob_slice := tsY_Prob.Slice(i, i+1, 0, nLabel)
		tsY_C_slice := tsY_C.Slice(i, i+1, 0, k)
		arr := IOC_MFADecoding(nTs, mat64.DenseCopyOf(tsY_Prob_slice), mat64.DenseCopyOf(tsY_C_slice), sigma, Bsub, k, sigmaFcts, nLabel)
		tsYhat.SetRow(i, arr)
	}
	//sFctStr := strconv.FormatFloat(sigmaFcts, 'f', 3, 64)
	//kStr := strconv.FormatInt(int64(k), 16)
	//oFile := "./" + resFolder + "/k" + kStr + "sFct" + sFctStr + ".txt"
	//writeFile(oFile, tsYhat)
	//F1 score
	mutex.Lock()
	sumF1 := 0.0
	sumAupr := 0.0
	for i := 0; i < nLabel; i++ {
		f1 := computeF1_3(tsYdata.ColView(i), tsYhat.ColView(i), rankCut)
		aupr := computeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
		if outPerLabel {
			sumResF1.Set(c, i, f1)
			sumResAupr.Set(c, i, aupr)
		}
		sumF1 += f1
		sumAupr += aupr
	}
	macroF1.Set(c, 0, float64(k))
	macroF1.Set(c, 1, sigmaFcts)
	macroF1.Set(c, 2, sumF1/float64(nLabel)+macroF1.At(c, 2))
	meanAupr.Set(c, 0, float64(k))
	meanAupr.Set(c, 1, sigmaFcts)
	meanAupr.Set(c, 2, sumAupr/float64(nLabel)+meanAupr.At(c, 2))
	mutex.Unlock()
}
func single_adaptiveTrainRLS_Regress_CG(i int, trXdataB *mat64.Dense, nFold int, nFea int, nTr int, tsXdataB *mat64.Dense, sigma *mat64.Dense, trY_Cdata *mat64.Dense, nTs int, tsY_C *mat64.Dense, randValues []float64, idxPerm []int) {
	defer wg.Done()
	beta, _, optMSE := adaptiveTrainRLS_Regress_CG(trXdataB, trY_Cdata.ColView(i), nFold, nFea, nTr, randValues, idxPerm)
	mutex.Lock()
	sigma.Set(0, i, math.Sqrt(optMSE))
	//bias term for tsXdata added before
	element := mat64.NewDense(0, 0, nil)
	element.Mul(tsXdataB, beta)
	for j := 0; j < nTs; j++ {
		tsY_C.Set(j, i, element.At(j, 0))
	}
	//fmt.Println(i, lamda, sigma)
	mutex.Unlock()
}

func ecocRun(tsXdata *mat64.Dense, tsYdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, rankCut *int, reg *bool, kSet []int, sigmaFctsSet []float64, sumResF1 *mat64.Dense, sumResAupr *mat64.Dense, macroF1 *mat64.Dense, meanAupr *mat64.Dense, nFold int, nK int, outPerLabel bool) (err error) {
	colSum, trYdata := posFilter(trYdata)
	tsYdata = posSelect(tsYdata, colSum)
	//vars
	nTr, nFea := trXdata.Caps()
	nTs, _ := tsXdata.Caps()
	_, nLabel := trYdata.Caps()
	nRowTsY, _ := tsYdata.Caps()

	if nFea < nLabel {
		fmt.Println("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
		os.Exit(0)
	}

	//tsY_prob
	tsY_Prob := mat64.NewDense(nRowTsY, nLabel, nil)
	//adding bias term for tsXData
	ones := make([]float64, nTs)
	for i := range ones {
		ones[i] = 1
	}
	tsXdataB := colStack(tsXdata, ones)
	//adding bias term for trXdata
	ones = make([]float64, nTr)
	for i := range ones {
		ones[i] = 1
	}
	trXdataB := colStack(trXdata, ones)
	//step 1
	for i := 0; i < nLabel; i++ {
		wMat, _, _ := adaptiveTrainLGR_Liblin(trXdata, trYdata.ColView(i), nFold, nFea)
		element := mat64.NewDense(0, 0, nil)
		element.Mul(tsXdataB, wMat)
		for j := 0; j < nTs; j++ {
			//the -1*element.At() is changed to 1*element.At() in this implementation
			//the sign was flipped in this golang library, as manually checked
			value := 1.0 / (1 + math.Exp(1*element.At(j, 0)))
			tsY_Prob.Set(j, i, value)
		}
	}
	fmt.Println("pass step 1 coding\n")

	//cca
	B := mat64.NewDense(0, 0, nil)
	if !*reg {
		var cca stat.CC
		err := cca.CanonicalCorrelations(trXdataB, trYdata, nil)
		if err != nil {
			log.Fatal(err)
		}
		B = cca.Right(nil, false)
	} else {
		//B is not the same with matlab code
		//_, B = ccaProjectTwoMatrix(trXdataB, trYdata)
		B = ccaProject(trXdataB, trYdata)
	}
	fmt.Println("pass step 2 cca coding\n")

	//CCA code
	trY_Cdata := mat64.NewDense(0, 0, nil)
	trY_Cdata.Mul(trYdata, B)
	//decoding with regression
	tsY_C := mat64.NewDense(nRowTsY, nLabel, nil)
	sigma := mat64.NewDense(1, nLabel, nil)
	//for workers
	randValues := RandListFromUniDist(nTr)
	idxPerm := rand.Perm(nTr)
	wg.Add(nLabel)
	for i := 0; i < nLabel; i++ {
		go single_adaptiveTrainRLS_Regress_CG(i, trXdataB, nFold, nFea, nTr, tsXdataB, sigma, trY_Cdata, nTs, tsY_C, randValues, idxPerm)
	}
	wg.Wait()
	fmt.Println("pass step 3 cg decoding\n")
	//decoding and step 4
	c := 0
	wg.Add(nK * len(sigmaFctsSet))
	for k := 0; k < nK; k++ {
		Bsub := mat64.DenseCopyOf(B.Slice(0, nLabel, 0, kSet[k]))
		for s := 0; s < len(sigmaFctsSet); s++ {
			go single_IOC_MFADecoding_and_result(outPerLabel, nTs, kSet[k], c, tsY_Prob, tsY_C, sigma, Bsub, sigmaFctsSet[s], nLabel, sumResF1, macroF1, sumResAupr, meanAupr, tsYdata, *rankCut)
			c += 1
		}
	}
	wg.Wait()
	return err
}

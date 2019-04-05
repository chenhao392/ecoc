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
	"strconv"
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
	var reg *bool = flag.Bool("r", false, "regularize CCA, default false")
	kSet := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	sigmaFctsSet := []float64{0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5}
	nFold := 5
	//var inThreads *int = flag.Int("p", 1, "number of threads")
	flag.Parse()
	rand.Seed(1)
	runtime.GOMAXPROCS(*inThreads)
	//read data
	tsXdata, _, _ := readFile(*tsX, false)
	tsYdata, _, _ := readFile(*tsY, false)
	trXdata, _, _ := readFile(*trX, false)
	trYdata, _, _ := readFile(*trY, false)
	fmt.Println(trYdata)
	//vars
	nTr, nFea := trXdata.Caps()
	nTs, _ := tsXdata.Caps()
	_, nLabel := trYdata.Caps()
	nRowTsY, _ := tsYdata.Caps()
	//CCA dims
	minDims := int(math.Min(float64(nFea), float64(nLabel)))
	//nComps := make([]int, minDims)
	//for i := 0; i < len(nComps); i++ {
	//	nComps[i] = i
	//}
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
	//pass Liblin
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
		Bsub := mat64.DenseCopyOf(B.Slice(0, nLabel, 0, kSet[k]))
		for s := 0; s < len(sigmaFctsSet); s++ {
			//go function
			//kSet[k]
			//sigmaFctsSet[s]
			go single_IOC_MFADecoding_and_result(nTs, kSet[k], c, tsY_Prob, tsY_C, sigma, Bsub, sigmaFctsSet[s], nLabel, sumRes, macroF1, tsYdata, *resFolder)
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

func single_IOC_MFADecoding_and_result(nTs int, k int, c int, tsY_Prob *mat64.Dense, tsY_C *mat64.Dense, sigma *mat64.Dense, Bsub *mat64.Dense, sigmaFcts float64, nLabel int, sumRes *mat64.Dense, macroF1 *mat64.Dense, tsYdata *mat64.Dense, resFolder string) {
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
	sFctStr := strconv.FormatFloat(sigmaFcts, 'f', 3, 64)
	kStr := strconv.FormatInt(int64(k), 16)
	oFile := "./" + resFolder + "/k" + kStr + "sFct" + sFctStr + ".txt"
	writeFile(oFile, tsYhat)
	//F1 score
	mutex.Lock()
	sum := 0.0
	for i := 0; i < nLabel; i++ {
		f1 := computeF1_2(tsYdata.ColView(i), tsYhat.ColView(i), 0.001)
		sumRes.Set(c, i, f1)
		sum += f1
		//fmt.Println(f1)
	}
	macroF1.Set(c, 0, float64(k))
	macroF1.Set(c, 1, sigmaFcts)
	macroF1.Set(c, 2, sum/float64(nLabel))
	mutex.Unlock()
}

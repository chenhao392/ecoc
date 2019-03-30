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
	"os"
	//"unsafe"
)

func main() {
	//input argv
	var tsX *string = flag.String("tsX", "data/tsX.10.txt", "testFeatureSet")
	var tsY *string = flag.String("tsY", "data/tsY.10.txt", "testLabelSet")
	var trX *string = flag.String("trX", "data/trX.10.txt", "trainFeatureSet")
	var trY *string = flag.String("trY", "data/trY.10.txt", "trainLabelSet")
	k := 4
	sigmaFcts := 0.5
	nFold := 5
	//var inThreads *int = flag.Int("p", 1, "number of threads")
	flag.Parse()
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
	//minDims := int(math.Min(float64(nFea), float64(nLabel)))
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
	var cca stat.CC
	err := cca.CanonicalCorrelations(trXdataB, trYdata, nil)
	if err != nil {
		log.Fatal(err)
	}
	B := cca.Right(nil, false)

	//skip as B is not the same with matlab code for debug
	//_, B = ccaProjectTwoMatrix(trXdataB, trYdata)
	//B = ccaProject(trXdataB, trYdata)
	fmt.Println("pass step 2 cca coding\n")
	//CCA code
	trY_Cdata := mat64.NewDense(0, 0, nil)
	trY_Cdata.Mul(trYdata, B)
	//decoding with regression
	tsY_C := mat64.NewDense(nRowTsY, nLabel, nil)
	sigma := mat64.NewDense(1, nLabel, nil)
	for i := 0; i < nLabel; i++ {
		beta, lamda, optMSE := adaptiveTrainRLS_Regress_CG(trXdataB, trY_Cdata.ColView(i), nFold, nFea, nTr)
		sigma.Set(0, i, math.Sqrt(optMSE))
		//bias term for tsXdata added before
		element := mat64.NewDense(0, 0, nil)
		element.Mul(tsXdataB, beta)
		for j := 0; j < nTs; j++ {
			tsY_C.Set(j, i, element.At(j, 0))
		}
		fmt.Println(lamda, optMSE)
	}
	fmt.Println("pass step 3 cg decoding\n")
	Bsub := mat64.DenseCopyOf(B.Slice(0, nLabel, 0, k))
	tsYhat := mat64.NewDense(nRowTsY, nLabel, nil)
	for i := 0; i < nTs; i++ {
		//the doc seems to be old, (0,x] seems to be correct
		//dim checked to be correct
		tsY_Prob_slice := tsY_Prob.Slice(i, i+1, 0, nLabel)
		tsY_C_slice := tsY_C.Slice(i, i+1, 0, k)
		arr := IOC_MFADecoding(nRowTsY, mat64.DenseCopyOf(tsY_Prob_slice), mat64.DenseCopyOf(tsY_C_slice), sigma, Bsub, k, sigmaFcts, nLabel)
		tsYhat.SetRow(i, arr)
	}
	//F1 score
	for i := 0; i < nLabel; i++ {
		f1 := computeF1_2(tsYdata.ColView(i), tsYhat.ColView(i))
		fmt.Println(f1)
	}
	os.Exit(0)
}

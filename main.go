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
	//"os"
	//"strconv"
	//"strings"
	//"unsafe"
)

func main() {
	//input argv
	var tsX *string = flag.String("tsX", "data/tsX.txt", "testFeatureSet")
	var tsY *string = flag.String("tsY", "data/tsY.txt", "testLabelSet")
	var trX *string = flag.String("trX", "data/trX.txt", "trainFeatureSet")
	var trY *string = flag.String("trY", "data/trY.txt", "trainLabelSet")
	//var inThreads *int = flag.Int("p", 1, "number of threads")
	flag.Parse()
	//read data
	tsXdata, _, _ := readFile(*tsX, false)
	tsYdata, _, _ := readFile(*tsY, false)
	trXdata, _, _ := readFile(*trX, false)
	trYdata, _, _ := readFile(*trY, false)
	//vars
	nTr, nFea := trXdata.Caps()
	nTs, _ := tsXdata.Caps()
	//_, nFea := trXdata.Caps()
	_, nLabel := trYdata.Caps()
	//lamda
	//sigmaFcts := [...]float64{0.5, 1, 2}
	//CCA dims
	minDims := int(math.Min(float64(nFea), float64(nLabel)))
	nComps := make([]int, minDims)
	for i := 0; i < len(nComps); i++ {
		nComps[i] = i
	}
	//tsY_prob
	nRowTsY, _ := tsYdata.Caps()
	tsY_prob := mat64.NewDense(nRowTsY, nLabel, nil)
	//tsY_prob := [][]float64{}
	//adding bias term for tsXData
	ones := make([]float64, nTs)
	for i := range ones {
		ones[i] = 1
	}
	tsXdata = colStack(tsXdata, ones)
	//for i := 0; i < nLabel; i++ {
	for i := 0; i < 1; i++ {
		wMat, _, _ := adaptiveTrainLGR_Liblin(trXdata, trYdata.ColView(i), 5, nFea)
		element := mat64.NewDense(0, 0, nil)
		element.Mul(tsXdata, wMat)
		for j := 0; j < nTs; j++ {
			value := 1.0 / (1 + math.Exp(-1*element.At(j, 0)))
			tsY_prob.Set(j, i, value)
		}
	}
	//pass Liblin
	fmt.Println("pass step 1 coding\n")
	//adding bias term for trXdata
	ones = make([]float64, nTr)
	for i := range ones {
		ones[i] = 1
	}
	trXdata = colStack(trXdata, ones)

	// Calculate the canonical correlations.
	var cca stat.CC
	//a, b := trXdata.Caps()
	//c, d := trYdata.Caps()
	//fmt.Println(a, b, c, d)
	err := cca.CanonicalCorrelations(trXdata, trYdata, nil)
	if err != nil {
		log.Fatal(err)
	}
	B := cca.Right(nil, true)
	fmt.Println("pass step 2 cca coding\n")
	//fmt.Printf("\n\nlabel projection = %.4f", mat64.Formatted(B.View(0, 0, nLabel-1, nLabel-1), mat64.Prefix("         ")))
	//CCA code
	trY_Cdata := mat64.NewDense(0, 0, nil)
	trY_Cdata.Mul(trYdata, B)
	//decoding with regression
	tsY_C := mat64.NewDense(nRowTsY, nLabel, nil)
	sigma := mat64.NewDense(1, nLabel, nil)
	//for i := 0; i < nLabel; i++ {
	for i := 0; i < 1; i++ {
		beta, lamda, optMSE := adaptiveTrainRLS_Regress_CG(trXdata, trY_Cdata.ColView(i), 5, nFea, nTr)
		sigma.Set(0, i, math.Sqrt(lamda))
		//bias term for tsXdata added before
		element := mat64.NewDense(0, 0, nil)
		element.Mul(tsXdata, beta)
		for j := 0; j < nTs; j++ {
			tsY_C.Set(j, i, element.At(j, 0))
		}
		fmt.Println(lamda, optMSE)
	}
	fmt.Println("pass step 3 cg decoding\n")
	//k=5,sigmaFcts 0.5
	k := 5
	sigmaFcts := 0.5
	tsYhat := mat64.NewDense(nRowTsY, nLabel, nil)
	for i := 0; i < 1; i++ {
		arr := IOC_MFADecoding(tsY_Prob.ColView(i), tsY_C.ColView(i), sigma, B, k, sigmaFcts, nLabel)
		//tsYhat.SetCol()
		fmt.Println(arr[0])
	}
}

package main

//#include <stdlib.h>
//import "C"
import (
	//"bufio"
	"flag"
	"fmt"
	"github.com/gonum/matrix/mat64"
	//"github.com/gonum/stat"
	//"log"
	"math"
	"os"
	//"strconv"
	//"strings"
	//"unsafe"
)

func main() {
	//input argv
	var tsX *string = flag.String("tsX", "data/tsX.10.txt", "testFeatureSet")
	var tsY *string = flag.String("tsY", "data/tsY.10.txt", "testLabelSet")
	var trX *string = flag.String("trX", "data/trX.10.txt", "trainFeatureSet")
	var trY *string = flag.String("trY", "data/trY.10.txt", "trainLabelSet")
	k := 5
	sigmaFcts := 0.5
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
	tsY_Prob := mat64.NewDense(nRowTsY, nLabel, nil)
	//tsY_prob := [][]float64{}
	//adding bias term for tsXData
	ones := make([]float64, nTs)
	for i := range ones {
		ones[i] = 1
	}
	tsXdataB := colStack(tsXdata, ones)
	//for i := 0; i < nLabel; i++ {
	for i := 0; i < 0; i++ {
		wMat, _, _ := adaptiveTrainLGR_Liblin(trXdata, trYdata.ColView(i), 5, nFea)
		//fmt.Println(wMat)
		element := mat64.NewDense(0, 0, nil)
		element.Mul(tsXdataB, wMat)
		for j := 0; j < nTs; j++ {
			//the -1*element.At() is changed to 1*element.At() in this implementation
			//the sign was flipped in this golang library, as manually checked
			value := 1.0 / (1 + math.Exp(1*element.At(j, 0)))
			//fmt.Println(element.At(j, 0), value)
			tsY_Prob.Set(j, i, value)
		}
	}
	//a, b := tsY_Prob.Caps()
	//fmt.Println(a, b)
	//fmt.Println(tsY_Prob)
	tsY_Prob, _, _ = readFile("tsY_probs.txt", false)
	//for i := 0; i < 10; i++ {
	//	fmt.Println(tsY_Prob.RawRowView(i))
	//}
	//os.Exit(0)
	//pass Liblin
	fmt.Println("pass step 1 coding\n")
	//adding bias term for trXdata
	ones = make([]float64, nTr)
	for i := range ones {
		ones[i] = 1
	}
	trXdataB := colStack(trXdata, ones)
	//for i := 0; i < 10; i++ {
	//	fmt.Println(trXdata.RawRowView(i))
	//}
	//os.Exit(0)
	// Calculate the canonical correlations.
	//var cca stat.CC
	//a, b := trXdata.Caps()
	//c, d := trYdata.Caps()
	//fmt.Println(a, b, c, d)
	//err := cca.CanonicalCorrelations(trXdataB, trYdata, nil)
	//err := cca.CanonicalCorrelations(mat64.DenseCopyOf(trXdata.T()), mat64.DenseCopyOf(trYdata.T()), nil)
	//if err != nil {
	//	log.Fatal(err)
	//}
	//B := cca.Right(nil, false)
	//A := cca.Right(nil, true)
	//C := cca.Corrs(nil)

	//skip as B is not the same with matlab code for debug
	_, B := ccaProjectTwoMatrix(trXdataB, trYdata)
	//B, _, _ = readFile("B.txt", false)
	fmt.Println("pass step 2 cca coding\n")
	//fmt.Println(B.At(0, 0))
	for i := 0; i < 10; i++ {
		fmt.Println(B.RawRowView(i))
	}
	//fmt.Println(B)
	//fmt.Println(B.T())
	//fmt.Println(A)
	//fmt.Println(C)
	os.Exit(0)
	//fmt.Printf("\n\nlabel projection = %.4f", mat64.Formatted(B.View(0, 0, nLabel-1, nLabel-1), mat64.Prefix("         ")))
	//CCA code
	trY_Cdata := mat64.NewDense(0, 0, nil)
	trY_Cdata.Mul(trYdata, B)
	//for i := 0; i < 10; i++ {
	//	fmt.Println(trY_Cdata.RawRowView(i))
	//}
	//decoding with regression
	tsY_C := mat64.NewDense(nRowTsY, nLabel, nil)
	sigma := mat64.NewDense(1, nLabel, nil)
	for i := 0; i < nLabel; i++ {
		//for i := 0; i < 0; i++ {
		beta, lamda, optMSE := adaptiveTrainRLS_Regress_CG(trXdataB, trY_Cdata.ColView(i), k, nFea, nTr)
		sigma.Set(0, i, math.Sqrt(lamda))
		//bias term for tsXdata added before
		element := mat64.NewDense(0, 0, nil)
		element.Mul(tsXdataB, beta)
		for j := 0; j < nTs; j++ {
			tsY_C.Set(j, i, element.At(j, 0))
		}
		fmt.Println(lamda, optMSE)
	}
	fmt.Println("pass step 3 cg decoding\n")
	//fmt.Printf("tsY_C=%.4f", mat.Formatted(tsY_C, mat.Prefix("     ")))
	//for i := 0; i < 10; i++ {
	//	fmt.Println(tsY_C.RawRowView(i))
	//}
	//fmt.Println(sigma)
	//tsY_C, _, _ = readFile("tsY_C.txt", false)
	//sigma, _, _ = readFile("sigma.txt", false)
	//fmt.Println(sigma)
	//for i := 0; i < 10; i++ {
	//	fmt.Println(tsY_C.RawRowView(i))
	//}
	//os.Exit(0)
	//k=5,sigmaFcts 0.5
	//nR, _ := B.Caps()
	Bsub := mat64.DenseCopyOf(B.Slice(0, nLabel, 0, k))
	//tsYhat := mat64.NewDense(nRowTsY, nLabel, nil)
	for i := 0; i < 20; i++ {
		//a, b := tsY_Prob.Caps()
		//fmt.Println(a, b, nLabel)
		//the doc seems to be old, (0,x] seems to be correct
		tsY_Prob_slice := tsY_Prob.Slice(i, i+1, 0, nLabel)
		tsY_C_slice := tsY_C.Slice(i, i+1, 0, k)
		arr := IOC_MFADecoding(nRowTsY, mat64.DenseCopyOf(tsY_Prob_slice), mat64.DenseCopyOf(tsY_C_slice), sigma, Bsub, k, sigmaFcts, nLabel)
		//arr := IOC_MFADecoding(tsY_Prob.ColView(i), tsY_C.ColView(i), sigma, B, k, sigmaFcts, nLabel)
		//tsYhat.SetCol()
		//fmt.Printf("%.3f", arr)
		fmt.Println(arr)
		fmt.Println("")
	}
	os.Exit(0)
}

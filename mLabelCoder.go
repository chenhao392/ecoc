package main

//#include <stdlib.h>
//import "C"
import (
	"bufio"
	"flag"
	"fmt"
	linear "github.com/chenhao392/lineargo"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	//"unsafe"
)

//for cross validation
type cvFold struct {
	X *mat64.Dense
	Y *mat64.Dense
}

func (f *cvFold) setXY(pos []int, neg []int, matX *mat64.Dense, vecY *mat64.Vector) {
	_, nColX := matX.Caps()
	//nColY := vecY.Len()
	nRowPos := len(pos)
	nRowNeg := len(neg)
	//fmt.Println(nRowPos, nRowNeg)
	f.X = mat64.NewDense(nRowPos+nRowNeg, nColX, nil)
	f.Y = mat64.NewDense(nRowPos+nRowNeg, 1, nil)
	for i := 0; i < nRowPos; i++ {
		//fmt.Println(i)
		f.X.SetRow(i, matX.RawRowView(pos[i]))
		f.Y.Set(i, 0, vecY.At(pos[i], 0))
	}
	for i := nRowPos; i < nRowPos+nRowNeg; i++ {
		f.X.SetRow(i, matX.RawRowView(neg[i-nRowPos]))
		f.Y.Set(i, 0, vecY.At(neg[i-nRowPos], 0))
	}
}

func minIdx(inArray []float64) (idx int) {
	m := 999999999.9
	for i, e := range inArray {
		if e < m {
			m = e
			idx = i
		}
	}
	return idx
}

func computeF1(X *mat64.Dense, Y *mat64.Dense, beta *mat64.Dense) (F1 float64) {
	//X = mat64.DenseCopyOf(X.T())
	n, _ := X.Caps()
	onesSlice := make([]float64, n)
	for i := range onesSlice {
		onesSlice[i] = 1
	}
	//ones := mat64.NewDense(1, n, onesSlice)
	//X2 := mat64.NewDense(0, 0, nil)
	//X2.Stack(ones, X)
	//X2 = mat64.DenseCopyOf(X2.T())
	X2 := colStack(X, onesSlice)
	Yh := mat64.NewDense(0, 0, nil)
	Yh.Mul(X2, beta)
	var tp int
	var fp int
	var fn int
	var tn int
	for i := 0; i < n; i++ {
		y := Y.At(i, 0)
		yh := Yh.At(i, 0)
		if y > 0 && yh > 0 {
			tp += 1
		} else if y <= 0 && yh > 0 {
			fp += 1
		} else if y > 0 && yh <= 0 {
			fn += 1
		} else if y <= 0 && yh <= 0 {
			tn += 1
		}
	}
	var prec float64
	var rec float64
	//P and R
	if tp+fn == 0 {
		prec = 0.5
	} else {
		prec = float64(tp) / (float64(tp) + float64(fp))
	}
	if tp+fn == 0 {
		rec = 0.5
	} else {
		rec = float64(tp) / (float64(tp) + float64(fn))
	}
	//F1
	if prec+rec == 0 {
		F1 = 0
	} else {
		F1 = 2 * float64(prec) * float64(rec) / (float64(prec) + float64(rec))
	}
	return F1
}

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
	nTr, _ := trXdata.Caps()
	nTs, _ := tsXdata.Caps()
	_, nFea := trXdata.Caps()
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
	//fmt.Printf("\n\nlabel projection = %.4f", mat64.Formatted(B.View(0, 0, nLabel-1, nLabel-1), mat64.Prefix("         ")))
}
func adaptiveTrainLGR_Liblin(X *mat64.Dense, Y *mat64.Vector, nFold int, nFeature int) (wMat *mat64.Dense, regulator float64, errFinal float64) {
	//prior := make([]float64, nFeature)
	lamda := []float64{0.0001, 0.001, 0.01, 1, 10}
	err := []float64{0, 0, 0, 0, 0}
	//lamda := []float64{0.0001, 0.001}
	//err := []float64{0, 0}
	//projMtx := mat64.NewDense(nFeature+1, nFeature+1, nil)
	//for i := 0; i <= nFeature; i++ {
	//	projMtx.Set(i, i, 1)
	//}
	nY := Y.Len()
	//index positive and megetive examples
	posIndex := make([]int, 0)
	negIndex := make([]int, 0)
	for i := 0; i < nY; i++ {
		if Y.At(i, 0) == 1 {
			posIndex = append(posIndex, i)
		} else {
			negIndex = append(negIndex, i)
		}
	}
	nPos := len(posIndex)
	nNeg := len(negIndex)
	//random permutation skipped for now
	//nFold == 1 skippped for noe
	if nFold == 1 {
		panic("nFold == 1 not implemetned yet.")
	}
	trainFold := make([]cvFold, nFold)
	testFold := make([]cvFold, nFold)
	for i := 0; i < nFold; i++ {
		posTrain := make([]int, 0)
		negTrain := make([]int, 0)
		posTest := make([]int, 0)
		negTest := make([]int, 0)
		posTestMap := map[int]int{}
		negTestMap := map[int]int{}
		//test set and map
		//a := i * nPos / nFold
		//b := (i+1)*nPos/nFold - 1
		//fmt.Println(a, b)
		for j := i * nPos / nFold; j < (i+1)*nPos/nFold-1; j++ {
			posTest = append(posTest, posIndex[j])
			posTestMap[j] = posIndex[j]
		}
		for j := i * nNeg / nFold; j < (i+1)*nNeg/nFold-1; j++ {
			negTest = append(negTest, negIndex[j])
			negTestMap[j] = negIndex[j]
		}
		//the rest is for training
		for j := 0; j < nPos; j++ {
			_, exist := posTestMap[j]
			if !exist {
				posTrain = append(posTrain, posIndex[j])
			}
		}
		for j := 0; j < nNeg; j++ {
			_, exist := negTestMap[j]
			if !exist {
				negTrain = append(negTrain, negIndex[j])
			}
		}
		trainFold[i].setXY(posTrain, negTrain, X, Y)
		testFold[i].setXY(posTest, negTest, X, Y)
	}
	//total error with different lamda
	for i := 0; i < len(lamda); i++ {
		for j := 0; j < nFold; j++ {
			//sensitiveness: the epsilon in loss function of SVR is set to 0.1 as the default value, not mentioned in matlab code
			//doc in the lineargo lib: If you do not want to change penalty for any of the classes, just set classWeights to nil.
			//So yes for this implementation, as the penalty not mentioned in matlab code
			//X: features, Y:label vector, bias,solver,cost,sensitiveness,stop,class_pelnalty
			LRmodel := linear.Train(trainFold[j].X, trainFold[j].Y, 1.0, 0, 1.0/lamda[i], 0.1, 0.0001, nil)
			w := LRmodel.W()
			lastW := []float64{Pop(&w)}
			w = append(lastW, w...)
			wMat := mat64.NewDense(len(w), 1, w)
			e := 1.0 - computeF1(testFold[j].X, testFold[j].Y, wMat)
			err[i] = err[i] + e
		}
	}
	//min error index
	idx := minIdx(err)
	regulator = 1.0 / lamda[idx]
	fmt.Println(idx, lamda[idx], regulator)
	Ymat := mat64.NewDense(Y.Len(), 1, nil)
	for i := 0; i < Y.Len(); i++ {
		Ymat.Set(i, 0, Y.At(i, 0))
	}
	LRmodel := linear.Train(X, Ymat, 1.0, 0, regulator, 0.1, 0.0001, nil)
	w := LRmodel.W()
	lastW := []float64{Pop(&w)}
	w = append(lastW, w...)
	wMat = mat64.NewDense(len(w), 1, w)
	//defer C.free(unsafe.Pointer(LRmodel))
	//nr_feature := LRmodel.Nfeature() + 1
	//w := []float64{-1}
	//w = append(w, LRmodel.W()...)
	errFinal = err[idx]
	return wMat, regulator, errFinal
}
func readFile(inFile string, rowName bool) (dataR *mat64.Dense, rName []string, err error) {
	//init
	lc, cc, _ := lcCount(inFile)
	if rowName {
		cc -= 1
	}
	data := mat64.NewDense(lc, cc, nil)
	rName = make([]string, 0)

	//file
	file, err := os.Open(inFile)
	if err != nil {
		return
	}
	defer file.Close()

	//load
	br := bufio.NewReaderSize(file, 32768000)
	r := 0
	for {
		line, isPrefix, err1 := br.ReadLine()
		if err1 != nil {
			break
		}
		if isPrefix {
			return
		}

		str := string(line)
		elements := strings.Split(str, "\t")
		if rowName {
			value := Shift(&elements)
			rName = append(rName, value)
		}
		for c, i := range elements {
			j, _ := strconv.ParseFloat(i, 64)
			data.Set(r, c, j)
		}
		r++
	}
	return data, rName, nil
}

//line count(nRow) and column count(nCol) for a tab separeted txt
func lcCount(filename string) (lc int, cc int, err error) {
	lc = 0
	cc = 0
	touch := true

	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	//load
	br := bufio.NewReaderSize(file, 32768000)
	for {
		line, isPrefix, err1 := br.ReadLine()
		if err1 != nil {
			break
		}
		if isPrefix {
			return
		}

		if touch {
			cc = strings.Count(string(line), "\t")
			cc += 1
			touch = false
		}
		lc++
	}
	return lc, cc, nil
}

func Shift(pToSlice *[]string) string {
	sValue := (*pToSlice)[0]
	*pToSlice = (*pToSlice)[1:len(*pToSlice)]
	return sValue
}

func Pop(pToSlice *[]float64) float64 {
	pValue := (*pToSlice)[len(*pToSlice)-1]
	*pToSlice = (*pToSlice)[0 : len(*pToSlice)-1]
	return pValue
}

//only row stacking available in mat64, the matlab code use "cbind"
func colStack(X *mat64.Dense, oneSlice []float64) (X2 *mat64.Dense) {
	X = mat64.DenseCopyOf(X.T())
	_, n := X.Caps()
	ones := mat64.NewDense(1, n, oneSlice)
	X2 = mat64.NewDense(0, 0, nil)
	X2.Stack(ones, X)
	X2 = mat64.DenseCopyOf(X2.T())
	return X2
}

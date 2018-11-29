package main

import (
	"fmt"
	linear "github.com/chenhao392/lineargo"
	"github.com/gonum/matrix/mat64"
)

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

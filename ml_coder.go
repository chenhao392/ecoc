package main

import (
	"fmt"
	linear "github.com/chenhao392/lineargo"
	//"github.com/gonum/gonum/stat/distuv"
	"github.com/gonum/matrix/mat64"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
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
func adaptiveTrainRLS_Regress_CG(X *mat64.Dense, Y *mat64.Vector, nFold int, nFeature int, nTr int) (beta *mat64.Dense, regulazor float64, optMSE float64) {
	//prior := mat64.NewDense(0, nFeature+1, nil)
	lamda := []float64{0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000, 10000}
	err := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}
	projMtx := mat64.NewDense(nFeature+1, nFeature+1, nil)
	projMtx.Set(0, 0, 1)
	//random permute index skipped for now
	//cv folds data
	trainFold := make([]cvFold, nFold)
	testFold := make([]cvFold, nFold)
	for i := 0; i < nFold; i++ {
		cvTrain := make([]int, 0)
		cvTest := make([]int, 0)
		cvTestMap := map[int]int{}
		for j := i * nTr / nFold; j < (i+1)*nTr/nFold-1; j++ {
			cvTest = append(cvTest, j)
			cvTestMap[j] = j
		}
		//the rest is for training
		for j := 0; j < nTr; j++ {
			_, exist := cvTestMap[j]
			if !exist {
				cvTrain = append(cvTrain, j)
			}
		}
		trainFold[i].setXYinDecoding(cvTrain, X, Y)
		testFold[i].setXYinDecoding(cvTest, X, Y)
	}
	//estimating error /weights
	for j := 0; j < len(lamda); j++ {
		for i := 0; i < nFold; i++ {
			//weights finalized for one lamda and one fold
			weights := TrainRLS_Regress_CG(trainFold[i].X, trainFold[i].Y, projMtx, lamda[j])
			//var testError float64 0
			//testErr := make([]float64, 0)
			term1 := mat64.NewDense(0, 0, nil)
			term2 := mat64.NewDense(0, 0, nil)
			//term3 := mat64.NewDense(0, 0, nil)
			//trXdata and tsXdata are "cbinded" previously in main
			term1.Mul(testFold[i].X, weights)
			term2.Sub(term1, testFold[i].Y)
			var sum float64 = 0
			var mean float64
			r, c := term2.Caps()
			for m := 0; m < r; m++ {
				for n := 0; n < c; n++ {
					sum += term2.At(m, n) * term2.At(m, n)
				}
			}
			mean = sum / float64(r*c)
			err[j] = err[j] + mean
			fmt.Println(err[j], mean)
		}
		err[j] = err[j] / float64(nFold)
	}
	//min error index
	idx := minIdx(err)
	optMSE = err[idx]
	regulazor = lamda[idx]
	//beta is weights
	//convert Y yo Ymat
	nY := Y.Len()
	Ymat := mat64.NewDense(nY, 1, nil)
	for i := 0; i < nY; i++ {
		Ymat.Set(i, 0, Y.At(i, 0))
	}
	beta = TrainRLS_Regress_CG(X, Ymat, projMtx, regulazor)
	return beta, regulazor, optMSE
	//beta=
}

func MulEleByFloat64(value float64, M *mat64.Dense) (M2 *mat64.Dense) {
	r, c := M.Caps()
	M2 = mat64.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			M2.Set(i, j, M.At(i, j)*value)
		}
	}
	return M2
}
func gradientCal(lamda float64, projMtx *mat64.Dense, weights *mat64.Dense, X *mat64.Dense, Y *mat64.Dense, products *mat64.Dense) (gradient *mat64.Dense) {
	term1 := mat64.NewDense(0, 0, nil)
	term2 := mat64.NewDense(0, 0, nil)
	term3 := mat64.NewDense(0, 0, nil)
	term4 := mat64.NewDense(0, 0, nil)
	term1.Sub(Y, products)
	Xt := mat64.DenseCopyOf(X.T())
	term2.Mul(Xt, term1)
	term3.Mul(projMtx, weights)
	//term4.MulElem(-1*lamda, term3)
	term4 = MulEleByFloat64(-1*lamda, term3)
	gradient = mat64.NewDense(0, 0, nil)
	gradient.Add(term4, term2)
	return gradient
}
func maxDiffCal(product *mat64.Dense, preProduct *mat64.Dense, n int) (maxDiff float64) {
	maxDiff = 0
	for i := 0; i < n; i++ {
		value := math.Abs(product.At(i, 0) - preProduct.At(i, 0))
		if maxDiff < value {
			maxDiff = value
		}
	}
	return maxDiff
}
func cgCal(gradient *mat64.Dense, preGradient *mat64.Dense, cg *mat64.Dense) (cg2 *mat64.Dense) {
	term1 := mat64.NewDense(0, 0, nil)
	term2 := mat64.NewDense(0, 0, nil)
	term3 := mat64.NewDense(0, 0, nil)
	term4 := mat64.NewDense(0, 0, nil)
	term5 := mat64.NewDense(0, 0, nil)
	beta := mat64.NewDense(0, 0, nil)
	term1.Sub(gradient, preGradient)
	Gt := mat64.DenseCopyOf(gradient.T())
	CGt := mat64.DenseCopyOf(cg.T())
	term2.Mul(CGt, term1)
	term3.Mul(Gt, term1)
	//term4 := mat64.DenseCopyOf(term3.Inverse())
	term4.Inverse(term3)
	//right division in matlab. A/B = A*inv(B)
	beta.Mul(term2, term4)
	term5.Mul(cg, beta)
	cg2 = mat64.NewDense(0, 0, nil)
	cg2.Sub(gradient, term5)
	return cg
}
func stepCal(gradient *mat64.Dense, cg *mat64.Dense, lamda float64, X *mat64.Dense) (step float64) {
	term1 := mat64.NewDense(0, 0, nil)
	term2 := mat64.NewDense(0, 0, nil)
	term3 := mat64.NewDense(0, 0, nil)
	//term4 := mat64.NewDense(0, 0, nil)
	term1.Mul(X, cg)
	var sum float64 = 0
	r, c := term1.Caps()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += term1.At(i, j) * term1.At(i, j)
		}
	}
	CGt := mat64.DenseCopyOf(cg.T())
	Gt := mat64.DenseCopyOf(gradient.T())
	term2.Mul(CGt, cg)
	term3.Mul(Gt, cg)
	step = term3.At(0, 0) / (lamda*term2.At(0, 0) + sum)
	return step
}
func deltaLossCal(Y *mat64.Dense, products *mat64.Dense, lamda float64, projMtx *mat64.Dense, weights *mat64.Dense, preProducts *mat64.Dense, preWeights *mat64.Dense) (deltaLoss float64) {
	term1 := mat64.NewDense(0, 0, nil)
	term2 := mat64.NewDense(0, 0, nil)
	//term3 := mat64.NewDense(0, 0, nil)
	//term4 := mat64.NewDense(0, 0, nil)
	term1.Mul(projMtx, preWeights)
	term2.Mul(projMtx, weights)
	//math.Pow(mat64.Norm(term1,2),2)*lamda
	//math.Pow(mat64.Norm(term2,2),2)*lamda
	var preSum float64 = 0
	r, c := Y.Caps()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			preSum += math.Pow(Y.At(i, j)-preProducts.At(i, j), 2)
		}
	}
	var sum float64 = 0
	r, c = Y.Caps()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += math.Pow(Y.At(i, j)-products.At(i, j), 2)
		}
	}
	deltaLoss = sum + math.Pow(mat64.Norm(term2, 2), 2)*lamda - preSum - math.Pow(mat64.Norm(term1, 2), 2)*lamda
	return deltaLoss
}

//func TrainRLS_Regress_CG(trFoldX *mat64.Dense, trFoldY *mat64.Vector, lamda float64) (weights *mat64.Dense) {
func TrainRLS_Regress_CG(trFoldX *mat64.Dense, trFoldY *mat64.Dense, projMtx *mat64.Dense, lamda float64) (weights *mat64.Dense) {
	n, p := trFoldX.Caps()
	//nY := trFoldY.Len()
	//trFoldYmat = mat64.NewDense(nY, 1, nil)
	//for i := 0; I < nY; i++ {
	//	trFoldYmat.Set(i, 0, trFoldY.At(i, 1))
	//}
	//weight
	//weights := make([]float64, 0)
	//preWeights := make([]float64, 0)
	weights = mat64.NewDense(p, 1, nil)
	preWeights := mat64.NewDense(p, 1, nil)
	var UformDist = distuv.Uniform{Min: -0.0001, Max: 0.0001}
	for k := 0; k < p; k++ {
		value := UformDist.Rand()
		//weights = append(weights, Uniform.Rand())
		weights.Set(k, 0, value)
		preWeights.Set(k, 0, value)
	}
	//products and gradient
	tmpData := make([]float64, 0)
	for k := 0; k < n; k++ {
		tmpData = append(tmpData, -1)
	}
	preProducts := mat64.NewDense(n, 1, tmpData)
	preGradient := mat64.NewDense(p, 1, nil)
	//pre calculation
	products := mat64.NewDense(0, 0, nil)
	products.Mul(trFoldX, weights)
	//W0 is all zeros for this app
	//gradient := gradientCal(lamda, projMtx, weights, trFoldX, trFoldYmat, products)
	gradient := gradientCal(lamda, projMtx, weights, trFoldX, trFoldY, products)
	iter := 0
	maxIter := 6000
	maxDiff := maxDiffCal(products, preProducts, n)
	cg := mat64.DenseCopyOf(gradient.View(0, 0, p, 1))
	//the while loop
	for maxDiff > 0.0000001 && iter < maxIter {
		iter++
		//conjugate gradient
		if iter > 1 {
			cg = cgCal(gradient, preGradient, cg)
		}
		//projection
		step := stepCal(gradient, cg, lamda, trFoldX)
		//copy(preProducts, products)
		//copy(preGradient, gradient)
		//copy(preWeights, weights)
		preProducts.Copy(products)
		preGradient.Copy(gradient)
		preWeights.Copy(weights)
		//update weight
		for k := 0; k < p; k++ {
			weights.Set(k, 0, preWeights.At(k, 0)+step*cg.At(k, 0))
		}
		products.Mul(trFoldX, weights)
		//gradient = gradientCal(lamda, projMtx, weights, trFoldX, trFoldYmat, products)
		//deltaLoss := deltaLossCal(trFoldYmat, products, lamda, projMtx, weights, preProducts)
		gradient = gradientCal(lamda, projMtx, weights, trFoldX, trFoldY, products)
		deltaLoss := deltaLossCal(trFoldY, products, lamda, projMtx, weights, preProducts, preWeights)

		for deltaLoss > 0.0000000001 {
			step = step / 10
			//update weight
			for k := 0; k < p; k++ {
				weights.Set(k, 1, preWeights.At(k, 0)+step*cg.At(k, 0))
			}
			products.Mul(trFoldX, weights)
			//gradient = gradientCal(lamda, projMtx, weights, trFoldX, trFoldYmat, products)
			gradient = gradientCal(lamda, projMtx, weights, trFoldX, trFoldY, products)
		}
		maxDiff = maxDiffCal(products, preProducts, n)
	}
	return weights
}

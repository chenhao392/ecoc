package main

import (
	"fmt"
	linear "github.com/chenhao392/lineargo"
	//"github.com/gonum/gonum/stat/distuv"
	"github.com/gonum/matrix/mat64"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"math/rand"
	//"os"
)

func adaptiveTrainLGR_Liblin(X *mat64.Dense, Y *mat64.Vector, nFold int, nFeature int) (wMat *mat64.Dense, regulator float64, errFinal float64) {
	//prior := make([]float64, nFeature)
	lamda := []float64{0.1, 1, 10}
	err := []float64{0, 0, 0}
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
	posCvSet := cvSplit(nPos, nFold)
	negCvSet := cvSplit(nNeg, nFold)
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
		//for j := i * nPos / nFold; j < (i+1)*nPos/nFold-1; j++ {
		for j := 0; j < len(posCvSet[i]); j++ {
			posTest = append(posTest, posIndex[posCvSet[i][j]])
			posTestMap[posCvSet[i][j]] = posIndex[posCvSet[i][j]]
		}
		for j := 0; j < len(negCvSet[i]); j++ {
			negTest = append(negTest, negIndex[negCvSet[i][j]])
			negTestMap[negCvSet[i][j]] = negIndex[negCvSet[i][j]]
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
		fmt.Println(posTest, posTrain)
		fmt.Println(negTest, negTrain)
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
			fmt.Println(i, j, e, err[i])
		}
		fmt.Println(i, err[i])
	}
	//min error index
	idx := minIdx(err)
	regulator = 1.0 / lamda[idx]
	fmt.Println("choose: ", idx, lamda[idx], regulator)
	Ymat := mat64.NewDense(Y.Len(), 1, nil)
	for i := 0; i < Y.Len(); i++ {
		Ymat.Set(i, 0, Y.At(i, 0))
	}
	LRmodel := linear.Train(X, Ymat, 1.0, 0, regulator, 0.1, 0.0001, nil)
	w := LRmodel.W()
	lastW := []float64{Pop(&w)}
	w = append(lastW, w...)
	//fmt.Println(w)
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
	lamda := []float64{0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000}
	err := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	//projMtx := mat64.NewDense(nFeature+1, nFeature+1, nil)
	//projMtx.Set(0, 0, 1)
	//idxPerm 0:nTr, value as random order nTr
	rand.Seed(1)
	idxPerm := rand.Perm(nTr)
	//cv folds data
	trainFold := make([]cvFold, nFold)
	testFold := make([]cvFold, nFold)
	for i := 0; i < nFold; i++ {
		cvTrain := make([]int, 0)
		cvTest := make([]int, 0)
		cvTestMap := map[int]int{}
		for j := i * nTr / nFold; j < (i+1)*nTr/nFold-1; j++ {
			cvTest = append(cvTest, idxPerm[j])
			cvTestMap[idxPerm[j]] = idxPerm[j]
		}
		//the rest is for training
		for j := 0; j < nTr; j++ {
			_, exist := cvTestMap[j]
			if !exist {
				cvTrain = append(cvTrain, j)
			}
		}
		//fmt.Println("fold", i, "train:", cvTrain)
		//fmt.Println("fold", i, "test:", cvTest)
		trainFold[i].setXYinDecoding(cvTrain, X, Y)
		testFold[i].setXYinDecoding(cvTest, X, Y)
	}
	//estimating error /weights
	for j := 0; j < len(lamda); j++ {
		for i := 0; i < nFold; i++ {
			//weights finalized for one lamda and one fold
			weights := TrainRLS_Regress_CG(trainFold[i].X, trainFold[i].Y, lamda[j])
			//var testError float64 0
			//testErr := make([]float64, 0)
			term1 := mat64.NewDense(0, 0, nil)
			term2 := mat64.NewDense(0, 0, nil)
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
		}
		err[j] = err[j] / float64(nFold)
		fmt.Println("error for", lamda[j], "is", err[j])

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
	beta = TrainRLS_Regress_CG(X, Ymat, regulazor)
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
func gradientCal(lamda float64, weights *mat64.Dense, X *mat64.Dense, Y *mat64.Dense, products *mat64.Dense) (gradient *mat64.Dense) {
	term1 := mat64.NewDense(0, 0, nil)
	term2 := mat64.NewDense(0, 0, nil)
	//term3 := mat64.NewDense(0, 0, nil)
	term4 := mat64.NewDense(0, 0, nil)
	//a, b := Y.Caps()
	//c, d := products.Caps()
	//fmt.Println(a, b, c, d)
	term1.Sub(Y, products)
	//Xt := mat64.DenseCopyOf(X.T())
	term2.Mul(X.T(), term1)
	//term3.Mul(projMtx, weights)
	//term4.MulElem(-1*lamda, term3)
	term4 = MulEleByFloat64(-1*lamda, weights)
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
	//Gt := mat64.DenseCopyOf(gradient.T())
	//CGt := mat64.DenseCopyOf(cg.T())
	term2.Mul(cg.T(), term1)
	term3.Mul(gradient.T(), term1)
	//term4 := mat64.DenseCopyOf(term3.Inverse())
	term4.Inverse(term2)
	//right division in matlab. A/B = A*inv(B)
	beta.Mul(term3, term4)
	term5.Mul(cg, beta)
	cg2 = mat64.NewDense(0, 0, nil)
	cg2.Sub(gradient, term5)
	return cg2
}
func stepCal(gradient *mat64.Dense, cg *mat64.Dense, lamda float64, X *mat64.Dense) (step float64) {
	term1 := mat64.NewDense(0, 0, nil)
	term2 := mat64.NewDense(0, 0, nil)
	term3 := mat64.NewDense(0, 0, nil)
	//term5 := mat64.NewDense(0, 0, nil)
	term1.Mul(X, cg)
	r, c := term1.Caps()
	//sum := make([]float64, c)
	sum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += term1.At(i, j) * term1.At(i, j)
		}
	}
	//CGt := mat64.DenseCopyOf(cg.T())
	//Gt := mat64.DenseCopyOf(gradient.T())
	term2.Mul(cg.T(), cg)
	//r, c = term2.Caps()
	term3.Mul(gradient.T(), cg)
	//term4 := mat64.NewDense(r, c, nil)
	//fmt.Println("term2", r, c)
	//r, c = cg.Caps()
	//fmt.Println("cg", r, c)
	//step = term3.At(0, 0) / (lamda*term2.At(0, 0) + sum)
	//for i := 0; i < r; i++ {
	//	for j := 0; j < c; j++ {
	//		term4.Set(i, j, term2.At(i, j)*lamda+sum)
	//	}
	//}
	//fmt.Println(term4)
	//term5.Inverse(term4)
	//step.Mul(term3, term5)
	//fmt.Println(term4)
	step = term3.At(0, 0) / (lamda*term2.At(0, 0) + sum)
	return step
}
func deltaLossCal(Y *mat64.Dense, products *mat64.Dense, lamda float64, weights *mat64.Dense, preProducts *mat64.Dense, preWeights *mat64.Dense) (deltaLoss float64) {
	//term1 := mat64.NewDense(0, 0, nil)
	//term2 := mat64.NewDense(0, 0, nil)
	//term3 := mat64.NewDense(0, 0, nil)
	//term4 := mat64.NewDense(0, 0, nil)
	//term1.Mul(projMtx, preWeights)
	//term2.Mul(projMtx, weights)
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
	deltaLoss = sum + math.Pow(mat64.Norm(weights, 2), 2)*lamda - preSum - math.Pow(mat64.Norm(preWeights, 2), 2)*lamda
	return deltaLoss
}

//func TrainRLS_Regress_CG(trFoldX *mat64.Dense, trFoldY *mat64.Vector, lamda float64) (weights *mat64.Dense) {
//func TrainRLS_Regress_CG(trFoldX *mat64.Dense, trFoldY *mat64.Dense, projMtx *mat64.Dense, lamda float64) (weights *mat64.Dense) {
func TrainRLS_Regress_CG(trFoldX *mat64.Dense, trFoldY *mat64.Dense, lamda float64) (weights *mat64.Dense) {
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
	gradient := gradientCal(lamda, weights, trFoldX, trFoldY, products)
	iter := 0
	maxIter := 6000
	maxDiff := maxDiffCal(products, preProducts, n)
	cg := mat64.DenseCopyOf(gradient.View(0, 0, p, 1))
	//a, b := cg.Caps()
	//c, d := gradient.Caps()
	//fmt.Println("cg and gradient", gradient.At(0, 0), cg.At(0, 0), a, b, c, d)
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
		//term1 := mat64.NewDense(0, 0, nil)
		//term1.Mul(step, cg)
		//update weight
		for k := 0; k < p; k++ {
			weights.Set(k, 0, preWeights.At(k, 0)+step*cg.At(k, 0))
		}
		products.Mul(trFoldX, weights)
		//gradient = gradientCal(lamda, projMtx, weights, trFoldX, trFoldYmat, products)
		//deltaLoss := deltaLossCal(trFoldYmat, products, lamda, projMtx, weights, preProducts)
		gradient = gradientCal(lamda, weights, trFoldX, trFoldY, products)
		deltaLoss := deltaLossCal(trFoldY, products, lamda, weights, preProducts, preWeights)

		for deltaLoss > 0.0000000001 {
			step = step / 10
			//step = MulEleByFloat64(0.1, step)
			//term2 := mat64.NewDense(0, 0, nil)
			//term2.Mul(step, cg)
			//update weight
			for k := 0; k < p; k++ {
				weights.Set(k, 0, preWeights.At(k, 0)+step*cg.At(k, 0))
			}
			products.Mul(trFoldX, weights)
			//gradient = gradientCal(lamda, projMtx, weights, trFoldX, trFoldYmat, products)
			gradient = gradientCal(lamda, weights, trFoldX, trFoldY, products)
		}
		maxDiff = maxDiffCal(products, preProducts, n)
	}
	if iter == maxIter {
		fmt.Println("reached max Iter for conjugate gradient.")
	}
	return weights
}

func IOC_MFADecoding(nRowTsY int, tsY_Prob *mat64.Dense, tsY_C *mat64.Dense, sigma *mat64.Dense, Bsub *mat64.Dense, k int, sigmaFcts float64, nLabel int) (tsYhatData []float64) {
	//Q
	//nRowTsY := tsY_prob.Caps()
	Q := mat64.NewDense(1, nLabel, nil)
	for i := 0; i < nLabel; i++ {
		Q.Set(0, i, tsY_Prob.At(0, i))
	}
	//sigma and B for top k elements
	//Bsub := mat64.NewDense(nLabel, k, nil)
	//fmt.Println(Bsub)
	sigmaSub := mat64.NewDense(1, k, nil)
	//transpose B as we don't have rawColView
	//B.T()
	for i := 0; i < k; i++ {
		//Bsub.SetCol(i, B.RawRowView(i))
		sigmaSub.Set(0, i, sigma.At(0, i)*sigmaFcts)
	}
	//converting data formats
	//decoding
	//iter := 0
	//random permutation for 0:nLabel, labels
	//rand.Seed(1)
	//idx 0:nLabel, value as random order nLabel index
	//idx := rand.Perm(nLabel)
	//idx := make([]int, nLabel)
	//for i := 0; i < nLabel; i++ {
	//	idx[i] = i
	//}
	//fmt.Println(idx)
	//ind
	ind := make([]int, nLabel)
	for i := 0; i < nLabel; i++ {
		ind[i] = 1
	}
	//init index
	i := 0
	//for i := 0; i < nLabel; i++ {
	for ind[i] > 0 {
		logPos := math.Log(tsY_Prob.At(0, i))
		logNeg := math.Log(1 - tsY_Prob.At(0, i))
		posFct := mat64.NewDense(1, k, nil)
		negFct := mat64.NewDense(1, k, nil)
		for j := 0; j < nLabel; j++ {
			//fmt.Println(j, i, Q)
			if j == i || Q.At(0, j) == 0 {
				continue
			}
			negFct = fOrderNegFctCal(negFct, tsY_C, Bsub, Q, j)
			//fmt.Println("j1:", j, negFct.At(0, 0))
			//second order, n is j2, golang is 0 based, sothat the for loop is diff on max
			for n := 0; n < j; n++ {
				if n == i || Q.At(0, n) == 0 {
					continue
				}
				negFct = sOrderNegFctCal(negFct, Bsub, Q, j, n)
				//fmt.Println(" j2:", n, negFct.At(0, 0))
			}
			//posFct
			posFct = posFctCal(posFct, Bsub, Q, i, j)
		}
		//fmt.Println(i, logPos, logNeg, negFct.At(0, 0), posFct.At(0, 1))
		//terms outside loop
		for l := 0; l < k; l++ {
			negFct.Set(0, l, negFct.At(0, l)+tsY_C.At(0, l)*tsY_C.At(0, l))
			value := Bsub.At(i, l)*Bsub.At(i, l) - 2*tsY_C.At(0, l)*Bsub.At(i, l)
			posFct.Set(0, l, posFct.At(0, l)+negFct.At(0, l)+value)
		}
		//sigma is full nLabel, but only top k used in the loop
		var negSum float64 = 0.0
		var posSum float64 = 0.0
		for l := 0; l < k; l++ {
			negValue := negFct.At(0, l) / (2 * sigmaSub.At(0, l) * sigmaSub.At(0, l))
			posValue := posFct.At(0, l) / (2 * sigmaSub.At(0, l) * sigmaSub.At(0, l))
			negFct.Set(0, l, negValue)
			posFct.Set(0, l, posValue)
			negSum = negSum + negValue
			posSum = posSum + posValue
		}
		//fmt.Println(i, logPos, logNeg, negFct.At(0, 0), posFct.At(0, 1))
		logPos = logPos - posSum
		logNeg = logNeg - negSum
		preQi := Q.At(0, i)
		newQi := math.Exp(logPos) / (math.Exp(logPos) + math.Exp(logNeg))
		//fmt.Println(i, preQi, newQi, posSum, negSum, logPos, logNeg, math.Exp(logPos), math.Exp(logNeg))
		//fmt.Println(i, preQi, newQi, logPos, logNeg)
		Q.Set(0, i, newQi)
		if (math.Abs(newQi - preQi)) > 0.0001 {
			//reset as all unprocessed
			//fmt.Println("reset at: ", i)
			for i := 0; i < nLabel; i++ {
				ind[i] = 1
			}
		}
		//mark as processed
		ind[i] = 0

		//find a new i with ind[i] == 1 value using idx order
		//no matter if reset to 1s, continue the outer for loop for 1s not processed with a larger idx
		//else, check if 1s exist and restart
		//fmt.Println(ind)
		isIdxFound := 0
		for j := i + 1; j < nLabel; j++ {
			if ind[j] == 1 {
				i = j
				isIdxFound = 1
				//fmt.Println("choose con: ", j, i)
				break
			}
		}
		if isIdxFound == 0 {
			for j := 0; j < nLabel; j++ {
				if ind[j] == 1 {
					i = j
					//fmt.Println("choose restart: ", j, i)
					break
				}
			}
		}
	}
	//return
	tsYhatData = make([]float64, 0)
	for i := 0; i < nLabel; i++ {
		tsYhatData = append(tsYhatData, Q.At(0, i))
	}
	return tsYhatData
}

func fOrderNegFctCal(negFct *mat64.Dense, tsY_C *mat64.Dense, Bsub *mat64.Dense, Q *mat64.Dense, j int) (newNegFct *mat64.Dense) {
	_, k := negFct.Caps()
	newNegFct = mat64.NewDense(1, k, nil)
	//term1 := mat64.NewDense(0, 0, nil)
	//BsubSlice := Bsub.Slice(j, j+1, 0, k)
	//term1.Mul(BsubSlice, tsY_C.T())
	//a, b := term1.Caps()
	//fmt.Println(a, b, k)
	for i := 0; i < k; i++ {
		value := Bsub.At(j, i) * Q.At(0, j)
		//value = value * value
		//fmt.Println(Bsub.At(j, i), Q.At(0, j), value)
		newNegFct.Set(0, i, negFct.At(0, i)+value*value-2*tsY_C.At(0, i)*Bsub.At(j, i)*Q.At(0, j))
		//value = 2 * tsY_C.At(0, i) * Bsub.At(j, i) * Q.At(0, j)
		//fmt.Println(tsY_C.At(0, i), value)
	}
	return newNegFct
}
func sOrderNegFctCal(negFct *mat64.Dense, Bsub *mat64.Dense, Q *mat64.Dense, j int, n int) (newNegFct *mat64.Dense) {
	_, k := negFct.Caps()
	newNegFct = mat64.NewDense(1, k, nil)
	for m := 0; m < k; m++ {
		value := 2 * Bsub.At(j, m) * Bsub.At(n, m) * Q.At(0, j) * Q.At(0, n)
		newNegFct.Set(0, m, negFct.At(0, m)+value)
	}
	return newNegFct
}
func posFctCal(posFct *mat64.Dense, Bsub *mat64.Dense, Q *mat64.Dense, i int, j int) (newPosFct *mat64.Dense) {
	_, k := posFct.Caps()
	newPosFct = mat64.NewDense(1, k, nil)
	for m := 0; m < k; m++ {
		value := 2 * Bsub.At(i, m) * Bsub.At(j, m) * Q.At(0, j)
		newPosFct.Set(0, m, posFct.At(0, m)+value)
		//fmt.Println("posFct: ", m, value)
	}
	return newPosFct
}

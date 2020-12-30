package src

import (
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
	"log"
	"math"
	//"math/rand"
	"runtime"
	"sync"
)

func single_IOC_MFADecoding_and_result(nTs int, k int, c int, tsY_Prob *mat64.Dense, tsY_C *mat64.Dense, sigma *mat64.Dense, Bsub *mat64.Dense, sigmaFcts float64, nLabel int, tsYdata *mat64.Dense, trYdata *mat64.Dense, rankCut int, minDims int, YhSet map[int]*mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	if k >= minDims {
		return
	}
	tsYhat := mat64.NewDense(nTs, nLabel, nil)
	for i := 0; i < nTs; i++ {
		//the doc seems to be old, (0,x] seems to be correct
		//dim checked to be correct
		arr := IOC_MFADecoding(nTs, i, tsY_Prob, tsY_C, sigma, Bsub, k, sigmaFcts, nLabel)
		tsYhat.SetRow(i, arr)
	}
	mutex.Lock()
	YhSet[c] = tsYhat
	mutex.Unlock()
}

//func single_adaptiveTrainRLS_Regress_CG(i int, trXdataB *mat64.Dense, folds map[int][]int, nFold int, nFea int, nTr int, tsXdataB *mat64.Dense, sigma *mat64.Dense, trY_Cdata *mat64.Dense, nTs int, tsY_C *mat64.Dense, randValues []float64, idxPerm []int, wg *sync.WaitGroup, mutex *sync.Mutex) {
//	defer wg.Done()
//	beta, _, optMSE := adaptiveTrainRLS_Regress_CG(trXdataB, trY_Cdata.ColView(i), folds, nFold, nFea, nTr, randValues, idxPerm)
//	//bias term for tsXdata added before
//	element := mat64.NewDense(0, 0, nil)
//	element.Mul(tsXdataB, beta)
//	mutex.Lock()
//	sigma.Set(0, i, math.Sqrt(optMSE))
//	for j := 0; j < nTs; j++ {
//		tsY_C.Set(j, i, element.At(j, 0))
//	}
//	mutex.Unlock()
//}

func single_sigmaAndtsYc_Update(i int, optMSE float64, regulazor float64, tsXdataB *mat64.Dense, trY_Cdata *mat64.Dense, trXdataB *mat64.Dense, randValues []float64, sigma *mat64.Dense, tsY_C *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//beta is weights, convert Y to Ymat
	nY, _ := trY_Cdata.Caps()
	nTs, _ := tsY_C.Caps()
	Ymat := mat64.NewDense(nY, 1, nil)
	for j := 0; j < nY; j++ {
		Ymat.Set(j, 0, trY_Cdata.At(j, i))
	}
	beta := TrainRLS_Regress_CG(trXdataB, Ymat, regulazor, randValues)
	//return beta, regulazor, optMSE
	//beta, _, optMSE := adaptiveTrainRLS_Regress_CG(trXdataB, trY_Cdata.ColView(i), folds, nFold, nFea, nTr, randValues, idxPerm, wg, mutex)
	//bias term for tsXdata added before
	element := mat64.NewDense(0, 0, nil)
	element.Mul(tsXdataB, beta)
	mutex.Lock()
	sigma.Set(0, i, math.Sqrt(optMSE))
	for j := 0; j < nTs; j++ {
		tsY_C.Set(j, i, element.At(j, 0))
	}
	mutex.Unlock()
}

func EcocRun(tsXdata *mat64.Dense, tsYdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, rankCut int, reg bool, kSet []int, lamdaSet []float64, nFold int, folds map[int][]int, randValues []float64, wg *sync.WaitGroup, mutex *sync.Mutex) (YhSet map[int]*mat64.Dense, colSum *mat64.Vector) {
	YhSet = make(map[int]*mat64.Dense)
	nK := len(kSet)
	sigmaFctsSet := lamdaToSigmaFctsSet(lamdaSet)
	colSum, trYdata = posFilter(trYdata)
	tsYdata = PosSelect(tsYdata, colSum)
	//vars
	_, nLabel := trYdata.Caps()
	nTr, nFea := trXdata.Caps()
	nTs, _ := tsXdata.Caps()
	//for rands
	//randValues := RandListFromUniDist(nTr, nFea)
	//idxPerm := rand.Perm(nTr)
	//min dims
	minDims := int(math.Min(float64(nFea), float64(nLabel)))
	if nFea < nLabel {
		log.Print("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
		return nil, nil
	}
	//tsY_prob for prob tuning
	tsY_Prob := mat64.NewDense(nTs, nLabel, nil)

	//adding bias term for tsXData, trXdata
	tsXdataB := addBiasTerm(nTs, tsXdata)
	trXdataB := addBiasTerm(nTr, trXdata)
	//regM := mat64.NewDense(1, nLabel, nil)
	//step 1
	wg.Add(nLabel)
	for i := 0; i < nLabel; i++ {
		go single_tsYlabel_linear(i, nTs, tsY_Prob, trXdata, trYdata, folds, nFold, nFea, tsXdataB, wg, mutex)
	}
	wg.Wait()
	log.Print("step 1: linear coding calculated.")
	//cca
	B := mat64.NewDense(0, 0, nil)
	if !reg {
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
	log.Print("step 2: cca coding calculated.")

	//CCA code
	trY_Cdata := mat64.NewDense(0, 0, nil)
	trY_Cdata.Mul(trYdata, B)
	//decoding with regression
	tsY_C := mat64.NewDense(nTs, nLabel, nil)
	sigma := mat64.NewDense(1, nLabel, nil)
	//nested err/weights estimation per label
	lamda := []float64{0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000}
	//err := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	errMap := make(map[int][]float64)
	for i := 0; i < nLabel; i++ {
		err := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
		errMap[i] = err
	}
	wg.Add(nLabel * len(lamda) * nFold)
	for i := 0; i < nLabel; i++ {
		//folds for regression estimates
		trainFold, testFold := subFolds(nFold, folds, nTr, trXdataB, trY_Cdata.ColView(i))
		//estimating error /weights
		for j := 0; j < len(lamda); j++ {
			for k := 0; k < nFold; k++ {
				go single_TrainRLS_Regress_CG_errUpdate(i, j, trainFold[k].X, testFold[k].X, trainFold[k].Y, testFold[k].Y, lamda[j], nFold, randValues, &errMap, wg, mutex)
			}
		}
	}
	wg.Wait()
	wg.Add(nLabel)
	//update sigma and tsY_C with best err and lamda
	for i := 0; i < nLabel; i++ {
		err := errMap[i]
		//min error index
		idx := minIdx(err)
		optMSE := err[idx]
		regulazor := lamda[idx]
		go single_sigmaAndtsYc_Update(i, optMSE, regulazor, tsXdataB, trY_Cdata, trXdataB, randValues, sigma, tsY_C, wg, mutex)
	}
	wg.Wait()
	log.Print("step 3: cg decoding finihsed.")
	//mean field decoding
	c := 0
	wg.Add(nK * len(sigmaFctsSet))
	for k := 0; k < nK; k++ {
		Bsub := mat64.DenseCopyOf(B.Slice(0, nLabel, 0, kSet[k]))
		for s := 0; s < len(sigmaFctsSet); s++ {
			go single_IOC_MFADecoding_and_result(nTs, kSet[k], c, tsY_Prob, tsY_C, sigma, Bsub, sigmaFctsSet[s], nLabel, tsYdata, trYdata, rankCut, minDims, YhSet, wg, mutex)
			c += 1
		}
	}
	wg.Wait()
	log.Print("step 4: mean field decoding finihsed.")
	runtime.GC()
	return YhSet, colSum
}

func single_tsYlabel_linear(i int, nTs int, tsY_Prob *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, folds map[int][]int, nFold int, nFea int, tsXdataB *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//wMat, regular, _, label := adaptiveTrainLGR_Liblin(trXdata, trYdata.ColView(i), folds, nFold, nFea)
	wMat, _, _, label := adaptiveTrainLGR_Liblin(trXdata, trYdata.ColView(i), folds, nFold, nFea)
	//regM.Set(0, i, regular)
	//tsY_Prob
	element := mat64.NewDense(0, 0, nil)
	element.Mul(tsXdataB, wMat)
	mutex.Lock()
	for j := 0; j < nTs; j++ {
		//the -1*element.At() is not making much sense, as the label would be 0/1
		//according to the origical LIBLINEAR doc
		//the sign is determined by the label. label 1 would be -1 * weight, label 0 would be weight
		if label == 1 {
			value := 1.0 / (1 + math.Exp(-1*element.At(j, 0)))
			tsY_Prob.Set(j, i, value)
		} else {
			value := 1.0 / (1 + math.Exp(1*element.At(j, 0)))
			tsY_Prob.Set(j, i, value)
		}
	}
	mutex.Unlock()
}

func adaptiveTrainLGR_Liblin(X *mat64.Dense, Y *mat64.Vector, folds map[int][]int, nFold int, nFeature int) (wMat *mat64.Dense, regulator float64, errFinal float64, label int) {
	//lamda := []float64{0.1, 1, 10}
	//err := []float64{0, 0, 0}
	lamda := []float64{0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000}
	err := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	//lamda := []float64{0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000}
	//err := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	nY := Y.Len()

	trainFold := make([]CvFold, nFold)
	testFold := make([]CvFold, nFold)

	for i := 0; i < nFold; i++ {
		posTrain := make([]int, 0)
		negTrain := make([]int, 0)
		posTest := make([]int, 0)
		negTest := make([]int, 0)
		posTestMap := map[int]int{}
		negTestMap := map[int]int{}
		//test set and map
		for j := 0; j < len(folds[i]); j++ {
			if Y.At(folds[i][j], 0) == 1.0 {
				posTest = append(posTest, folds[i][j])
				posTestMap[folds[i][j]] = folds[i][j]
			} else {
				negTest = append(negTest, folds[i][j])
				negTestMap[folds[i][j]] = folds[i][j]
			}
		}
		//the rest is for training
		for j := 0; j < nY; j++ {
			if Y.At(j, 0) == 1.0 {
				_, exist := posTestMap[j]
				if !exist {
					posTrain = append(posTrain, j)
				}
			} else {
				_, exist := negTestMap[j]
				if !exist {
					negTrain = append(negTrain, j)
				}
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
			LRmodel := Train(trainFold[j].X, trainFold[j].Y, 1.0, 1, 1.0/lamda[i], 0.1, 0.00001, nil)
			w := LRmodel.W()
			lastW := []float64{Pop(&w)}
			w = append(lastW, w...)
			fLabel := LRmodel.Label()
			if fLabel == 0 {
				for k := 0; k < len(w); k++ {
					w[k] = w[k] * -1
				}
			}
			wMat := mat64.NewDense(len(w), 1, w)
			e := 1.0 - computeF1(testFold[j].X, testFold[j].Y, wMat)
			err[i] = err[i] + e
		}
		err[i] = err[i] / float64(nFold)
	}
	//min error index
	idx := minIdx(err)
	regulator = 1.0 / lamda[idx]
	Ymat := mat64.NewDense(Y.Len(), 1, nil)
	for i := 0; i < Y.Len(); i++ {
		Ymat.Set(i, 0, Y.At(i, 0))
	}
	//LRmodel := Train(X, Ymat, 1.0, 0, regulator, 0.1, 0.001, nil)
	LRmodel := Train(X, Ymat, 1.0, 1, regulator, 0.1, 0.00001, nil)
	w := LRmodel.W()
	lastW := []float64{Pop(&w)}
	w = append(lastW, w...)
	wMat = mat64.NewDense(len(w), 1, w)
	label = LRmodel.Label()
	//defer C.free(unsafe.Pointer(LRmodel))
	errFinal = err[idx]
	return wMat, regulator, errFinal, label
}
func adaptiveTrainRLS_Regress_CG(X *mat64.Dense, Y *mat64.Vector, folds map[int][]int, nFold int, nFeature int, nTr int, randValues []float64, idxPerm []int, wg *sync.WaitGroup, mutex *sync.Mutex) (beta *mat64.Dense, regulazor float64, optMSE float64) {

	//wg.Add(len(lamda) * nFold)
	return beta, regulazor, optMSE
}

func subFolds(nFold int, folds map[int][]int, nTr int, X *mat64.Dense, Y *mat64.Vector) (trainFold []CvFold, testFold []CvFold) {
	trainFold = make([]CvFold, nFold)
	testFold = make([]CvFold, nFold)
	for i := 0; i < nFold; i++ {
		cvTrain := make([]int, 0)
		cvTest := make([]int, 0)
		cvTestMap := map[int]int{}
		for j := 0; j < len(folds[i]); j++ {
			cvTest = append(cvTest, folds[i][j])
			cvTestMap[folds[i][j]] = folds[i][j]
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
	return trainFold, testFold
}

func single_TrainRLS_Regress_CG_errUpdate(i int, j int, trFoldX *mat64.Dense, tsFoldX *mat64.Dense, trFoldY *mat64.Dense, tsFoldY *mat64.Dense, lamda float64, nFold int, randValues []float64, errMap *map[int][]float64, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//weights finalized for one lamda and one fold
	weights := TrainRLS_Regress_CG(trFoldX, trFoldY, lamda, randValues)
	term1 := mat64.NewDense(0, 0, nil)
	term2 := mat64.NewDense(0, 0, nil)
	//trXdata and tsXdata are "cbinded" previously in main
	term1.Mul(tsFoldX, weights)
	term2.Sub(term1, tsFoldY)
	var sum float64 = 0
	var mean float64
	r, c := term2.Caps()
	for m := 0; m < r; m++ {
		for n := 0; n < c; n++ {
			sum += term2.At(m, n) * term2.At(m, n)
		}
	}
	mean = sum / float64(r*c)
	mutex.Lock()
	(*errMap)[i][j] = (*errMap)[i][j] + mean/float64(nFold)
	mutex.Unlock()
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
	term3 := mat64.NewDense(0, 0, nil)
	term1.Sub(Y, products)
	term2.Mul(X.T(), term1)
	term3 = MulEleByFloat64(-1*lamda, weights)
	gradient = mat64.NewDense(0, 0, nil)
	gradient.Add(term3, term2)
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
	term6 := mat64.NewDense(0, 0, nil)
	term7 := mat64.NewDense(0, 0, nil)
	beta := mat64.NewDense(0, 0, nil)
	term1.Sub(gradient, preGradient)
	term2.Mul(cg.T(), term1)
	term3.Mul(gradient.T(), term1)
	//right matrix division in matlab. A/B = A*inv(B)
	//term4.Inverse(term2)
	//beta.Mul(term3, term4)
	//This A*inv(B) works if B is roughly square
	//beta=A*B'*inv(B*B')
	//A is Term3, B is Term2
	term4.Mul(term2, term2.T())
	term6.Inverse(term4)
	term7.Mul(term3, term2.T())
	beta.Mul(term7, term6)
	term5.Mul(cg, beta)
	cg2 = mat64.NewDense(0, 0, nil)
	cg2.Sub(gradient, term5)
	return cg2
}
func stepCal(gradient *mat64.Dense, cg *mat64.Dense, lamda float64, X *mat64.Dense) (step float64) {
	term1 := mat64.NewDense(0, 0, nil)
	term2 := mat64.NewDense(0, 0, nil)
	term3 := mat64.NewDense(0, 0, nil)
	term1.Mul(X, cg)
	r, c := term1.Caps()
	sum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += term1.At(i, j) * term1.At(i, j)
		}
	}
	term2.Mul(cg.T(), cg)
	term3.Mul(gradient.T(), cg)
	step = term3.At(0, 0) / (lamda*term2.At(0, 0) + sum)
	return step
}
func deltaLossCal(Y *mat64.Dense, products *mat64.Dense, lamda float64, weights *mat64.Dense, preProducts *mat64.Dense, preWeights *mat64.Dense) (deltaLoss float64) {
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

func RandListFromUniDist(length int, length2 int) (values []float64) {
	var UformDist = distuv.Uniform{Min: -0.00000001, Max: 0.00000001}
	if length < length2 {
		length = length2
	}
	for k := 0; k <= length; k++ {
		value := UformDist.Rand()
		values = append(values, value)
	}
	return values
}
func TrainRLS_Regress_CG(trFoldX *mat64.Dense, trFoldY *mat64.Dense, lamda float64, randValues []float64) (weights *mat64.Dense) {
	n, p := trFoldX.Caps()
	//weight
	weights = mat64.NewDense(p, 1, nil)
	preWeights := mat64.NewDense(p, 1, nil)
	for k := 0; k < p; k++ {
		//value := UformDist.Rand()
		weights.Set(k, 0, randValues[k])
		preWeights.Set(k, 0, randValues[k])
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
	gradient := gradientCal(lamda, weights, trFoldX, trFoldY, products)
	iter := 0
	maxIter := 6000
	maxDiff := maxDiffCal(products, preProducts, n)
	cg := mat64.DenseCopyOf(gradient.View(0, 0, p, 1))
	//the while loop
	for maxDiff > 0.0000001 && iter < maxIter {
		iter += 1
		//conjugate gradient
		if iter > 1 {
			cg = cgCal(gradient, preGradient, cg)
		}
		//projection
		step := stepCal(gradient, cg, lamda, trFoldX)
		preProducts.Copy(products)
		preGradient.Copy(gradient)
		preWeights.Copy(weights)
		//update weight
		for k := 0; k < p; k++ {
			weights.Set(k, 0, preWeights.At(k, 0)+step*cg.At(k, 0))
		}
		products.Mul(trFoldX, weights)
		gradient = gradientCal(lamda, weights, trFoldX, trFoldY, products)
		deltaLoss := deltaLossCal(trFoldY, products, lamda, weights, preProducts, preWeights)

		for deltaLoss > 0.0000000001 {
			step = step / 10
			//fmt.Println("step and dLoss: ", step, deltaLoss)
			//update weight
			for k := 0; k < p; k++ {
				weights.Set(k, 0, preWeights.At(k, 0)+step*cg.At(k, 0))
			}
			products.Mul(trFoldX, weights)
			gradient = gradientCal(lamda, weights, trFoldX, trFoldY, products)
			deltaLoss = deltaLossCal(trFoldY, products, lamda, weights, preProducts, preWeights)
			if step == 0.0 {
				log.Print("early stop at conjugate gradient.")
				break
			}
		}
		maxDiff = maxDiffCal(products, preProducts, n)
	}
	if iter == maxIter {
		log.Print("reached max Iter for conjugate gradient.")
	}
	return weights
}

func IOC_MFADecoding(nRowTsY int, rowIdx int, tsY_Prob *mat64.Dense, tsY_C *mat64.Dense, sigma *mat64.Dense, Bsub *mat64.Dense, k int, sigmaFcts float64, nLabel int) (tsYhatData []float64) {
	//Q
	Q := mat64.NewDense(1, nLabel, nil)
	for i := 0; i < nLabel; i++ {
		Q.Set(0, i, tsY_Prob.At(rowIdx, i))
	}
	//sigma and B for top k elements
	sigmaSub := mat64.NewDense(1, k, nil)
	for i := 0; i < k; i++ {
		sigmaSub.Set(0, i, sigma.At(0, i)*sigmaFcts)
	}
	//ind
	ind := make([]int, nLabel)
	for i := 0; i < nLabel; i++ {
		ind[i] = 1
	}
	//init index
	i := 0
	posFct := mat64.NewDense(1, k, nil)
	negFct := mat64.NewDense(1, k, nil)
	for ind[i] > 0 {
		logPos := math.Log(tsY_Prob.At(rowIdx, i))
		logNeg := math.Log(1 - tsY_Prob.At(rowIdx, i))
		//reset posFct and negFct to zeros
		for l := 0; l < k; l++ {
			negFct.Set(0, l, 0)
			posFct.Set(0, l, 0)
		}
		for j := 0; j < nLabel; j++ {
			if j == i || Q.At(0, j) == 0 {
				continue
			}
			//negFct = fOrderNegFctCal(negFct, tsY_C, Bsub, Q, j, rowIdx)
			fOrderNegFctCal(negFct, tsY_C, Bsub, Q, j, rowIdx)
			//second order, n is j2, golang is 0 based, so that the for loop is diff on max
			for n := 0; n < j; n++ {
				if n == i || Q.At(0, n) == 0 {
					continue
				}
				//negFct = sOrderNegFctCal(negFct, Bsub, Q, j, n)
				sOrderNegFctCal(negFct, Bsub, Q, j, n)
			}
			//posFct
			//posFct = posFctCal(posFct, Bsub, Q, i, j)
			posFctCal(posFct, Bsub, Q, i, j)
		}
		//terms outside loop
		for l := 0; l < k; l++ {
			negFct.Set(0, l, negFct.At(0, l)+tsY_C.At(rowIdx, l)*tsY_C.At(rowIdx, l))
			value := Bsub.At(i, l)*Bsub.At(i, l) - 2*tsY_C.At(rowIdx, l)*Bsub.At(i, l)
			posFct.Set(0, l, posFct.At(0, l)+negFct.At(0, l)+value)
		}
		//sigma is full nLabel length, but only top k used in the loop
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
		logPos = logPos - posSum
		logNeg = logNeg - negSum
		preQi := Q.At(0, i)
		newQi := math.Exp(logPos) / (math.Exp(logPos) + math.Exp(logNeg))
		Q.Set(0, i, newQi)
		if (math.Abs(newQi - preQi)) > 0.0001 {
			//reset as all unprocessed
			for i := 0; i < nLabel; i++ {
				ind[i] = 1
			}
		}
		//mark as processed
		ind[i] = 0

		//find a new i with ind[i] == 1 value using idx order
		//no matter if reset to 1s, continue the outer for loop for 1s not processed with a larger idx
		//else, check if 1s exist and restart
		isIdxFound := 0
		for j := i + 1; j < nLabel; j++ {
			if ind[j] == 1 {
				i = j
				isIdxFound = 1
				break
			}
		}
		if isIdxFound == 0 {
			for j := 0; j < nLabel; j++ {
				if ind[j] == 1 {
					i = j
					break
				}
			}
		}
	}
	//return
	tsYhatData = make([]float64, 0)
	for i := 0; i < nLabel; i++ {
		if math.IsNaN(Q.At(0, i)) {
			tsYhatData = append(tsYhatData, 0.0)
		} else {
			tsYhatData = append(tsYhatData, Q.At(0, i))
		}
	}
	return tsYhatData
}

func fOrderNegFctCal(negFct *mat64.Dense, tsY_C *mat64.Dense, Bsub *mat64.Dense, Q *mat64.Dense, j int, rowIdx int) {
	_, k := negFct.Caps()
	for i := 0; i < k; i++ {
		value := Bsub.At(j, i) * Q.At(0, j)
		negFct.Set(0, i, negFct.At(0, i)+value*value-2*tsY_C.At(rowIdx, i)*Bsub.At(j, i)*Q.At(0, j))
	}
	return
}
func sOrderNegFctCal(negFct *mat64.Dense, Bsub *mat64.Dense, Q *mat64.Dense, j int, n int) {
	_, k := negFct.Caps()
	for m := 0; m < k; m++ {
		value := 2 * Bsub.At(j, m) * Bsub.At(n, m) * Q.At(0, j) * Q.At(0, n)
		negFct.Set(0, m, negFct.At(0, m)+value)
	}
	return
}
func posFctCal(posFct *mat64.Dense, Bsub *mat64.Dense, Q *mat64.Dense, i int, j int) {
	_, k := posFct.Caps()
	for m := 0; m < k; m++ {
		value := 2 * Bsub.At(i, m) * Bsub.At(j, m) * Q.At(0, j)
		posFct.Set(0, m, posFct.At(0, m)+value)
	}
	return
}

func addBiasTerm(len int, mat *mat64.Dense) (mat2 *mat64.Dense) {
	ones := make([]float64, len)
	for i := range ones {
		ones[i] = 1
	}
	mat2 = colStack(mat, ones)
	return mat2
}

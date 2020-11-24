package src

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"log"
	"sync"
)

func ReadNetworkPropagate(trRowName []string, tsRowName []string, trYdata *mat64.Dense, inNetworkFile []string, priorMatrixFile []string, isAddPrior bool, isDada bool, alpha float64, wg *sync.WaitGroup, mutex *sync.Mutex) (tsXdata *mat64.Dense, trXdata *mat64.Dense, indAccum []int) {
	tsXdata = mat64.NewDense(0, 0, nil)
	trXdata = mat64.NewDense(0, 0, nil)
	// for filtering prior genes, only those in training set are used for propagation
	trGeneMap := make(map[string]int)
	for i := 0; i < len(trRowName); i++ {
		trGeneMap[trRowName[i]] = i
	}
	//network
	network := mat64.NewDense(0, 0, nil)
	idIdx := make(map[string]int)
	idxToId := make(map[int]string)
	indAccum = make([]int, 0)
	for i := 0; i < len(inNetworkFile); i++ {
		//idIdx as gene -> idx in net
		log.Print("loading network file: ", inNetworkFile[i])
		network, idIdx, idxToId = ReadNetwork(inNetworkFile[i])
		if !isAddPrior {
			sPriorData, ind := PropagateSet(network, trYdata, idIdx, trRowName, trGeneMap, isDada, alpha, wg, mutex)
			tsXdata, trXdata = FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
			indAccum = append(indAccum, ind...)
		} else {
			for j := 0; j < len(priorMatrixFile); j++ {
				priorData, priorGeneID, priorIdxToId := ReadNetwork(priorMatrixFile[j])
				sPriorData, ind := PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdata, idIdx, idxToId, trRowName, trGeneMap, isDada, alpha, wg, mutex)
				tsXdata, trXdata = FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
				indAccum = append(indAccum, ind...)
			}
		}
	}
	return tsXdata, trXdata, indAccum
}
func ReadNetworkPropagateCV(f int, folds map[int][]int, trRowName []string, tsRowName []string, trYdata *mat64.Dense, inNetworkFile []string, priorMatrixFile []string, isAddPrior bool, isDada bool, alpha float64, wg *sync.WaitGroup, mutex *sync.Mutex) (cvTrain []int, cvTest []int, trXdataCV *mat64.Dense, indAccum []int) {
	cvTrain = make([]int, 0)
	cvTest = make([]int, 0)
	cvTestMap := map[int]int{}
	nTr, _ := trYdata.Caps()
	for j := 0; j < len(folds[f]); j++ {
		cvTest = append(cvTest, folds[f][j])
		cvTestMap[folds[f][j]] = folds[f][j]
	}
	//the rest is for training
	for j := 0; j < nTr; j++ {
		_, exist := cvTestMap[j]
		if !exist {
			cvTrain = append(cvTrain, j)
		}
	}
	//generating ECOC
	//trXdataCV should use genes in trYdata for training only
	trGeneMapCV := make(map[string]int)
	for j := 0; j < len(cvTrain); j++ {
		trGeneMapCV[trRowName[cvTrain[j]]] = cvTrain[j]
	}
	trXdataCV = mat64.NewDense(0, 0, nil)
	_, nColY := trYdata.Caps()
	trYdataCV := mat64.NewDense(len(cvTrain), nColY, nil)
	trRowNameCV := make([]string, 0)
	for s := 0; s < len(cvTrain); s++ {
		trYdataCV.SetRow(s, trYdata.RawRowView(cvTrain[s]))
		trRowNameCV = append(trRowNameCV, trRowName[cvTrain[s]])
	}
	//codes
	indAccum = make([]int, 0)
	for i := 0; i < len(inNetworkFile); i++ {
		//idIdx as gene -> idx in net
		network, idIdx, idxToId := ReadNetwork(inNetworkFile[i])
		if !isAddPrior {
			sPriorData, ind := PropagateSet(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, isDada, alpha, wg, mutex)
			indAccum = append(indAccum, ind...)
			_, nTrLabel := trYdataCV.Caps()
			_, nLabel := sPriorData.Caps()
			tmpTrXdata := mat64.NewDense(len(trRowName), nLabel, nil)
			//trX
			cLabel := 0
			for l := 0; l < nTrLabel; l++ {
				if ind[l] > 1 {
					for k := 0; k < len(trRowName); k++ {
						_, exist := idIdx[trRowName[k]]
						if exist {
							tmpTrXdata.Set(k, cLabel, sPriorData.At(idIdx[trRowName[k]], cLabel)/float64(ind[l]))
							//adding trY label as max value to trX
							if trYdata.At(k, l) == 1.0 {
								_, exist2 := cvTestMap[k]
								if !exist2 {
									tmpTrXdata.Set(k, cLabel, 1.0/float64(ind[l]))
								}
							}
						}
					}
					cLabel += 1
				}
			}
			nRow, _ := trXdataCV.Caps()
			if nRow == 0 {
				trXdataCV = tmpTrXdata
			} else {
				trXdataCV = ColStackMatrix(trXdataCV, tmpTrXdata)
			}
		} else {
			for j := 0; j < len(priorMatrixFile); j++ {
				priorData, priorGeneID, priorIdxToId := ReadNetwork(priorMatrixFile[j])
				sPriorData, ind := PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdataCV, idIdx, idxToId, trRowNameCV, trGeneMapCV, isDada, alpha, wg, mutex)
				indAccum = append(indAccum, ind...)
				_, nTrLabel := trYdataCV.Caps()
				_, nLabel := sPriorData.Caps()
				tmpTrXdata := mat64.NewDense(len(trRowName), nLabel, nil)
				//trX
				cLabel := 0
				for l := 0; l < nTrLabel; l++ {
					if ind[l] > 1 {
						for k := 0; k < len(trRowName); k++ {
							_, exist := idIdx[trRowName[k]]
							if exist {
								tmpTrXdata.Set(k, cLabel, sPriorData.At(idIdx[trRowName[k]], cLabel)/float64(ind[l]))
								if trYdata.At(k, l) == 1.0 {
									_, exist2 := cvTestMap[k]
									if !exist2 {
										tmpTrXdata.Set(k, cLabel, 1.0/float64(ind[l]))
									}
								}
							}
						}
						cLabel += 1
					}
				}
				nRow, _ := trXdataCV.Caps()
				if nRow == 0 {
					trXdataCV = tmpTrXdata
				} else {
					trXdataCV = ColStackMatrix(trXdataCV, tmpTrXdata)
				}
			}
		}
	}
	return cvTrain, cvTest, trXdataCV, indAccum
}

func TuneAndPredict(nFold int, folds map[int][]int, fBetaThres float64, nK int, nKnn int, isFirst bool, isKnn bool, kSet []int, lamdaSet []float64, reg bool, rankCut int, trainFold []CvFold, testFold []CvFold, indAccum []int, tsXdata *mat64.Dense, tsYdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) (trainMeasureUpdated *mat64.Dense, testMeasureUpdated *mat64.Dense, tsYhat *mat64.Dense, thres *mat64.Dense, Yhat *mat64.Dense, YhatCalibrated *mat64.Dense, Ylabel *mat64.Dense) {
	//traing data per hyperparameter
	YhPlattSet := make(map[int]*mat64.Dense)
	YhPlattSetCalibrated := make(map[int]*mat64.Dense)
	yPlattSet := make(map[int]*mat64.Dense)
	yPredSet := make(map[int]*mat64.Dense)
	iFoldMarker := make(map[int]*mat64.Dense)
	xSet := make(map[int]*mat64.Dense)
	plattRobustMeasure := make(map[int]*mat64.Dense)
	plattRobustLamda := []float64{0.0, 0.04, 0.08, 0.12, 0.16, 0.2}
	_, nLabel := trYdata.Caps()
	globalBetaRaw := mat64.NewDense(nK*len(lamdaSet), nLabel, nil)
	globalBetaKnn := mat64.NewDense(nK*len(lamdaSet), nLabel, nil)
	//subFolds
	subFolds := make(map[int]map[int][]int)
	//nested folds for training data
	wg.Add(nFold)
	for f := 0; f < nFold; f++ {
		go single_SOIS(subFolds, f, trainFold[f].Y, nFold, 10, 0, false, wg, mutex)
	}
	wg.Wait()
	log.Print("nested SOIS folds generated.")
	//measure matrix
	nL := nK * len(lamdaSet)
	trainMeasure := mat64.NewDense(nL, 15, nil)
	testMeasure := mat64.NewDense(1, 9, nil)

	for i := 0; i < nFold; i++ {
		YhSet, colSum := EcocRun(testFold[i].X, testFold[i].Y, trainFold[i].X, trainFold[i].Y, rankCut, reg, kSet, lamdaSet, nFold, subFolds[i], nK, wg, mutex)
		tsYFold := PosSelect(testFold[i].Y, colSum)

		c := 0
		wg.Add(nK * len(lamdaSet))
		//accum calculated training data
		for m := 0; m < nK; m++ {
			for n := 0; n < len(lamdaSet); n++ {
				go single_AccumTsYdata(tsYFold, YhSet[c], i, c, colSum, testFold[i].X, testFold[i].IndAccum, YhPlattSet, YhPlattSetCalibrated, yPlattSet, iFoldMarker, yPredSet, xSet, plattRobustMeasure, plattRobustLamda, wg, mutex)
				c += 1
			}
		}
		wg.Wait()
		log.Print("step 4: recalibration data obtained.")
	}
	//update all meassures before or after KNN calibration
	for i := 0; i < nFold; i++ {
		c := 0
		wg.Add(nK * len(lamdaSet))
		for m := 0; m < nK; m++ {
			for n := 0; n < len(lamdaSet); n++ {
				//trainMeasure, globalBetaRaw and globalBetaKnn recorded
				go single_RecordMeasures(rankCut, kSet[m], lamdaSet[n], c, nLabel, nKnn, trainMeasure, globalBetaRaw, globalBetaKnn, i, yPlattSet, YhPlattSet, YhPlattSetCalibrated, yPredSet, xSet, iFoldMarker, plattRobustMeasure, plattRobustLamda, posLabelRls, negLabelRls, wg, mutex)
				c += 1
			}
		}
		wg.Wait()
	}
	log.Print("pass training.")

	//choosing object function, all hyper parameters, nDim in CCA, lamda and kNN calibration
	//the index value is 0-based, so that the same index in code above
	objectBaseNum := 11
	if isFirst {
		objectBaseNum = 13
		log.Print("choose aupr for first label as object function in tuning.")
	} else {
		log.Print("choose micro-aupr for all labels as object function in tuning.")
	}
	cBestRaw, vBestRaw, vbMauprRaw, vbMf1Raw := BestHyperParameterSetByMeasure(trainMeasure, objectBaseNum, false)
	cBestKnn, vBestKnn, vbMauprKnn, vbMf1Knn := BestHyperParameterSetByMeasure(trainMeasure, objectBaseNum, true)
	cBest := 0
	//isKnn := false
	if vBestKnn > vBestRaw || isKnn {
		isKnn = true
		cBest = cBestKnn
	} else {
		cBest = cBestRaw
	}
	//best training parameters, data and max Pos scaling factor
	kSet = []int{int(trainMeasure.At(cBest, 0))}
	lamdaSet = []float64{trainMeasure.At(cBest, 1)}
	//maxArr := []float64{}
	thres = mat64.NewDense(0, 0, nil)
	plattAB := mat64.NewDense(0, 0, nil)
	//minMSElamda
	_, nCol := plattRobustMeasure[cBest].Caps()
	minMSElamda := make([]float64, nCol)
	minMSE := make([]float64, nCol)
	for p := 0; p < len(plattRobustLamda); p++ {
		for q := 0; q < nCol; q++ {
			if p == 0 {
				minMSE[q] = plattRobustMeasure[cBest].At(p, q)
				minMSElamda[q] = plattRobustLamda[p]
			} else {
				if plattRobustMeasure[cBest].At(p, q) < minMSE[q] {
					minMSE[q] = plattRobustMeasure[cBest].At(p, q)
					minMSElamda[q] = plattRobustLamda[p]
				}
			}
		}
	}
	//platt scale for testing
	YhPlattScale, _ := QuantileNorm(YhPlattSet[cBest], mat64.NewDense(0, 0, nil), false)
	YhPlattScale, plattAB, _ = Platt(YhPlattScale, yPlattSet[cBest], YhPlattScale, minMSElamda)
	if isKnn {
		maValue := fmt.Sprintf("%.3f", vbMauprKnn/float64(nFold))
		mf1Value := fmt.Sprintf("%.3f", vbMf1Knn/float64(nFold))
		betaValue := mat64.NewDense(1, nLabel, nil)
		log.Print("choose kNN calibration with microAupr of "+maValue+" and microF1 of ", mf1Value, ".")
		log.Print("choose kNN calibration with these beta values per label for F-thresholding.")
		str := ""
		for p := 0; p < nLabel; p++ {
			betaValue.Set(0, p, globalBetaKnn.At(cBest, p)/float64(nFold))
			s := fmt.Sprintf("%.3f", globalBetaKnn.At(cBest, p)/float64(nFold))
			str = str + "\t" + s
		}
		log.Print(str)
		thres = FscoreThres(yPlattSet[cBest], YhPlattSetCalibrated[cBest], betaValue)
	} else {
		maValue := fmt.Sprintf("%.3f", vbMauprRaw/float64(nFold))
		mf1Value := fmt.Sprintf("%.3f", vbMf1Raw/float64(nFold))
		betaValue := mat64.NewDense(1, nLabel, nil)
		log.Print("choose raw score with microAupr of "+maValue+" and microF1 of ", mf1Value, ".")
		log.Print("choose raw score with these beta values per label for F-thresholding.")
		str := ""

		for p := 0; p < nLabel; p++ {
			betaValue.Set(0, p, globalBetaRaw.At(cBest, p)/float64(nFold))
			s := fmt.Sprintf("%.3f", globalBetaRaw.At(cBest, p)/float64(nFold))
			str = str + "\t" + s
		}
		log.Print(str)
		thres = FscoreThres(yPlattSet[cBest], YhPlattScale, betaValue)
	}
	//testing run with cBest hyperparameter
	YhSet, _ := EcocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, lamdaSet, nFold, folds, 1, wg, mutex)
	tsYhat, _ = QuantileNorm(YhSet[0], mat64.NewDense(0, 0, nil), false)
	tsYhat = PlattScaleSet(tsYhat, plattAB)
	if isKnn {
		tsXdata = RefillIndCol(tsXdata, indAccum)
		tsYhat = MultiLabelRecalibrate(nKnn, tsYhat, tsXdata, yPlattSet[cBest], yPredSet[cBest], xSet[cBest], posLabelRls, negLabelRls, wg, mutex)
	}
	tsYhat, thres = SoftThresScale(tsYhat, thres)
	//corresponding testing measures
	c := 0
	i := 0
	for j := 0; j < len(lamdaSet); j++ {
		accuracy, microF1, microAupr, macroAupr, agMicroF1, _, firstAupr := Report(tsYdata, tsYhat, thres, rankCut, false)
		testMeasure.Set(c, 0, float64(kSet[i]))
		testMeasure.Set(c, 1, lamdaSet[j])
		testMeasure.Set(c, 2, testMeasure.At(c, 2)+1.0)
		testMeasure.Set(c, 3, testMeasure.At(c, 3)+accuracy)
		testMeasure.Set(c, 4, testMeasure.At(c, 4)+microF1)
		testMeasure.Set(c, 5, testMeasure.At(c, 5)+microAupr)
		testMeasure.Set(c, 6, testMeasure.At(c, 6)+macroAupr)
		testMeasure.Set(c, 7, testMeasure.At(c, 7)+agMicroF1)
		testMeasure.Set(c, 8, testMeasure.At(c, 8)+firstAupr)
		c += 1
	}
	return trainMeasure, testMeasure, tsYhat, thres, YhPlattScale, YhPlattSetCalibrated[cBest], yPlattSet[cBest]
}

func single_AccumTsYdata(tsYfold *mat64.Dense, rawTsYhat *mat64.Dense, iFold int, c int, colSum *mat64.Vector, tsX *mat64.Dense, indAccum []int, YhPlattSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, yPredSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	tsYhat, _ := QuantileNorm(rawTsYhat, mat64.NewDense(0, 0, nil), false)
	_, nCol := tsYhat.Caps()
	minMSElamda := make([]float64, nCol)
	minMSE := make([]float64, nCol)
	_, isDefinedMSE := plattRobustMeasure[c]
	mutex.Lock()
	if !isDefinedMSE {
		plattRobustMeasure[c] = mat64.NewDense(len(plattRobustLamda), nCol, nil)
	}
	for p := 0; p < len(plattRobustLamda); p++ {
		tmpLamda := make([]float64, 0)
		for q := 0; q < nCol; q++ {
			tmpLamda = append(tmpLamda, plattRobustLamda[p])
		}
		_, _, mseArr := Platt(tsYhat, tsYfold, tsYhat, tmpLamda)
		for q := 0; q < nCol; q++ {
			plattRobustMeasure[c].Set(p, q, plattRobustMeasure[c].At(p, q)+mseArr[q])
			if p == 0 {
				minMSE[q] = mseArr[q]
				minMSElamda[q] = plattRobustLamda[p]
			} else {
				if mseArr[q] < minMSE[q] {
					minMSE[q] = mseArr[q]
					minMSElamda[q] = plattRobustLamda[p]
				}
			}
		}
	}
	mutex.Unlock()

	tsYhat, _, _ = Platt(tsYhat, tsYfold, tsYhat, minMSElamda)
	//raw thres added
	rawBeta := FscoreBeta(tsYfold, tsYhat)
	rawThres := FscoreThres(tsYfold, tsYhat, rawBeta)
	//tsYhat, rawThres = QuantileNorm(tsYhat, rawThres, true)
	tsYhat, rawThres = SoftThresScale(tsYhat, rawThres)
	//accum to add information for KNN calibaration
	mutex.Lock()
	AccumTsYdata(iFold, c, colSum, rawTsYhat, tsYfold, tsX, indAccum, YhPlattSet, YhPlattSetCalibrated, yPlattSet, iFoldMarker, yPredSet, xSet, rawThres)
	mutex.Unlock()
}

func single_RecordMeasures(rankCut int, k int, lamda float64, c int, nLabel int, nKnn int, trainMeasure *mat64.Dense, globalBetaRaw *mat64.Dense, globalBetaKnn *mat64.Dense, i int, yPlattSet map[int]*mat64.Dense, YhPlattSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPredSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	yPlattTrain, yPredTrain, xTrain, xTest, tsYhat, tsYfold := SubSetTrain(i, yPlattSet[c], YhPlattSet[c], yPredSet[c], xSet[c], iFoldMarker[c])
	//calculate platt scaled tsYhat again for measures
	_, nCol := tsYhat.Caps()
	minMSElamda := make([]float64, nCol)
	minMSE := make([]float64, nCol)
	for p := 0; p < len(plattRobustLamda); p++ {
		for q := 0; q < nCol; q++ {
			if p == 0 {
				minMSE[q] = plattRobustMeasure[c].At(p, q)
				minMSElamda[q] = plattRobustLamda[p]
			} else {
				if plattRobustMeasure[c].At(p, q) < minMSE[q] {
					minMSE[q] = plattRobustMeasure[c].At(p, q)
					minMSElamda[q] = plattRobustLamda[p]
				}
			}
		}
	}
	tsYhat, _ = QuantileNorm(tsYhat, mat64.NewDense(0, 0, nil), false)
	tsYhat, _, _ = Platt(tsYhat, tsYfold, tsYhat, minMSElamda)
	beta := FscoreBeta(tsYfold, tsYhat)
	rawThres := FscoreThres(tsYfold, tsYhat, beta)
	tsYhat, rawThres = SoftThresScale(tsYhat, rawThres)
	accuracy, microF1, microAupr, macroAupr, _, optScore, firstAupr := Report(tsYfold, tsYhat, rawThres, rankCut, false)
	mutex.Lock()
	for p := 0; p < nLabel; p++ {
		globalBetaRaw.Set(c, p, globalBetaRaw.At(c, p)+beta.At(0, p))
	}
	trainMeasure.Set(c, 0, float64(k))
	trainMeasure.Set(c, 1, lamda)
	trainMeasure.Set(c, 2, trainMeasure.At(c, 2)+1.0)
	trainMeasure.Set(c, 3, trainMeasure.At(c, 3)+accuracy)
	trainMeasure.Set(c, 5, trainMeasure.At(c, 5)+microF1)
	trainMeasure.Set(c, 7, trainMeasure.At(c, 7)+microAupr)
	trainMeasure.Set(c, 9, trainMeasure.At(c, 9)+macroAupr)
	trainMeasure.Set(c, 11, trainMeasure.At(c, 11)+optScore)
	trainMeasure.Set(c, 13, trainMeasure.At(c, 13)+firstAupr)
	mutex.Unlock()
	//probability to be recalibrated for label dependency, subset train by fold
	if nKnn > 0 {
		tsYhat = MultiLabelRecalibrate(nKnn, tsYhat, xTest, yPlattTrain, yPredTrain, xTrain, posLabelRls, negLabelRls, wg, mutex)
		rawBeta := FscoreBeta(tsYfold, tsYhat)
		rawThres := FscoreThres(tsYfold, tsYhat, rawBeta)
		tsYhat, rawThres = SoftThresScale(tsYhat, rawThres)
		//update MultiLabelRecalibrate tsYhat to YhPlattSet
		YhPlattSetUpdate(i, c, YhPlattSetCalibrated, tsYhat, iFoldMarker[c])
		accuracy, microF1, microAupr, macroAupr, _, optScore, firstAupr = Report(tsYfold, tsYhat, rawThres, rankCut, false)
		mutex.Lock()
		for p := 0; p < nLabel; p++ {
			globalBetaKnn.Set(c, p, globalBetaKnn.At(c, p)+rawBeta.At(0, p))
		}
		trainMeasure.Set(c, 4, trainMeasure.At(c, 4)+accuracy)
		trainMeasure.Set(c, 6, trainMeasure.At(c, 6)+microF1)
		trainMeasure.Set(c, 8, trainMeasure.At(c, 8)+microAupr)
		trainMeasure.Set(c, 10, trainMeasure.At(c, 10)+macroAupr)
		trainMeasure.Set(c, 12, trainMeasure.At(c, 12)+optScore)
		trainMeasure.Set(c, 14, trainMeasure.At(c, 14)+firstAupr)
		mutex.Unlock()
	}
}

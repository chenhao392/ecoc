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

func single_AccumTsYdata(globalBeta *mat64.Dense, tsYfold *mat64.Dense, rawTsYhat *mat64.Dense, iFold int, c int, colSum *mat64.Vector, tsX *mat64.Dense, indAccum []int, YhRawSet map[int]*mat64.Dense, YhPlattSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, yPredSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, wg *sync.WaitGroup, mutex *sync.Mutex) {
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
	tsYhat, _ = SoftThresScale(tsYhat, rawThres)
	mutex.Lock()
	//beta for thres
	for p := 0; p < nCol; p++ {
		globalBeta.Set(c, p, globalBeta.At(c, p)+rawBeta.At(0, p))
	}
	//accum to add information for KNN calibaration
	AccumTsYdata(iFold, c, colSum, tsYhat, rawTsYhat, tsYfold, tsX, indAccum, YhRawSet, YhPlattSet, YhPlattSetCalibrated, yPlattSet, iFoldMarker, yPredSet, xSet, rawThres)
	mutex.Unlock()
}

func single_RecordMeasures(rankCut int, k int, lamda float64, c int, nLabel int, nKnn int, isKnn bool, trainMeasure *mat64.Dense, globalBeta *mat64.Dense, i int, yPlattSet map[int]*mat64.Dense, YhPlattSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPredSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	yPlattTrain, yPredTrain, xTrain, xTest, tsYhat, tsYfold := SubSetTrain(i, yPlattSet[c], YhPlattSet[c], yPredSet[c], xSet[c], iFoldMarker[c])
	//calculate platt scaled tsYhat again for measures
	//tmp comment out as yPlattSet record scalded data for now
	//_, nCol := tsYhat.Caps()
	//minMSElamda := make([]float64, nCol)
	//minMSE := make([]float64, nCol)
	//for p := 0; p < len(plattRobustLamda); p++ {
	//	for q := 0; q < nCol; q++ {
	//		if p == 0 {
	//			minMSE[q] = plattRobustMeasure[c].At(p, q)
	//			minMSElamda[q] = plattRobustLamda[p]
	//		} else {
	//			if plattRobustMeasure[c].At(p, q) < minMSE[q] {
	//				minMSE[q] = plattRobustMeasure[c].At(p, q)
	//				minMSElamda[q] = plattRobustLamda[p]
	//			}
	//		}
	//	}
	//}
	//tsYhat, _ = QuantileNorm(tsYhat, mat64.NewDense(0, 0, nil), false)
	//tsYhat, _, _ = Platt(tsYhat, tsYfold, tsYhat, minMSElamda)
	//beta := FscoreBeta(tsYfold, tsYhat)
	//rawThres := FscoreThres(tsYfold, tsYhat, beta)
	//tsYhat, rawThres = SoftThresScale(tsYhat, rawThres)

	//tmp thres for Report, as data reacaled
	rawThres := mat64.NewDense(1, nLabel, nil)
	for p := 0; p < nLabel; p++ {
		rawThres.Set(0, p, 0.5)
	}
	accuracy, microF1, microAupr, macroAupr, _, optScore, macroAuprSet := Report(tsYfold, tsYhat, rawThres, rankCut, false)
	mutex.Lock()
	trainMeasure.Set(c, 0, float64(k))
	trainMeasure.Set(c, 1, lamda)
	trainMeasure.Set(c, 2, trainMeasure.At(c, 2)+1.0)
	trainMeasure.Set(c, 3, trainMeasure.At(c, 3)+accuracy)
	trainMeasure.Set(c, 5, trainMeasure.At(c, 5)+microF1)
	trainMeasure.Set(c, 7, trainMeasure.At(c, 7)+microAupr)
	trainMeasure.Set(c, 9, trainMeasure.At(c, 9)+macroAupr)
	trainMeasure.Set(c, 11, trainMeasure.At(c, 11)+optScore)
	for p := 0; p < nLabel; p++ {
		idx := 11 + (p+1)*2
		trainMeasure.Set(c, idx, trainMeasure.At(c, idx)+macroAuprSet[p])
	}
	mutex.Unlock()
	//probability to be recalibrated for label dependency, subset train by fold
	if isKnn {
		//tsYhat = MultiLabelRecalibrate(nKnn, tsYhat, xTest, yPlattTrain, yPredTrain, xTrain, posLabelRls, negLabelRls, wg, mutex)
		tsYhat = MultiLabelRecalibrate_SingleThread(nKnn, tsYhat, xTest, yPlattTrain, yPredTrain, xTrain, posLabelRls, negLabelRls)
		//update MultiLabelRecalibrate tsYhat to YhPlattSetCalibrated
		YhPlattSetUpdate(i, c, YhPlattSetCalibrated, tsYhat, iFoldMarker[c])
		rawBeta := FscoreBeta(tsYfold, tsYhat)
		rawThres := FscoreThres(tsYfold, tsYhat, rawBeta)
		tsYhat, rawThres = SoftThresScale(tsYhat, rawThres)
		accuracy, microF1, microAupr, macroAupr, _, optScore, macroAuprSet = Report(tsYfold, tsYhat, rawThres, rankCut, false)
		mutex.Lock()
		for p := nLabel; p < nLabel*2; p++ {
			globalBeta.Set(c, p, globalBeta.At(c, p)+rawBeta.At(0, p-nLabel))
		}
		trainMeasure.Set(c, 4, trainMeasure.At(c, 4)+accuracy)
		trainMeasure.Set(c, 6, trainMeasure.At(c, 6)+microF1)
		trainMeasure.Set(c, 8, trainMeasure.At(c, 8)+microAupr)
		trainMeasure.Set(c, 10, trainMeasure.At(c, 10)+macroAupr)
		trainMeasure.Set(c, 12, trainMeasure.At(c, 12)+optScore)
		for p := 0; p < nLabel; p++ {
			idx := 12 + (p+1)*2
			trainMeasure.Set(c, idx, trainMeasure.At(c, idx)+macroAuprSet[p])
		}
		mutex.Unlock()
	}
}

func ancillaryByHyperParameterSet(cBestArr []int, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, YhRawSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense, globalBeta *mat64.Dense, nFold int, nLabel int, isPerLabel bool, isKnn bool) (thres *mat64.Dense, plattAB *mat64.Dense, YhPlattScale *mat64.Dense, thresKnn *mat64.Dense) {
	thres = mat64.NewDense(1, nLabel, nil)
	thresKnn = mat64.NewDense(1, nLabel, nil)
	plattAB = mat64.NewDense(2, nLabel, nil)
	betaValue := mat64.NewDense(1, nLabel, nil)
	betaValueKnn := mat64.NewDense(1, nLabel, nil)
	nRow, _ := YhRawSet[0].Caps()
	YhPlattScale = mat64.NewDense(nRow, nLabel, nil)
	//log
	if isPerLabel {
		log.Print("choose auprs for labels as object functions in tuning.")
	} else {
		log.Print("choose micro-aupr for all labels as object function in tuning.")
	}
	//thres and plattAB sets
	for i := 0; i < len(cBestArr); i++ {
		cBest := cBestArr[i]
		plattABtmp := mat64.NewDense(0, 0, nil)
		thresTmp := mat64.NewDense(0, 0, nil)
		thresKnnTmp := mat64.NewDense(0, 0, nil)
		YhPlattScaleTmp := mat64.NewDense(0, 0, nil)
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
		YhPlattScaleTmp, _ = QuantileNorm(YhRawSet[cBest], mat64.NewDense(0, 0, nil), false)
		YhPlattScaleTmp, plattABtmp, _ = Platt(YhPlattScaleTmp, yPlattSet[cBest], YhPlattScaleTmp, minMSElamda)
		//betaValue
		betaValueTmp := mat64.NewDense(1, nLabel, nil)
		betaValueKnnTmp := mat64.NewDense(1, nLabel, nil)
		//thresTmp for each cBest
		for p := 0; p < nLabel; p++ {
			betaValueTmp.Set(0, p, globalBeta.At(cBest, p)/float64(nFold))
		}
		thresTmp = FscoreThres(yPlattSet[cBest], YhPlattScaleTmp, betaValueTmp)
		if isKnn {
			log.Print("choose to use Knn.")
			for p := nLabel; p < nLabel*2; p++ {
				betaValueKnnTmp.Set(0, p-nLabel, globalBeta.At(cBest, p)/float64(nFold))
			}
			thresKnnTmp = FscoreThres(yPlattSet[cBest], YhPlattSetCalibrated[cBest], betaValueKnnTmp)
		} else {
			log.Print("choose to use raw score.")
		}
		//Soft scale for final training set prediction
		YhPlattScaleTmp, _ = SoftThresScale(YhPlattScaleTmp, thresTmp)
		//if not per label, first cBest is the only cBest
		if !isPerLabel {
			thres = thresTmp
			thresKnn = thresKnnTmp
			plattAB = plattABtmp
			betaValue = betaValueTmp
			betaValueKnn = betaValueKnnTmp
			YhPlattScale = YhPlattScaleTmp
			break
			//else set thres and plattAB for each label
		} else {
			thres.Set(0, i, thresTmp.At(0, i))
			thresKnn.Set(0, i, thresKnnTmp.At(0, i))
			plattAB.Set(0, i, plattABtmp.At(0, i))
			plattAB.Set(1, i, plattABtmp.At(1, i))
			betaValue.Set(0, i, betaValueTmp.At(0, i))
			betaValueKnn.Set(0, i, betaValueKnnTmp.At(0, i))
			for q := 0; q < nLabel; q++ {
				YhPlattScale.Set(q, i, YhPlattScaleTmp.At(q, i))
			}
		}
	}
	//log beta
	log.Print("choose these beta values per label for F-thresholding.")
	str := ""
	for p := 0; p < nLabel; p++ {
		s := fmt.Sprintf("%.3f", betaValue.At(0, p))
		str = str + "\t" + s
	}
	log.Print(str)
	if isKnn {
		log.Print("choose these after calibration beta values per label for F-thresholding.")
		str = ""
		for p := 0; p < nLabel; p++ {
			s := fmt.Sprintf("%.3f", betaValueKnn.At(0, p))
			str = str + "\t" + s
		}
		log.Print(str)
	}
	return thres, plattAB, YhPlattScale, thresKnn
}

func TuneAndPredict(nFold int, folds map[int][]int, fBetaThres float64, nK int, nKnn int, isPerLabel bool, isKnn bool, kSet []int, lamdaSet []float64, reg bool, rankCut int, trainFold []CvFold, testFold []CvFold, indAccum []int, tsXdata *mat64.Dense, tsYdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) (trainMeasureUpdated *mat64.Dense, testMeasureUpdated *mat64.Dense, tsYhat *mat64.Dense, thres *mat64.Dense, Yhat *mat64.Dense, YhatCalibrated *mat64.Dense, Ylabel *mat64.Dense) {
	//traing data per hyperparameter
	YhRawSet := make(map[int]*mat64.Dense)
	YhPlattSet := make(map[int]*mat64.Dense)
	YhPlattSetCalibrated := make(map[int]*mat64.Dense)
	yPlattSet := make(map[int]*mat64.Dense)
	yPredSet := make(map[int]*mat64.Dense)
	iFoldMarker := make(map[int]*mat64.Dense)
	xSet := make(map[int]*mat64.Dense)
	plattRobustMeasure := make(map[int]*mat64.Dense)
	plattRobustLamda := []float64{0.0, 0.04, 0.08, 0.12, 0.16, 0.2}
	_, nLabel := trYdata.Caps()
	globalBeta := mat64.NewDense(nK*len(lamdaSet), nLabel*2, nil)
	//subFolds
	subFolds := make(map[int]map[int][]int)
	//double check if kNN
	if nKnn <= 0 {
		isKnn = false
		log.Print("set kNN calibration to false as nKnn is not positive.")
	}
	//nested folds for training data
	wg.Add(nFold)
	for f := 0; f < nFold; f++ {
		go single_SOIS(subFolds, f, trainFold[f].Y, nFold, 10, 0, false, wg, mutex)
	}
	wg.Wait()
	log.Print("nested SOIS folds generated.")
	//measure matrix
	nL := nK * len(lamdaSet)
	trainMeasure := mat64.NewDense(nL, 14+nLabel*2, nil)
	testMeasure := mat64.NewDense(1, 8+nLabel, nil)

	for i := 0; i < nFold; i++ {
		YhSet, colSum := EcocRun(testFold[i].X, testFold[i].Y, trainFold[i].X, trainFold[i].Y, rankCut, reg, kSet, lamdaSet, nFold, subFolds[i], wg, mutex)
		tsYFold := PosSelect(testFold[i].Y, colSum)

		c := 0
		wg.Add(nK * len(lamdaSet))
		//accum calculated training data
		for m := 0; m < nK; m++ {
			for n := 0; n < len(lamdaSet); n++ {
				go single_AccumTsYdata(globalBeta, tsYFold, YhSet[c], i, c, colSum, testFold[i].X, testFold[i].IndAccum, YhRawSet, YhPlattSet, YhPlattSetCalibrated, yPlattSet, iFoldMarker, yPredSet, xSet, plattRobustMeasure, plattRobustLamda, wg, mutex)
				c += 1
			}
		}
		wg.Wait()
		log.Print("step 5: recalibration data obtained.")
	}
	//update all meassures before or after KNN calibration
	for i := 0; i < nFold; i++ {
		c := 0
		wg.Add(nK * len(lamdaSet))
		for m := 0; m < nK; m++ {
			for n := 0; n < len(lamdaSet); n++ {
				//trainMeasure, globalBeta recorded
				go single_RecordMeasures(rankCut, kSet[m], lamdaSet[n], c, nLabel, nKnn, isKnn, trainMeasure, globalBeta, i, yPlattSet, YhPlattSet, YhPlattSetCalibrated, yPredSet, xSet, iFoldMarker, plattRobustMeasure, plattRobustLamda, posLabelRls, negLabelRls, wg, mutex)
				c += 1
			}
		}
		wg.Wait()
	}
	log.Print("pass training.")

	//choosing object function, all hyper parameters, nDim in CCA, lamda and kNN calibration
	//the index value is 0-based, so that the same index in code above
	objectBaseNum := 11
	if isKnn {
		objectBaseNum = 12
	}
	cBest, _, _, _ := BestHyperParameterSetByMeasure(trainMeasure, objectBaseNum, nLabel, isPerLabel)
	thres, plattAB, YhPlattScale, thresKnn := ancillaryByHyperParameterSet(cBest, plattRobustMeasure, plattRobustLamda, YhRawSet, YhPlattSetCalibrated, yPlattSet, globalBeta, nFold, nLabel, isPerLabel, isKnn)
	//only the best hyperparameter set to use if not "isPerLabel"
	//thus the reset
	if !isPerLabel {
		kSet = []int{int(trainMeasure.At(cBest[0], 0))}
		lamdaSet = []float64{trainMeasure.At(cBest[0], 1)}
	}
	//best training parameters for testing
	YhSet, _ := EcocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, lamdaSet, nFold, folds, wg, mutex)
	//quantile normalization
	if isPerLabel {
		tsYhat = PerlLabelQuantileNorm(YhSet, cBest)
	} else {
		tsYhat, _ = QuantileNorm(YhSet[0], mat64.NewDense(0, 0, nil), false)
	}
	//scales
	tsYhat = PlattScaleSet(tsYhat, plattAB)
	tsYhat, thres = SoftThresScale(tsYhat, thres)
	//knn calibration
	if isKnn {
		tsXdata = RefillIndCol(tsXdata, indAccum)
		if isPerLabel {
			tsYhat = PerlLabelMultiLabelRecalibrate(cBest, nKnn, tsYhat, tsXdata, yPlattSet, yPredSet, xSet, posLabelRls, negLabelRls, wg, mutex)
		} else {
			tsYhat = MultiLabelRecalibrate(nKnn, tsYhat, tsXdata, yPlattSet[cBest[0]], yPredSet[cBest[0]], xSet[cBest[0]], posLabelRls, negLabelRls, wg, mutex)
		}
		tsYhat, thres = SoftThresScale(tsYhat, thresKnn)
	}
	//corresponding testing measures
	detectNanInf := NanFilter(tsYhat)
	if detectNanInf {
		log.Print("NaN or Inf detected in final prediction. Masked to zeros.")
	}
	accuracy, microF1, microAupr, macroAupr, agMicroF1, _, macroAuprSet := Report(tsYdata, tsYhat, thres, rankCut, false)
	if isPerLabel {
		testMeasure.Set(0, 0, 0.0)
		testMeasure.Set(0, 1, 0.0)
	} else {
		testMeasure.Set(0, 0, float64(kSet[0]))
		testMeasure.Set(0, 1, lamdaSet[0])
	}
	testMeasure.Set(0, 2, testMeasure.At(0, 2)+1.0)
	testMeasure.Set(0, 3, testMeasure.At(0, 3)+accuracy)
	testMeasure.Set(0, 4, testMeasure.At(0, 4)+microF1)
	testMeasure.Set(0, 5, testMeasure.At(0, 5)+microAupr)
	testMeasure.Set(0, 6, testMeasure.At(0, 6)+macroAupr)
	testMeasure.Set(0, 7, testMeasure.At(0, 7)+agMicroF1)
	for p := 0; p < nLabel; p++ {
		testMeasure.Set(0, 8+p, macroAuprSet[p])
	}
	if isPerLabel {
		YhPlattCalibrated, yPlatt := PerLabelBestCalibrated(cBest, YhPlattSetCalibrated, yPlattSet)
		return trainMeasure, testMeasure, tsYhat, thres, YhPlattScale, YhPlattCalibrated, yPlatt
	} else {
		return trainMeasure, testMeasure, tsYhat, thres, YhPlattScale, YhPlattSetCalibrated[cBest[0]], yPlattSet[cBest[0]]
	}
}

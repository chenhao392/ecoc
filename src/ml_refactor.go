package src

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"log"
	"sync"
)

func ReadNetworkPropagate(trRowName []string, tsRowName []string, trYdata *mat64.Dense, inNetworkFile []string, transLabels *mat64.Dense, isDada bool, alpha float64, threads int, wg *sync.WaitGroup, mutex *sync.Mutex) (tsXdata *mat64.Dense, trXdata *mat64.Dense, indAccum []int) {
	tsXdata = mat64.NewDense(0, 0, nil)
	trXdata = mat64.NewDense(0, 0, nil)
	// for filtering prior genes, only those in training set are used for propagation
	trGeneMap := make(map[string]int)
	for i := 0; i < len(trRowName); i++ {
		trGeneMap[trRowName[i]] = i
	}
	//network
	//network := mat64.NewDense(0, 0, nil)
	//idIdx := make(map[string]int)
	//idxToId := make(map[int]string)
	indAccum = make([]int, 0)
	//loading network gene ids
	//idIdx, idxToId, _ = IdIdxGen(inNetworkFile[0])
	//for i := 1; i < len(inNetworkFile); i++ {
	//	//idIdx as gene -> idx in net
	//	idIdxTmp, idxToIdTmp, _ := IdIdxGen(inNetworkFile[i])
	//	idIdx, idxToId = AccumIds(idIdxTmp, idxToIdTmp, idIdx, idxToId)
	//}
	//mean network for missing value
	//nGene := len(idIdx)
	//totalNet := mat64.NewDense(nGene, nGene, nil)
	//countNet := mat64.NewDense(nGene, nGene, nil)
	//for i := 0; i < len(inNetworkFile); i++ {
	//	log.Print("loading network file: ", inNetworkFile[i])
	//	totalNet, countNet = FillNetwork(inNetworkFile[i], idIdx, idxToId, totalNet, countNet)
	//}
	//meanNet = MeanNet(totalNet, countNet)
	//propagating networks
	log.Print("propagating networks.")
	for i := 0; i < len(inNetworkFile); i++ {
		//network := mat64.DenseCopyOf(meanNet)
		//network = UpdateNetwork(inNetworkFile[i], idIdx, idxToId, network)
		//network = ReadNetwork(inNetworkFile[i], idIdx, idxToId, network)
		//network, idIdx, idxToId := ReadNetwork(inNetworkFile[i])
		network, idIdx, _ := ReadNetwork(inNetworkFile[i])
		//sPriorData, ind := PropagateSetDLP(network, trYdata, idIdx, trRowName, trGeneMap, alpha, threads, wg)
		sPriorData, ind := PropagateSet(network, trYdata, idIdx, trRowName, trGeneMap, transLabels, isDada, alpha, wg, mutex)
		//sPriorData, ind := PropagateSet2D(network, trYdata, idIdx, trRowName, trGeneMap, transLabels, isDada, alpha, wg, mutex)
		//FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
		tsXdata, trXdata = FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
		indAccum = append(indAccum, ind...)
	}
	return tsXdata, trXdata, indAccum
}
func ReadNetworkPropagateCV(f int, folds map[int][]int, trRowName []string, tsRowName []string, trYdata *mat64.Dense, inNetworkFile []string, transLabels *mat64.Dense, isDada bool, alpha float64, threads int, wg *sync.WaitGroup, mutex *sync.Mutex) (cvTrain []int, cvTest []int, trXdataCV *mat64.Dense, indAccum []int) {
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
		//network, idIdx, idxToId := ReadNetwork(inNetworkFile[i])
		network, idIdx, _ := ReadNetwork(inNetworkFile[i])
		//network := mat64.DenseCopyOf(meanNet)
		//network = UpdateNetwork(inNetworkFile[i], idIdx, idxToId, network)

		//sPriorData, ind := PropagateSetDLP(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, alpha, threads, wg)
		sPriorData, ind := PropagateSet(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, transLabels, isDada, alpha, wg, mutex)
		//sPriorData, ind := PropagateSet2D(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, transLabels, isDada, alpha, wg, mutex)
		indAccum = append(indAccum, ind...)
		trXdataCV = FeatureDataStackCV(sPriorData, trRowName, idIdx, trXdataCV, trYdataCV, ind)
	}
	return cvTrain, cvTest, trXdataCV, indAccum
}

func single_AccumTsYdata(fBetaThres float64, isAutoBeta bool, globalBeta *mat64.Dense, tsYfold *mat64.Dense, rawTsYhat *mat64.Dense, iFold int, c int, colSum *mat64.Vector, tsX *mat64.Dense, indAccum []int, YhRawSet *map[int]*mat64.Dense, YhPlattSet *map[int]*mat64.Dense, YhPlattSetCalibrated *map[int]*mat64.Dense, yPlattSet *map[int]*mat64.Dense, iFoldMarker *map[int]*mat64.Dense, yPredSet *map[int]*mat64.Dense, xSet *map[int]*mat64.Dense, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	tsYhat := mat64.DenseCopyOf(rawTsYhat)
	nColGlobal := colSum.Len()
	_, nColFold := tsYfold.Caps()
	minMSElamda := make([]float64, nColFold)
	minMSE := make([]float64, nColFold)
	_, isDefinedMSE := plattRobustMeasure[c]
	mutex.Lock()
	if !isDefinedMSE {
		plattRobustMeasure[c] = mat64.NewDense(len(plattRobustLamda), nColGlobal, nil)
	}
	for p := 0; p < len(plattRobustLamda); p++ {
		tmpLamda := make([]float64, 0)
		//lamda per fold label
		for q := 0; q < nColFold; q++ {
			tmpLamda = append(tmpLamda, plattRobustLamda[p])
		}
		//mse per fold label
		_, _, mseArr := Platt(tsYhat, tsYfold, tsYhat, tmpLamda)
		//record mse and choose minMSElamda by minMSE
		qF := -1
		qG := 0
		for qG < nColGlobal {
			//qF inc if qG label exist in fold
			if colSum.At(qG, 0) == 1.0 {
				qF++
			}
			//in case first label missing, start updating when qF is updated
			if qF >= 0 {
				plattRobustMeasure[c].Set(p, qG, plattRobustMeasure[c].At(p, qG)+mseArr[qF])
				if p == 0 {
					minMSE[qF] = mseArr[qF]
					minMSElamda[qF] = plattRobustLamda[p]
				} else {
					if mseArr[qF] < minMSE[qF] {
						minMSE[qF] = mseArr[qF]
						minMSElamda[qF] = plattRobustLamda[p]
					}
				}
			}
			qG++
		}
	}
	mutex.Unlock()
	tsYhat, _, _ = Platt(tsYhat, tsYfold, tsYhat, minMSElamda)
	rawBeta := FscoreBeta(tsYfold, tsYhat, fBetaThres, isAutoBeta)
	rawThres := FscoreThres(tsYfold, tsYhat, rawBeta)
	tsYhat, _ = SoftThresScale(tsYhat, rawThres)
	mutex.Lock()
	//beta for thres
	qF := -1
	qG := 0
	for qG < nColGlobal {
		//qF inc if qG label exist in fold
		if colSum.At(qG, 0) == 1.0 {
			qF++
		}
		//in case first label missing, start updating when qF is updated
		if qF >= 0 {
			globalBeta.Set(c, qG, globalBeta.At(c, qG)+rawBeta.At(0, qG))
		}
		qG++
	}
	//accum to add information for KNN calibaration
	//YhRawSet, YhPlattSet, YhPlattSetCalibrated, yPlattSet, iFoldMarker, yPredSet, xSet = AccumTsYdata(iFold, c, colSum, tsYhat, rawTsYhat, tsYfold, tsX, indAccum, YhRawSet, YhPlattSet, YhPlattSetCalibrated, yPlattSet, iFoldMarker, yPredSet, xSet, rawThres)
	nRow, _ := tsYhat.Caps()
	tsYh2 := mat64.NewDense(nRow, nColGlobal, nil)
	rawTsYh2 := mat64.NewDense(nRow, nColGlobal, nil)
	tsYhCalib2 := mat64.NewDense(nRow, nColGlobal, nil)
	tsY2 := mat64.NewDense(nRow, nColGlobal, nil)
	predY2 := mat64.NewDense(nRow, nColGlobal, nil)
	//fold marker
	iFoldmat2 := mat64.NewDense(nRow, 1, nil)
	for i := 0; i < nRow; i++ {
		iFoldmat2.Set(i, 0, float64(iFold))
	}
	//tsYh, rawTsYh, tsY, and predY
	tC := 0
	for j := 0; j < nColGlobal; j++ {
		if colSum.At(j, 0) == 1.0 {
			for i := 0; i < nRow; i++ {
				tsYh2.Set(i, j, tsYhat.At(i, tC))
				rawTsYh2.Set(i, j, rawTsYhat.At(i, tC))
				tsY2.Set(i, j, tsYfold.At(i, tC))
				if tsYh2.At(i, j) > rawThres.At(0, j) {
					predY2.Set(i, j, 1.0)
				}
			}
			tC += 1
		} else {
			for i := 0; i < nRow; i++ {
				tsYh2.Set(i, j, -1.0)
				rawTsYh2.Set(i, j, -1.0)
				tsY2.Set(i, j, -1.0)
				predY2.Set(i, j, -1.0)
			}
		}
	}
	//is the matrix defined previously?
	_, isYh := (*YhPlattSet)[c]
	if !isYh {
		(*YhPlattSet)[c] = tsYh2
		(*YhRawSet)[c] = rawTsYh2
		(*YhPlattSetCalibrated)[c] = tsYhCalib2
		(*yPlattSet)[c] = tsY2
		(*xSet)[c] = tsX
		(*iFoldMarker)[c] = iFoldmat2
		(*yPredSet)[c] = predY2
	} else {
		newYh := mat64.NewDense(0, 0, nil)
		newRawYh := mat64.NewDense(0, 0, nil)
		newYhCalib := mat64.NewDense(0, 0, nil)
		newY := mat64.NewDense(0, 0, nil)
		newX := mat64.NewDense(0, 0, nil)
		newIfoldmat := mat64.NewDense(0, 0, nil)
		newPredY := mat64.NewDense(0, 0, nil)

		newYh.Stack((*YhPlattSet)[c], tsYh2)
		newRawYh.Stack((*YhRawSet)[c], rawTsYh2)
		newYhCalib.Stack((*YhPlattSetCalibrated)[c], tsYhCalib2)
		newY.Stack((*yPlattSet)[c], tsY2)
		newX.Stack((*xSet)[c], tsX)
		newIfoldmat.Stack((*iFoldMarker)[c], iFoldmat2)
		newPredY.Stack((*yPredSet)[c], predY2)

		(*YhPlattSet)[c] = newYh
		(*YhRawSet)[c] = newRawYh
		(*YhPlattSetCalibrated)[c] = newYhCalib
		(*yPlattSet)[c] = newY
		(*xSet)[c] = newX
		(*iFoldMarker)[c] = newIfoldmat
		(*yPredSet)[c] = newPredY
	}
	mutex.Unlock()
}

func single_RecordMeasures(rankCut int, k int, lamda float64, c int, nLabel int, nKnn int, isKnn bool, fBetaThres float64, isAutoBeta bool, trainMeasure *mat64.Dense, globalBeta *mat64.Dense, i int, yPlattSet map[int]*mat64.Dense, YhPlattSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPredSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//potential minus one in these mats, to be fixed as in single_AccumTsYdata
	yPlattTrain, yPredTrain, xTrain, xTest, tsYhat, tsYfold := SubSetTrain(i, yPlattSet[c], YhPlattSet[c], yPredSet[c], xSet[c], iFoldMarker[c])
	//tmp thres for Report, as data has been recaled
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
		idx := 13 + p*2
		trainMeasure.Set(c, idx, trainMeasure.At(c, idx)+macroAuprSet[p])
	}
	mutex.Unlock()
	//probability to be recalibrated for label dependency, subset train by fold
	if isKnn {
		kNNidx := TopKnnLabelIdx(macroAuprSet, 0.5)
		//tsYhat = MultiLabelRecalibrate(nKnn, tsYhat, xTest, yPlattTrain, yPredTrain, xTrain, posLabelRls, negLabelRls, wg, mutex)
		tsYhat = MultiLabelRecalibrate_SingleThread(nKnn, tsYhat, xTest, yPlattTrain, yPredTrain, xTrain, posLabelRls, negLabelRls, kNNidx)
		//update MultiLabelRecalibrate tsYhat to YhPlattSetCalibrated
		YhPlattSetUpdate(i, c, YhPlattSetCalibrated, tsYhat, iFoldMarker[c])
		rawBeta := FscoreBeta(tsYfold, tsYhat, fBetaThres, isAutoBeta)
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
			idx := 14 + p*2
			trainMeasure.Set(c, idx, trainMeasure.At(c, idx)+macroAuprSet[p])
		}
		mutex.Unlock()
	}
}

func ancillaryByHyperParameterSet(cBestArr []int, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, YhRawSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense, globalBeta *mat64.Dense, nFold int, nLabel int, isPerLabel bool, isKnn bool) (thres map[int]*mat64.Dense, plattAB map[int]*mat64.Dense, YhPlattScale *mat64.Dense, thresKnn *mat64.Dense) {
	nRow, _ := YhRawSet[0].Caps()
	thres = make(map[int]*mat64.Dense)
	plattAB = make(map[int]*mat64.Dense)
	YhPlattScale = mat64.NewDense(nRow, nLabel, nil)
	thresKnn = mat64.NewDense(1, nLabel, nil)
	betaValue := mat64.NewDense(1, nLabel, nil)
	betaValueKnn := mat64.NewDense(1, nLabel, nil)
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
		YhPlattScaleTmp, plattABtmp, _ = Platt(YhRawSet[cBest], yPlattSet[cBest], YhRawSet[cBest], minMSElamda)
		//betaValue and thres
		betaValueTmp := mat64.NewDense(1, nLabel, nil)
		betaValueKnnTmp := mat64.NewDense(1, nLabel, nil)
		for p := 0; p < nLabel; p++ {
			betaValueTmp.Set(0, p, globalBeta.At(cBest, p)/float64(nFold))
		}
		thresTmp = FscoreThres(yPlattSet[cBest], YhPlattScaleTmp, betaValueTmp)
		//LogColSum(thresTmp)
		if isKnn {
			for p := nLabel; p < nLabel*2; p++ {
				betaValueKnnTmp.Set(0, p-nLabel, globalBeta.At(cBest, p)/float64(nFold))
			}
			thresKnnTmp = FscoreThres(yPlattSet[cBest], YhPlattSetCalibrated[cBest], betaValueKnnTmp)
		}
		//Soft scale for final training set prediction
		YhPlattScaleTmp, _ = SoftThresScale(YhPlattScaleTmp, thresTmp)
		thres[i] = thresTmp
		plattAB[i] = plattABtmp
		if !isPerLabel {
			betaValue = betaValueTmp
			if isKnn {
				thresKnn = thresKnnTmp
				betaValueKnn = betaValueKnnTmp
			}
			YhPlattScale = YhPlattScaleTmp
			//if not per label, first cBest is the only cBest
			break
		} else {
			//else set thres and plattAB for each label
			betaValue.Set(0, i, betaValueTmp.At(0, i))
			if isKnn {
				thresKnn.Set(0, i, thresKnnTmp.At(0, i))
				betaValueKnn.Set(0, i, betaValueKnnTmp.At(0, i))
			}
			for q := 0; q < nRow; q++ {
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
	log.Print("choose these thres values per label for F-thresholding.")
	str = ""
	for p := 0; p < nLabel; p++ {
		if isPerLabel {
			s := fmt.Sprintf("%.3f", thres[p].At(0, p))
			str = str + "\t" + s
		} else {
			s := fmt.Sprintf("%.3f", thres[0].At(0, p))
			str = str + "\t" + s
		}
	}
	log.Print(str)
	return thres, plattAB, YhPlattScale, thresKnn
}

func TuneAndPredict(nFold int, folds map[int][]int, randValues []float64, fBetaThres float64, isAutoBeta bool, nK int, nKnn int, isPerLabel bool, isKnn bool, kSet []int, lamdaSet []float64, reg bool, rankCut int, trainFold []CvFold, testFold []CvFold, indAccum []int, tsXdata *mat64.Dense, tsYdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) (trainMeasureUpdated *mat64.Dense, testMeasureUpdated *mat64.Dense, tsYhat *mat64.Dense, thres *mat64.Dense, Yhat *mat64.Dense, YhatCalibrated *mat64.Dense, Ylabel *mat64.Dense) {
	//traing data per hyperparameter
	YhRawSet := make(map[int]*mat64.Dense)
	YhPlattSet := make(map[int]*mat64.Dense)
	YhPlattSetCalibrated := make(map[int]*mat64.Dense)
	yPlattSet := make(map[int]*mat64.Dense) //tsY
	yPredSet := make(map[int]*mat64.Dense)  //binTsY
	iFoldMarker := make(map[int]*mat64.Dense)
	xSet := make(map[int]*mat64.Dense) //tsX
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
	for f := 0; f < nFold; f++ {
		tmpFold := SOIS(trainFold[f].Y, nFold, 10, 0, false)
		subFolds[f] = tmpFold
	}
	log.Print("nested SOIS folds generated.")
	//measure matrix
	nL := nK * len(lamdaSet)
	trainMeasure := mat64.NewDense(nL, 13+nLabel*2, nil)
	testMeasure := mat64.NewDense(1, 8+nLabel, nil)

	for i := 0; i < nFold; i++ {
		idxToPrint := i + 1
		log.Print("starting fold ", idxToPrint, " of ", nFold, " folds.")
		YhSet, colSum := EcocRun(testFold[i].X, testFold[i].Y, trainFold[i].X, trainFold[i].Y, rankCut, reg, kSet, lamdaSet, nFold, subFolds[i], randValues, wg, mutex)
		tsYFold := PosSelect(testFold[i].Y, colSum)

		c := 0
		wg.Add(nK * len(lamdaSet))
		//accum calculated training data
		//plattRobustMeasure,beta,tsYh and etc calculated and recorded
		for m := 0; m < nK; m++ {
			for n := 0; n < len(lamdaSet); n++ {
				go single_AccumTsYdata(fBetaThres, isAutoBeta, globalBeta, tsYFold, YhSet[c], i, c, colSum, testFold[i].X, testFold[i].IndAccum, &YhRawSet, &YhPlattSet, &YhPlattSetCalibrated, &yPlattSet, &iFoldMarker, &yPredSet, &xSet, plattRobustMeasure, plattRobustLamda, wg, mutex)
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
				go single_RecordMeasures(rankCut, kSet[m], lamdaSet[n], c, nLabel, nKnn, isKnn, fBetaThres, isAutoBeta, trainMeasure, globalBeta, i, yPlattSet, YhPlattSet, YhPlattSetCalibrated, yPredSet, xSet, iFoldMarker, plattRobustMeasure, plattRobustLamda, posLabelRls, negLabelRls, wg, mutex)
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
	thresSet, plattABset, YhPlattScale, thresKnn := ancillaryByHyperParameterSet(cBest, plattRobustMeasure, plattRobustLamda, YhRawSet, YhPlattSetCalibrated, yPlattSet, globalBeta, nFold, nLabel, isPerLabel, isKnn)
	//only the best hyperparameter set to use if not "isPerLabel"
	//thus the reset for saving computational power
	if !isPerLabel {
		kSet = []int{int(trainMeasure.At(cBest[0], 0))}
		lamdaSet = []float64{trainMeasure.At(cBest[0], 1)}
	}
	//best training parameters for testing
	YhSet, _ := EcocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, lamdaSet, nFold, folds, randValues, wg, mutex)
	//scales
	//thres := mat64.NewDense(0, 0, nil)
	if isPerLabel {
		//platt and quantile
		tsYhat = PerLabelScaleSet(YhSet, plattABset, cBest)
		tsYhat, thres = PerLabelSoftThresScale(tsYhat, thresSet)
	} else {
		tsYhat = PlattScaleSet(YhSet[0], plattABset[0])
		tsYhat, thres = SoftThresScale(tsYhat, thresSet[0])
	}
	//knn calibration
	if isKnn {
		//tsXdata = RefillIndCol(tsXdata, indAccum)
		if isPerLabel {
			tsYhat = PerlLabelMultiLabelRecalibrate(YhSet, cBest, nKnn, trainMeasure, tsXdata, yPlattSet, yPredSet, xSet, plattABset, thresSet, posLabelRls, negLabelRls, wg, mutex)
		} else {
			thresData := mat64.NewDense(1, nLabel, nil)
			for i := 0; i < nLabel; i++ {
				thresData.Set(0, i, 0.5)
			}
			_, _, _, _, _, _, macroAuprSet := Report(yPlattSet[cBest[0]], YhPlattSet[cBest[0]], thresData, rankCut, false)
			kNNidx := TopKnnLabelIdx(macroAuprSet, 0.5)
			tsYhat = MultiLabelRecalibrate(nKnn, tsYhat, tsXdata, yPlattSet[cBest[0]], yPredSet[cBest[0]], xSet[cBest[0]], posLabelRls, negLabelRls, kNNidx, wg, mutex)
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

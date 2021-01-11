package src

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"log"
	"sync"
)

func ReadAndNormNetworks(inNetworkFile []string, modeIdx int, wg *sync.WaitGroup, mutex *sync.Mutex) (networkSet map[int]*mat64.Dense, idIdxSet map[int]map[string]int) {
	//network
	networkSet = make(map[int]*mat64.Dense)
	idIdxSet = make(map[int]map[string]int)
	//simple network
	if modeIdx == 1 {
		for i := 0; i < len(inNetworkFile); i++ {
			networkSet[i], idIdxSet[i], _ = ReadNetwork(inNetworkFile[i])
		}
	} else if modeIdx == 2 {
		//missing value filled by ave network
		idxToId := make(map[int]string)
		//loading network gene ids
		idIdx, idxToId, _ := IdIdxGen(inNetworkFile[0])
		for i := 1; i < len(inNetworkFile); i++ {
			//idIdx as gene -> idx in net
			idIdxTmp, idxToIdTmp, _ := IdIdxGen(inNetworkFile[i])
			idIdx, idxToId = AccumIds(idIdxTmp, idxToIdTmp, idIdx, idxToId)
		}
		//mean network for missing value
		nGene := len(idIdx)
		totalNet := mat64.NewDense(nGene, nGene, nil)
		countNet := mat64.NewDense(nGene, nGene, nil)
		for i := 0; i < len(inNetworkFile); i++ {
			log.Print("loading network file: ", inNetworkFile[i])
			totalNet, countNet = FillNetwork(inNetworkFile[i], idIdx, idxToId, totalNet, countNet)
		}
		meanNet := MeanNet(totalNet, countNet)
		for i := 0; i < len(inNetworkFile); i++ {
			network := mat64.DenseCopyOf(meanNet)
			networkSet[i] = UpdateNetwork(inNetworkFile[i], idIdx, idxToId, network)
			idIdxSet[i] = idIdx
		}
	}

	//normalize
	wg.Add(len(inNetworkFile))
	for i := 0; i < len(inNetworkFile); i++ {
		go single_normNet(i, networkSet, wg, mutex)
	}
	wg.Wait()
	return networkSet, idIdxSet
}

func single_normNet(i int, networkSet map[int]*mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//network, _ := colNorm(networkSet[i])
	//network = rwrNetwork(network, 0.5)
	network, _ := dNorm(networkSet[i])
	mutex.Lock()
	networkSet[i] = network
	mutex.Unlock()
}

func PropagateNetworks(trRowName []string, tsRowName []string, trYdata *mat64.Dense, networkSet map[int]*mat64.Dense, idIdxSet map[int]map[string]int, transLabels *mat64.Dense, isDada bool, alpha float64, threads int, wg *sync.WaitGroup, mutex *sync.Mutex) (tsXdata *mat64.Dense, trXdata *mat64.Dense, indAccum []int) {
	tsXdata = mat64.NewDense(0, 0, nil)
	trXdata = mat64.NewDense(0, 0, nil)
	// for filtering prior genes, only those in training set are used for propagation
	trGeneMap := make(map[string]int)
	for i := 0; i < len(trRowName); i++ {
		trGeneMap[trRowName[i]] = i
	}
	indAccum = make([]int, 0)
	//propagating networks
	log.Print("propagating networks.")
	for i := 0; i < len(networkSet); i++ {
		//sPriorData, ind := PropagateSetDLP(network, trYdata, idIdx, trRowName, trGeneMap, alpha, threads, wg)
		sPriorData, ind := PropagateSet(networkSet[i], trYdata, idIdxSet[i], trRowName, trGeneMap, transLabels, isDada, alpha, wg, mutex)
		//sPriorData, ind := PropagateSet2D(network, trYdata, idIdx, trRowName, trGeneMap, transLabels, isDada, alpha, wg, mutex)
		tsXdata, trXdata = FeatureDataStack(sPriorData, tsRowName, trRowName, idIdxSet[i], tsXdata, trXdata, trYdata, ind)
		indAccum = append(indAccum, ind...)
	}
	return tsXdata, trXdata, indAccum
}
func PropagateNetworksCV(f int, folds map[int][]int, trRowName []string, tsRowName []string, trYdata *mat64.Dense, networkSet map[int]*mat64.Dense, idIdxSet map[int]map[string]int, transLabels *mat64.Dense, isDada bool, alpha float64, threads int, wg *sync.WaitGroup, mutex *sync.Mutex) (cvTrain []int, cvTest []int, trXdataCV *mat64.Dense, indAccum []int) {
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
	for i := 0; i < len(networkSet); i++ {
		//idIdx as gene -> idx in net
		//sPriorData, ind := PropagateSetDLP(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, alpha, threads, wg)
		sPriorData, ind := PropagateSet(networkSet[i], trYdataCV, idIdxSet[i], trRowNameCV, trGeneMapCV, transLabels, isDada, alpha, wg, mutex)
		//sPriorData, ind := PropagateSet2D(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, transLabels, isDada, alpha, wg, mutex)
		indAccum = append(indAccum, ind...)
		trXdataCV = FeatureDataStackCV(sPriorData, trRowName, idIdxSet[i], trXdataCV, trYdataCV, ind)
	}
	return cvTrain, cvTest, trXdataCV, indAccum
}

func single_AccumTsYdata(objFuncIndex int, fBetaThres float64, isAutoBeta bool, globalBeta *mat64.Dense, tsYfold *mat64.Dense, rawTsYhat *mat64.Dense, iFold int, c int, colSum *mat64.Vector, tsX *mat64.Dense, indAccum []int, YhRawSet *map[int]*mat64.Dense, YhPlattSet *map[int]*mat64.Dense, YhPlattSetCalibrated *map[int]*mat64.Dense, yPlattSet *map[int]*mat64.Dense, iFoldMarker *map[int]*mat64.Dense, yPredSet *map[int]*mat64.Dense, xSet *map[int]*mat64.Dense, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	tsYhat := mat64.DenseCopyOf(rawTsYhat)
	nColGlobal := colSum.Len()
	_, nColFold := tsYfold.Caps()
	minMSElamda := make([]float64, nColFold)
	minMSE := make([]float64, nColFold)
	tmpPlattRobustMeasure := mat64.NewDense(len(plattRobustLamda), nColGlobal, nil)
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
				//plattRobustMeasure[c].Set(p, qG, plattRobustMeasure[c].At(p, qG)+mseArr[qF])
				tmpPlattRobustMeasure.Set(p, qG, tmpPlattRobustMeasure.At(p, qG)+mseArr[qF])
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
	//data processing, this data
	tsYhat, _, _ = Platt(tsYhat, tsYfold, tsYhat, minMSElamda)
	tsYhat, _ = QuantileNorm(tsYhat, mat64.NewDense(0, 0, nil), false)
	rawBeta := FscoreBeta(tsYfold, tsYhat, objFuncIndex, fBetaThres, isAutoBeta)
	rawThres := FscoreThres(tsYfold, tsYhat, rawBeta)
	tsYhat, _ = SoftThresScale(tsYhat, rawThres)
	//data processing, matching global data structure
	//accum to add information for KNN calibaration
	nRow, _ := tsYhat.Caps()
	tsYh2 := mat64.NewDense(nRow, nColGlobal, nil)
	rawTsYh2 := mat64.NewDense(nRow, nColGlobal, nil)
	tsYhCalib2 := mat64.NewDense(nRow, nColGlobal, nil)
	tsY2 := mat64.NewDense(nRow, nColGlobal, nil)
	predY2 := mat64.NewDense(nRow, nColGlobal, nil)
	//data processing, fold marker
	iFoldmat2 := mat64.NewDense(nRow, 1, nil)
	for i := 0; i < nRow; i++ {
		iFoldmat2.Set(i, 0, float64(iFold))
	}
	//data processing, tsYh, rawTsYh, tsY, and predY
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

	//lock and update
	mutex.Lock()
	_, isDefinedMSE := plattRobustMeasure[c]
	if !isDefinedMSE {
		plattRobustMeasure[c] = tmpPlattRobustMeasure
	}
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
			globalBeta.Set(c, qG, globalBeta.At(c, qG)+rawBeta.At(0, qF))
			globalBeta.Set(c, qG+nColGlobal, globalBeta.At(c, qG+nColGlobal)+1.0)
		}
		qG++
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

func single_RecordMeasures(objFuncIndex int, rankCut int, k int, lamda float64, c int, nLabel int, nKnn int, isKnn bool, fBetaThres float64, isAutoBeta bool, trainMeasure *mat64.Dense, globalBeta *mat64.Dense, i int, yPlattSet map[int]*mat64.Dense, YhPlattSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPredSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//potential minus one in these mats, to be fixed as in single_AccumTsYdata
	yPlattTrain, yPredTrain, xTrain, xTest, tsYhat, tsYfold, indMinus := SubSetTrain(i, yPlattSet[c], YhPlattSet[c], yPredSet[c], xSet[c], iFoldMarker[c])
	//tmp thres for Report, as data has been recaled
	rawThres := mat64.NewDense(1, nLabel, nil)
	for p := 0; p < nLabel; p++ {
		rawThres.Set(0, p, 0.5)
	}
	accuracy, microF1, microAupr, macroAupr, _, optScore, macroAuprSet := Report(objFuncIndex, tsYfold, tsYhat, rawThres, rankCut, false)
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
		_, isMinus := indMinus[p]
		if !isMinus {
			trainMeasure.Set(c, idx, trainMeasure.At(c, idx)+macroAuprSet[p])
			trainMeasure.Set(c, idx+1, trainMeasure.At(c, idx+1)+1.0)
		}
	}
	mutex.Unlock()
	//probability to be recalibrated for label dependency, subset train by fold
	if isKnn {
		kNNidx := TopKnnLabelIdx(macroAuprSet, 0.5)
		//tsYhat = MultiLabelRecalibrate(nKnn, tsYhat, xTest, yPlattTrain, yPredTrain, xTrain, posLabelRls, negLabelRls, wg, mutex)
		tsYhat = MultiLabelRecalibrate_SingleThread(nKnn, tsYhat, xTest, yPlattTrain, yPredTrain, xTrain, posLabelRls, negLabelRls, kNNidx)
		//update MultiLabelRecalibrate tsYhat to YhPlattSetCalibrated
		YhPlattSetUpdate(i, c, YhPlattSetCalibrated, tsYhat, iFoldMarker[c])
		rawBeta := FscoreBeta(tsYfold, tsYhat, objFuncIndex, fBetaThres, isAutoBeta)
		rawThres := FscoreThres(tsYfold, tsYhat, rawBeta)
		tsYhat, rawThres = SoftThresScale(tsYhat, rawThres)
		accuracy, microF1, microAupr, macroAupr, _, optScore, macroAuprSet = Report(objFuncIndex, tsYfold, tsYhat, rawThres, rankCut, false)
		mutex.Lock()
		trainMeasure.Set(c, 4, trainMeasure.At(c, 4)+accuracy)
		trainMeasure.Set(c, 6, trainMeasure.At(c, 6)+microF1)
		trainMeasure.Set(c, 8, trainMeasure.At(c, 8)+microAupr)
		trainMeasure.Set(c, 10, trainMeasure.At(c, 10)+macroAupr)
		trainMeasure.Set(c, 12, trainMeasure.At(c, 12)+optScore)
		mutex.Unlock()
	}
}

func ancillaryByHyperParameterSet(objFuncIndex int, cBestArr []int, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, YhRawSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, globalBeta *mat64.Dense, nFold int, nLabel int, nKnn int, isPerLabel bool, isKnn bool, wg *sync.WaitGroup, mutex *sync.Mutex) (thres *mat64.Dense, plattAB *mat64.Dense, YhPlattScale *mat64.Dense, thresKnn *mat64.Dense, YhPlattCalibrated *mat64.Dense, Y *mat64.Dense, kNNidx map[int]int) {
	//init
	nRow, _ := YhRawSet[0].Caps()
	thres = mat64.NewDense(0, 0, nil)
	thresKnn = mat64.NewDense(1, nLabel, nil)
	plattAB = mat64.NewDense(0, 0, nil)
	YhPlattScale = mat64.NewDense(nRow, nLabel, nil)
	YhPlattCalibrated = mat64.NewDense(nRow, nLabel, nil)
	Y = mat64.NewDense(0, 0, nil)
	betaValue := mat64.NewDense(1, nLabel, nil)
	betaValueKnn := mat64.NewDense(1, nLabel, nil)
	//thres and plattAB sets
	if !isPerLabel {
		cBest := cBestArr[0]
		minMSElamda := getMinMSElamda(cBest, plattRobustMeasure, plattRobustLamda)
		YhPlattScale, plattAB, _ = Platt(YhRawSet[cBest], yPlattSet[cBest], YhRawSet[cBest], minMSElamda)
		YhPlattScale, _ = QuantileNorm(YhPlattScale, mat64.NewDense(0, 0, nil), false)
		//betaValue and thres
		for p := 0; p < nLabel; p++ {
			betaValue.Set(0, p, globalBeta.At(cBest, p)/globalBeta.At(cBest, p+nLabel))
		}
		thres = FscoreThres(yPlattSet[cBest], YhPlattScale, betaValue)
	} else {
		YhPlattScale, plattAB, betaValue, thres, Y = PerLabelBetaEst(objFuncIndex, cBestArr, plattRobustMeasure, plattRobustLamda, YhRawSet, yPlattSet, iFoldMarker, nFold)
	}
	YhPlattScale, _ = SoftThresScale(YhPlattScale, thres)
	//additional knn steps
	if isKnn {
		//top aupr label for calibration
		_, _, _, _, _, _, macroAuprSet := Report(objFuncIndex, Y, YhPlattScale, thres, 1, false)
		kNNidx := TopKnnLabelIdx(macroAuprSet, 0.5)
		//get yPred for calibration
		yPred := mat64.NewDense(nRow, nLabel, nil)
		for i := 0; i < nRow; i++ {
			for j := 0; j < nLabel; j++ {
				if YhPlattScale.At(i, j) >= 0.5 {
					yPred.Set(i, j, 1.0)
				}
			}
		}
		//get X for calibration
		X := mat64.NewDense(nRow, nLabel, nil)
		for j := 0; j < len(cBestArr); j++ {
			cBest := cBestArr[j]
			for i := 0; i < nRow; i++ {
				if xSet[cBest].At(i, j) >= 0.0 {
					X.Set(i, j, xSet[cBest].At(i, j))
				}
			}
		}

		//split yPred, X, Y and Yh to folds and estimate betas
		betaValueCount := mat64.NewDense(1, nLabel, nil)
		reorderY := mat64.NewDense(nRow, nLabel, nil)
		iRow := 0

		for i := 0; i < nFold; i++ {
			yTrain, yPredTrain, xTrain, xTest, tsYhat, tsYfold, indMinus := SubSetTrain(i, Y, YhPlattScale, yPred, X, iFoldMarker[0])
			subYhPlattCalibrated := MultiLabelRecalibrate(nKnn, tsYhat, xTest, yTrain, yPredTrain, xTrain, posLabelRls, negLabelRls, kNNidx, wg, mutex)
			//accum calibated data
			nSubRow, _ := subYhPlattCalibrated.Caps()
			for m := 0; m < nSubRow; m++ {
				for n := 0; n < nLabel; n++ {
					YhPlattCalibrated.Set(iRow, n, subYhPlattCalibrated.At(m, n))
					reorderY.Set(iRow, n, tsYfold.At(m, n))
				}
				iRow += 1
			}
			subYhPlattCalibrated, _ = QuantileNorm(subYhPlattCalibrated, mat64.NewDense(0, 0, nil), false)
			subBetaValue := FscoreBeta(tsYfold, subYhPlattCalibrated, objFuncIndex, 1.0, true)
			for j := 0; j < nLabel; j++ {
				_, isMinus := indMinus[j]
				if !isMinus {
					betaValueKnn.Set(0, j, betaValueKnn.At(0, j)+subBetaValue.At(0, j))
					betaValueCount.Set(0, j, betaValueCount.At(0, j)+1.0)
				}
			}

		}
		//ave beta
		for j := 0; j < nLabel; j++ {
			if betaValueKnn.At(0, j) > 0.0 {
				betaValueKnn.Set(0, j, betaValueKnn.At(0, j)/betaValueCount.At(0, j))
			}
		}

		YhPlattCalibrated, _ = QuantileNorm(YhPlattCalibrated, mat64.NewDense(0, 0, nil), false)
		thresKnn = FscoreThres(reorderY, YhPlattCalibrated, betaValueKnn)
	}

	//log beta
	log.Print("choose these log10(beta) values per label for F-thresholding.")
	str := ""
	for p := 0; p < nLabel; p++ {
		s := fmt.Sprintf("%.3f", betaValue.At(0, p))
		str = str + "\t" + s
	}
	log.Print(str)
	log.Print("choose these thres values per label for F-thresholding.")
	str = ""
	for p := 0; p < nLabel; p++ {
		s := fmt.Sprintf("%.3f", thres.At(0, p))
		str = str + "\t" + s
	}
	log.Print(str)
	if isKnn {
		log.Print("choose these after calibration log10(beta) values per label for F-thresholding.")
		str = ""
		for p := 0; p < nLabel; p++ {
			s := fmt.Sprintf("%.3f", betaValueKnn.At(0, p))
			str = str + "\t" + s
		}
		log.Print(str)
		log.Print("choose these after calibration thres values per label for F-thresholding.")
		str = ""
		for p := 0; p < nLabel; p++ {
			s := fmt.Sprintf("%.3f", thresKnn.At(0, p))
			str = str + "\t" + s
		}
		log.Print(str)
	}
	return thres, plattAB, YhPlattScale, thresKnn, YhPlattCalibrated, Y, kNNidx
}

func TuneAndPredict(objFuncIndex int, nFold int, folds map[int][]int, randValues []float64, fBetaThres float64, isAutoBeta bool, nK int, nKnn int, isPerLabel bool, isKnn bool, kSet []int, lamdaSet []float64, reg bool, rankCut int, trainFold []CvFold, testFold []CvFold, indAccum []int, tsXdata *mat64.Dense, tsYdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) (trainMeasureUpdated *mat64.Dense, testMeasureUpdated *mat64.Dense, tsYhat *mat64.Dense, Yhat *mat64.Dense, YhatCalibrated *mat64.Dense, Ylabel *mat64.Dense) {
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
				go single_AccumTsYdata(objFuncIndex, fBetaThres, isAutoBeta, globalBeta, tsYFold, YhSet[c], i, c, colSum, testFold[i].X, testFold[i].IndAccum, &YhRawSet, &YhPlattSet, &YhPlattSetCalibrated, &yPlattSet, &iFoldMarker, &yPredSet, &xSet, plattRobustMeasure, plattRobustLamda, wg, mutex)
				c += 1
			}
		}
		log.Print("step 5: recalibration data obtained.")
	}
	//update all meassures before or after KNN calibration
	for i := 0; i < nFold; i++ {
		c := 0
		wg.Add(nK * len(lamdaSet))
		for m := 0; m < nK; m++ {
			for n := 0; n < len(lamdaSet); n++ {
				//trainMeasure, globalBeta recorded
				go single_RecordMeasures(objFuncIndex, rankCut, kSet[m], lamdaSet[n], c, nLabel, nKnn, isKnn, fBetaThres, isAutoBeta, trainMeasure, globalBeta, i, yPlattSet, YhPlattSet, YhPlattSetCalibrated, yPredSet, xSet, iFoldMarker, plattRobustMeasure, plattRobustLamda, posLabelRls, negLabelRls, wg, mutex)
				c += 1
			}
		}
		wg.Wait()
	}
	log.Print("pass training.")

	//choosing object function, all hyper parameters, nDim in CCA, lamda and kNN calibration
	//the index value is 0-based, so that the same index in code above
	objectBaseNum := 11
	cBest, _, _, _ := BestHyperParameterSetByMeasure(trainMeasure, objectBaseNum, nLabel, isPerLabel, isKnn)
	thres, plattAB, YhPlattScale, thresKnn, YhPlattCalibrated, Y, kNNidx := ancillaryByHyperParameterSet(objFuncIndex, cBest, plattRobustMeasure, plattRobustLamda, YhRawSet, YhPlattSetCalibrated, yPlattSet, xSet, iFoldMarker, posLabelRls, negLabelRls, globalBeta, nFold, nLabel, nKnn, isPerLabel, isKnn, wg, mutex)
	//only the best hyperparameter set to use if not "isPerLabel"
	//thus the reset for saving computational power
	if !isPerLabel {
		kSet = []int{int(trainMeasure.At(cBest[0], 0))}
		lamdaSet = []float64{trainMeasure.At(cBest[0], 1)}
	}
	//best training parameters for testing
	YhSet, _ := EcocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, lamdaSet, nFold, folds, randValues, wg, mutex)
	//scales
	if isPerLabel {
		//platt and quantile
		tsYhat = PerLabelScaleSet(YhSet, plattAB, cBest)
	} else {
		tsYhat = PlattScaleSet(YhSet[0], plattAB)
		tsYhat, _ = QuantileNorm(tsYhat, mat64.NewDense(0, 0, nil), false)
	}
	tsYhat, _ = SoftThresScale(tsYhat, thres)
	//knn calibration
	if isKnn {
		//tsXdata = RefillIndCol(tsXdata, indAccum)
		//get yPred for calibration
		nRow, _ := trYdata.Caps()
		trYhat, _ := SoftThresScale(trYdata, thres)
		//yPred
		yPred := mat64.NewDense(nRow, nLabel, nil)
		for i := 0; i < nRow; i++ {
			for j := 0; j < nLabel; j++ {
				if trYhat.At(i, j) >= 0.5 {
					yPred.Set(i, j, 1.0)
				}
			}
		}
		tsYhat = MultiLabelRecalibrate(nKnn, tsYhat, tsXdata, trYdata, yPred, trXdata, posLabelRls, negLabelRls, kNNidx, wg, mutex)
		tsYhat, _ = QuantileNorm(tsYhat, mat64.NewDense(0, 0, nil), false)
		tsYhat, thres = SoftThresScale(tsYhat, thresKnn)
	}
	//corresponding testing measures
	detectNanInf := NanFilter(tsYhat)
	if detectNanInf {
		log.Print("NaN or Inf detected in final prediction. Masked to zeros.")
	}
	accuracy, microF1, microAupr, macroAupr, agMicroF1, _, macroAuprSet := Report(objFuncIndex, tsYdata, tsYhat, thres, rankCut, false)
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
		return trainMeasure, testMeasure, tsYhat, YhPlattScale, YhPlattCalibrated, Y
	} else {
		return trainMeasure, testMeasure, tsYhat, YhPlattScale, YhPlattSetCalibrated[cBest[0]], yPlattSet[cBest[0]]
	}
}

func getMinMSElamda(cBest int, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64) []float64 {
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
	return minMSElamda
}

func PerLabelBetaEst(objFuncIndex int, cBestArr []int, plattRobustMeasure map[int]*mat64.Dense, plattRobustLamda []float64, YhRawSet map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, nFold int) (YhPlattScale *mat64.Dense, plattAB *mat64.Dense, betaValue *mat64.Dense, thres *mat64.Dense, Y *mat64.Dense) {
	nRow, nLabel := YhRawSet[0].Caps()
	YhPlattScale = mat64.NewDense(nRow, nLabel, nil)
	YhPlattScaleTmp := mat64.NewDense(0, 0, nil)
	Y = mat64.NewDense(nRow, nLabel, nil)
	plattAB = mat64.NewDense(2, nLabel, nil)
	plattABtmp := mat64.NewDense(0, 0, nil)
	betaValue = mat64.NewDense(1, nLabel, nil)
	betaValueCount := mat64.NewDense(1, nLabel, nil)
	for j := 0; j < len(cBestArr); j++ {
		cBest := cBestArr[j]
		minMSElamda := getMinMSElamda(cBest, plattRobustMeasure, plattRobustLamda)
		YhPlattScaleTmp, plattABtmp, _ = Platt(YhRawSet[cBest], yPlattSet[cBest], YhRawSet[cBest], minMSElamda)
		plattAB.Set(0, j, plattABtmp.At(0, j))
		plattAB.Set(1, j, plattABtmp.At(1, j))

		for i := 0; i < nRow; i++ {
			YhPlattScale.Set(i, j, YhPlattScaleTmp.At(i, j))
			if yPlattSet[cBest].At(i, j) == 1.0 {
				Y.Set(i, j, yPlattSet[cBest].At(i, j))
			}
		}
	}
	//split Yh and Y to folds and estimate betas
	for i := 0; i < nFold; i++ {
		subYhPlattScale, subY, indMinus := SubSetYs(i, YhPlattScale, Y, iFoldMarker[0])
		subYhPlattScale, _ = QuantileNorm(subYhPlattScale, mat64.NewDense(0, 0, nil), false)
		subBetaValue := FscoreBeta(subY, subYhPlattScale, objFuncIndex, 1.0, true)
		for j := 0; j < nLabel; j++ {
			_, isMinus := indMinus[j]
			if !isMinus {
				betaValue.Set(0, j, betaValue.At(0, j)+subBetaValue.At(0, j))
				betaValueCount.Set(0, j, betaValueCount.At(0, j)+1.0)
			}
		}
	}
	//ave beta
	for j := 0; j < nLabel; j++ {
		if betaValueCount.At(0, j) > 0.0 {
			betaValue.Set(0, j, betaValue.At(0, j)/betaValueCount.At(0, j))
		}
	}
	YhPlattScale, _ = QuantileNorm(YhPlattScale, mat64.NewDense(0, 0, nil), false)
	thres = FscoreThres(Y, YhPlattScale, betaValue)
	return YhPlattScale, plattAB, betaValue, thres, Y
}

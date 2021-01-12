package src

import (
	"github.com/gonum/matrix/mat64"
	"log"
	"math"
	//"os"
	//"fmt"
	"runtime"
	"sort"
	"sync"
)

func FeatureDataStack(sPriorData *mat64.Dense, tsRowName []string, trRowName []string, idIdx map[string]int, tsXdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, ind []int) (*mat64.Dense, *mat64.Dense) {
	_, nLabel := sPriorData.Caps()
	//nTrLabel might be more than nLabel as it can be filterred
	_, nTrLabel := trYdata.Caps()
	tmpTsXdata := mat64.NewDense(len(tsRowName), nLabel, nil)
	tmpTrXdata := mat64.NewDense(len(trRowName), nLabel, nil)
	cLabel := 0
	//trX
	for l := 0; l < nTrLabel; l++ {
		if ind[l] > 1 {
			for k := 0; k < len(trRowName); k++ {
				_, exist := idIdx[trRowName[k]]
				if exist {
					tmpTrXdata.Set(k, cLabel, sPriorData.At(idIdx[trRowName[k]], cLabel))
				}
			}
			cLabel += 1
		}
	}
	//tsX
	cLabel = 0
	for l := 0; l < nTrLabel; l++ {
		if ind[l] > 1 {
			for k := 0; k < len(tsRowName); k++ {
				_, exist := idIdx[tsRowName[k]]
				if exist {
					tmpTsXdata.Set(k, cLabel, sPriorData.At(idIdx[tsRowName[k]], cLabel))
				}
			}
			cLabel += 1

		}
	}
	nRow, _ := trXdata.Caps()
	if nRow == 0 {
		tsXdata = tmpTsXdata
		trXdata = tmpTrXdata
	} else {
		tsXdata = ColStackMatrix(tsXdata, tmpTsXdata)
		trXdata = ColStackMatrix(trXdata, tmpTrXdata)
	}
	return tsXdata, trXdata
}

func FeatureDataStackCV(sPriorData *mat64.Dense, trRowName []string, idIdx map[string]int, trXdataCV *mat64.Dense, trYdataCV *mat64.Dense, ind []int) *mat64.Dense {

	//feature stack
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
					//tmpTrXdata.Set(k, cLabel, sPriorData.At(idIdx[trRowName[k]], cLabel)/float64(ind[l]))
					tmpTrXdata.Set(k, cLabel, sPriorData.At(idIdx[trRowName[k]], cLabel))
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
	return trXdataCV
}

func NpAlphaEstimate(folds map[int][]int, trRowName []string, tsRowName []string, trYdata *mat64.Dense, networkSet map[int]*mat64.Dense, idIdxSet map[int]map[string]int, transLabels *mat64.Dense, isDada bool, threads int, wg *sync.WaitGroup, mutex *sync.Mutex) []float64 {
	//alphaArr := []float64{0.5, 0.4, 0.6, 0.7, 0.3, 0.2, 0.8, 0.1, 0.9}
	alphaArr := []float64{0.5, 0.2, 0.8}
	_, nLabel := trYdata.Caps()
	aMatrix := mat64.NewDense(len(alphaArr), nLabel*len(networkSet), nil)
	aSet := make([]float64, nLabel*len(networkSet))
	for i := 0; i < len(alphaArr); i++ {
		//init alpha per label
		alphaSet := make([]float64, nLabel*len(networkSet))
		for j := 0; j < nLabel*len(networkSet); j++ {
			alphaSet[j] = alphaArr[i]
		}
		//calculate aupr
		for f := 0; f < len(folds); f++ {
			_, _, _, _, auprSet := PropagateNetworksCV(f, folds, trRowName, tsRowName, trYdata, networkSet, idIdxSet, transLabels, isDada, alphaSet, threads, wg, mutex)
			//log.Print("fold,alpha: ", f, alphaArr[i])
			//str := ""
			//for j := 0; j < nLabel*len(networkSet); j++ {
			//	str = str + fmt.Sprintf("\t%.2f", auprSet[j])
			//}
			//log.Print(str)
			//accum aupr
			for j := 0; j < nLabel*len(networkSet); j++ {
				aMatrix.Set(i, j, aMatrix.At(i, j)+auprSet[j])
			}
		}
	}
	//best alpha
	for j := 0; j < nLabel*len(networkSet); j++ {
		var sortMap []kv
		for i := 0; i < len(alphaArr); i++ {
			sortMap = append(sortMap, kv{i, aMatrix.At(i, j)})
		}
		sort.Slice(sortMap, func(p, q int) bool {
			return sortMap[p].Value > sortMap[q].Value
		})
		maxThres := 0.99 * sortMap[0].Value
		for i := 0; i < len(alphaArr); i++ {
			if aMatrix.At(i, j) >= maxThres {
				aSet[j] = alphaArr[i]
				//log.Print("choose idx: ", i, " and alpha: ", aSet[j], " with aupr: ", aMatrix.At(i, j))
				break
			}
		}

		//str := ""
		//for i := 0; i < len(alphaArr); i++ {
		//	str = str + fmt.Sprintf("\t%d|%.2f", sortMap[i].Key, sortMap[i].Value)
		//}
		//log.Print(str)
	}
	return aSet
}

func NPauprSet(tsYdataCV *mat64.Dense, sPriorData *mat64.Dense, tsRowName []string, idIdx map[string]int, ind []int) (auprSet []float64) {
	//feature stack
	_, nTrLabel := tsYdataCV.Caps()
	auprSet = make([]float64, nTrLabel)
	//tsX
	cLabel := 0
	for l := 0; l < nTrLabel; l++ {
		if ind[l] > 1 {
			tmpArr := make([]float64, len(tsRowName))
			for k := 0; k < len(tsRowName); k++ {
				_, exist := idIdx[tsRowName[k]]
				if exist {
					//tmpTsXVec.Set(k, 0, sPriorData.At(idIdx[tsRowName[k]], cLabel))
					tmpArr[k] = sPriorData.At(idIdx[tsRowName[k]], cLabel)
				}
			}
			tmpTsXVec := mat64.NewVector(len(tsRowName), tmpArr)
			tAupr, _, _, _ := ComputeAupr(tsYdataCV.ColView(l), tmpTsXVec, 1.0)
			auprSet[l] = auprSet[l] + tAupr
			cLabel += 1
		}
	}
	return auprSet
}

func PropagateSet(network *mat64.Dense, trYdata *mat64.Dense, idIdx map[string]int, idArr []string, trGeneMap map[string]int, transLabels *mat64.Dense, isDada bool, alphaSet []float64, wg *sync.WaitGroup, mutex *sync.Mutex) (sPriorData *mat64.Dense, ind []int) {
	nTrGene, nTrLabel := trYdata.Caps()
	nNetworkGene, _ := network.Caps()
	//ind for prior/label gene set mapping at least one gene to the network
	ind = make([]int, nTrLabel)
	for j := 0; j < nTrLabel; j++ {
		inGene := make([]int, 0)
		for i := 0; i < nTrGene; i++ {
			_, exist := idIdx[idArr[i]]
			_, existTr := trGeneMap[idArr[i]]
			if trYdata.At(i, j) > 0 && exist && existTr {
				ind[j] += 1
				inGene = append(inGene, idIdx[idArr[i]])
			}
		}
	}
	nOutLabel := 0
	for i := 0; i < nTrLabel; i++ {
		if ind[i] > 1 {
			nOutLabel += 1
		}
	}
	//propagate Set by label, smoothed prior data
	sPriorData = mat64.NewDense(nNetworkGene, nOutLabel, nil)
	c := 0
	wg.Add(nOutLabel)
	for j := 0; j < nTrLabel; j++ {
		if ind[j] > 1 {
			trY := mat64.NewDense(nNetworkGene, 1, nil)
			for i := 0; i < nTrGene; i++ {
				_, exist := idIdx[idArr[i]]
				_, existTr := trGeneMap[idArr[i]]
				if trYdata.At(i, j) > 0 && exist && existTr {
					trY.Set(idIdx[idArr[i]], 0, trYdata.At(i, j)/float64(ind[j]))
				}
			}
			prior := mat64.DenseCopyOf(trY)
			if isDada {
				go single_sPriorDataDada(network, sPriorData, prior, trY, nNetworkGene, alphaSet[j], c, wg, mutex)
			} else {
				go single_sPriorData(network, sPriorData, prior, trY, nNetworkGene, 100, alphaSet[j], c, wg, mutex)
			}
			c += 1
		}
	}
	wg.Wait()
	runtime.GC()
	//label by label smooth
	//wg.Add(nNetworkGene)
	//for j := 0; j < nNetworkGene; j++ {
	//	go single_LabelGraphSmooth(j, ind, sPriorData, transLabels, 100, wg, mutex)
	//}
	//wg.Wait()
	//colum norm to max 1
	//for j := 0; j < nOutLabel; j++ {
	//	max := 0.0
	//	for i := 0; i < nNetworkGene; i++ {
	//		if sPriorData.At(i, j) > max {
	//			max = sPriorData.At(i, j)
	//		}
	//	}
	//	for i := 0; i < nNetworkGene; i++ {
	//		sPriorData.Set(i, j, sPriorData.At(i, j)/max)
	//	}
	//}

	return sPriorData, ind
}

func single_sPriorData(network *mat64.Dense, sPriorData *mat64.Dense, prior *mat64.Dense, trY *mat64.Dense, nNetworkGene int, maxItr int, alpha float64, c int, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//n by 1 matrix
	sPrior1 := propagate(network, alpha, prior, trY, maxItr)
	mutex.Lock()
	for i := 0; i < nNetworkGene; i++ {
		sPriorData.Set(i, c, sPrior1.At(i, 0))
	}
	mutex.Unlock()
}
func single_sPriorDataDada(network *mat64.Dense, sPriorData *mat64.Dense, prior *mat64.Dense, trY *mat64.Dense, nNetworkGene int, alpha float64, c int, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//n by 1 matrix
	sPrior1 := propagate(network, alpha, prior, trY, 100)
	sPrior2 := propagate(network, 1.0, prior, trY, 100)
	max := 0.0
	for i := 0; i < nNetworkGene; i++ {
		value := sPrior1.At(i, 0) / sPrior2.At(i, 0)
		if !math.IsInf(value, 1) && !math.IsNaN(value) {
			if max < value {
				max = value
			}
		}
	}
	mutex.Lock()
	for i := 0; i < nNetworkGene; i++ {
		value := sPrior1.At(i, 0) / sPrior2.At(i, 0)
		if trY.At(i, 0) == 1.0 {
			sPriorData.Set(i, c, 1.0)
		} else if math.IsInf(value, 1) {
			sPriorData.Set(i, c, 1.0)
		} else if math.IsNaN(value) {
			sPriorData.Set(i, c, 0.0)
		} else {
			sPriorData.Set(i, c, value/max)
		}
	}
	mutex.Unlock()
}

func expPknn(Pknn *mat64.Dense, goro int, wg *sync.WaitGroup) (knnIdx map[int][]int, PknnExp *mat64.Dense, PknnMap map[int]map[int]int) {
	knnIdx = make(map[int][]int)
	nNetworkGene, _ := Pknn.Caps()
	nPknnExp := 0
	PknnMap = make(map[int]map[int]int)
	for i := 0; i < nNetworkGene; i++ {
		kArr := make([]int, 0)
		kMap := make(map[int]int)
		for k := 0; k < nNetworkGene; k++ {
			if Pknn.At(i, k) > 0.0 {
				kArr = append(kArr, k)
				kMap[k] = nPknnExp
				nPknnExp += 1
			}
		}
		knnIdx[i] = kArr
		PknnMap[i] = kMap
	}
	//exp matrix
	PknnExp = ParaPairProduct(Pknn, knnIdx, PknnMap, goro, nPknnExp, wg)
	return knnIdx, PknnExp, PknnMap
}

func PropagateSetDLP(network *mat64.Dense, trYdata *mat64.Dense, idIdx map[string]int, idArr []string, trGeneMap map[string]int, alpha float64, nThreads int, wg *sync.WaitGroup) (sPriorData *mat64.Dense, ind []int) {
	//alpha = 0.2
	lamda := 1.0
	P, nNetworkGene := TransitionMatrixByK(network, 999999999)
	P, _ = dNorm(P)
	Pknn, _ := TransitionMatrixByK(network, 10)
	Pknn, _ = dNorm(Pknn)
	knnIdx := make(map[int][]int)
	for i := 0; i < nNetworkGene; i++ {
		kArr := make([]int, 0)
		for k := 0; k < nNetworkGene; k++ {
			if Pknn.At(i, k) > 0.0 {
				kArr = append(kArr, k)
			}
		}
		knnIdx[i] = kArr
	}
	//all possible pairs' product
	//knnIdx, PknnExp, PknnMap := expPknn(Pknn, nThreads, wg)
	//knnPairP := ParaPairProduct(Pknn, knnIdx, nThreads, wg)
	//filter out zero positive instance labels
	Y, ind, nOutLabel := trYdataFilter(trYdata, idArr, idIdx, trGeneMap, nNetworkGene)
	lY := mat64.DenseCopyOf(Y) //original label Y
	//propagate Set by label, smoothed prior data
	itr := 0
	maxItr := 100
	res := 100.0
	thres := 10.0
	nY := mat64.NewDense(0, 0, nil)
	tY := mat64.NewDense(0, 0, nil)
	nP := mat64.NewDense(0, 0, nil)
	for itr <= maxItr && res > thres {
		//propagating labels
		//nY := mat64.NewDense(0, 0, nil)
		nY.Mul(P, Y)
		for i := 0; i < nNetworkGene; i++ {
			for j := 0; j < nOutLabel; j++ {
				if lY.At(i, j) == 1.0 {
					nY.Set(i, j, nY.At(i, j)*0.2+0.8)
					//nY.Set(i, j, 1.0)
				} else {
					nY.Set(i, j, nY.At(i, j)*0.2)
				}
			}
		}
		//new fusion kernel
		tY = ParaFusKernel(P, Y, nThreads, alpha, wg)
		//tY := mat64.DenseCopyOf(P)
		//tY := mat64.NewDense(0, 0, nil)
		//tY.Mul(Y, Y.T())
		//fusKernel := mat64.NewDense(nNetworkGene, nNetworkGene, nil)
		//for i := 0; i < nNetworkGene; i++ {
		//	for j := 0; j < nNetworkGene; j++ {
		//		ele := (P.At(i, j) + alpha*tY.At(i, j))
		//		fusKernel.Set(i, j, ele)
		//	}
		//}
		//updated transition matrix
		nP = ParaTransP(Pknn, tY, knnIdx, nThreads, lamda, wg)
		//nP := mat64.NewDense(nNetworkGene, nNetworkGene, nil)
		//for i := 0; i < nNetworkGene; i++ {
		//	for j := 0; j < nNetworkGene; j++ {
		//		kArr := make([]int, 0)
		//		lArr := make([]int, 0)
		//		ele := 0.0
		//		for k := 0; k < nNetworkGene; k++ {
		//			if Pknn.At(i, k) > 0.0 {
		//				kArr = append(kArr, k)
		//			}
		//			//k -> l , as i -> j
		//			if Pknn.At(j, k) > 0.0 {
		//				lArr = append(lArr, k)
		//			}
		//		}
		//		//each k,l element, kernal part
		//		for k := 0; k < len(kArr); k++ {
		//			eleK := Pknn.At(i, kArr[k])
		//			for l := 0; l < len(lArr); l++ {
		//				eleL := Pknn.At(j, lArr[l])
		//				eleP := P.At(kArr[k], lArr[l])
		//				eleY := 0.0
		//				for m := 0; m < nOutLabel; m++ {
		//					eleY += Y.At(kArr[k], m) * Y.At(lArr[l], m)
		//				}
		//				ele += eleK * eleL * (eleP + alpha*eleY)
		//			}
		//		}
		//		nP.Set(i, j, ele+lamda)
		//	}
		//}

		//term1 := mat64.NewDense(0, 0, nil)
		//term1.Mul(Pknn, fusKernel)
		//nP.Mul(term1, Pknn)
		//for i := 0; i < nNetworkGene; i++ {
		//	nP.Set(i, i, lamda+nP.At(i, i))
		//}
		nP, _ = dNorm(nP)
		//update res, redefine Y and P
		//NormScale(nY)
		res = math.Abs(mat64.Sum(nY) - mat64.Sum(Y))
		log.Print("res: ", res, " at itr: ", itr)
		P = mat64.DenseCopyOf(nP)
		Y = mat64.DenseCopyOf(nY)
		itr += 1
		runtime.GC()
	}
	return Y, ind
}

func PropagateSet2D(network *mat64.Dense, trYdata *mat64.Dense, idIdx map[string]int, idArr []string, trGeneMap map[string]int, transLabels *mat64.Dense, isDada bool, alpha float64, wg *sync.WaitGroup, mutex *sync.Mutex) (sPriorData *mat64.Dense, ind []int) {
	network, nNetworkGene := dNorm(network)
	nTrGene, nTrLabel := trYdata.Caps()
	//ind for prior/label gene set mapping at least one gene to the network
	ind = make([]int, nTrLabel)
	for j := 0; j < nTrLabel; j++ {
		inGene := make([]int, 0)
		for i := 0; i < nTrGene; i++ {
			_, exist := idIdx[idArr[i]]
			_, existTr := trGeneMap[idArr[i]]
			if trYdata.At(i, j) > 0 && exist && existTr {
				ind[j] += 1
				inGene = append(inGene, idIdx[idArr[i]])
			}
		}
	}
	nOutLabel := 0
	for i := 0; i < nTrLabel; i++ {
		if ind[i] > 1 {
			nOutLabel += 1
		}
	}
	//propagate Set by label, smoothed prior data
	sPriorData = mat64.NewDense(nNetworkGene, nOutLabel, nil)
	itr := 0
	maxItr := 3
	res := 1.0
	thres := 0.01
	for itr <= maxItr && res > thres {
		preSum := mat64.Sum(sPriorData)
		//lp with maxItr 1
		c := 0
		wg.Add(nOutLabel)
		for j := 0; j < nTrLabel; j++ {
			if ind[j] > 1 {
				trY := mat64.NewDense(nNetworkGene, 1, nil)
				for i := 0; i < nTrGene; i++ {
					_, exist := idIdx[idArr[i]]
					_, existTr := trGeneMap[idArr[i]]
					if trYdata.At(i, j) > 0 && exist && existTr {
						trY.Set(idIdx[idArr[i]], 0, trYdata.At(i, j)/float64(ind[j]))
					}
				}
				prior := mat64.NewDense(0, 0, nil)
				if itr == 0 {
					prior = mat64.DenseCopyOf(trY)
				} else {
					prior = mat64.NewDense(nNetworkGene, 1, nil)
					for k := 0; k < nNetworkGene; k++ {
						prior.Set(k, 0, sPriorData.At(k, c))
					}
				}
				go single_sPriorData(network, sPriorData, prior, trY, nNetworkGene, 100, alpha, c, wg, mutex)
				c += 1
			}
		}
		wg.Wait()
		//lgs with maxItr 1
		wg.Add(nNetworkGene)
		for j := 0; j < nNetworkGene; j++ {
			go single_LabelGraphSmooth(j, ind, sPriorData, transLabels, 100, wg, mutex)
		}
		wg.Wait()
		//update res and itr
		sum := mat64.Sum(sPriorData)
		res = math.Abs(sum - preSum)
		log.Print("at itr ", itr, ", res is ", res)
		itr += 1
	}
	runtime.GC()
	//colum norm to max 1
	for j := 0; j < nOutLabel; j++ {
		max := 0.0
		for i := 0; i < nNetworkGene; i++ {
			if sPriorData.At(i, j) > max {
				max = sPriorData.At(i, j)
			}
		}
		for i := 0; i < nNetworkGene; i++ {
			sPriorData.Set(i, j, sPriorData.At(i, j)/max)
		}
	}
	return sPriorData, ind
}
func propagate(network *mat64.Dense, alpha float64, inPrior *mat64.Dense, trY *mat64.Dense, maxIter int) (smoothPrior *mat64.Dense) {
	sum := mat64.Sum(inPrior)
	r, _ := inPrior.Caps()
	restart := mat64.NewDense(r, 1, nil)
	prior := mat64.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		restart.Set(i, 0, inPrior.At(i, 0)/sum)
		//restart.Set(i, 0, trY.At(i, 0))
		prior.Set(i, 0, inPrior.At(i, 0)/sum)
		//prior.Set(i, 0, inPrior.At(i, 0))
	}
	thres := 0.000001
	//maxIter := 100
	i := 0
	res := 1.0
	for res > thres && i < maxIter {
		prePrior := mat64.DenseCopyOf(prior)
		term1 := mat64.NewDense(0, 0, nil)
		term1.Mul(network, prior)
		for i := 0; i < r; i++ {
			prior.Set(i, 0, alpha*term1.At(i, 0)+(1-alpha)*restart.At(i, 0))
		}
		res = 0.0
		for i := 0; i < r; i++ {
			res += math.Abs(prior.At(i, 0) - prePrior.At(i, 0))
		}
		i += 1
	}
	//max rescale skipped as it conflict with MLSMOTE and KNN calibration
	//max := 0.0
	//for i := 0; i < r; i++ {
	//	if prior.At(i, 0) > max {
	//		max = prior.At(i, 0)
	//	}
	//}
	//if max == 0.0 {
	//	max = 1.0
	//}
	//for i := 0; i < r; i++ {
	//	if trY.At(i, 0) == 1.0 {
	//		prior.Set(i, 0, 1.0)
	//	} else {
	//		prior.Set(i, 0, prior.At(i, 0)/max)
	//	}
	//}
	return prior
}

func single_LabelGraphSmooth(idx int, ind []int, sPriorData *mat64.Dense, preTransLabels *mat64.Dense, maxIter int, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	//paper suggested default
	alpha := 0.3
	_, nCol := sPriorData.Caps()
	rowData := mat64.NewDense(1, nCol, nil)
	rowData.SetRow(0, sPriorData.RawRowView(idx))
	initRowData := mat64.DenseCopyOf(rowData)
	transLabels := mat64.NewDense(0, 0, nil)
	//reset transLabels according to ind
	isTruncated := false
	s := len(ind)
	for i := 0; i < len(ind); i++ {
		if ind[i] == 0 {
			isTruncated = true
			s -= 1
		}
	}
	if !isTruncated {
		transLabels = mat64.DenseCopyOf(preTransLabels)
	} else {
		transLabels = mat64.NewDense(s, s, nil)
		rT := -1
		for i := 0; i < len(ind); i++ {
			//skip ind Zero rows
			if ind[i] == 0 {
				continue
			} else {
				rT += 1
			}
			cT := -1
			for j := 0; j < len(ind); j++ {
				if ind[j] == 0 {
					continue
				} else {
					cT += 1
				}
				v := preTransLabels.At(i, j)
				transLabels.Set(rT, cT, v)
			}
		}

	}
	//iterations
	thres := 0.001
	//maxIter := 100
	i := 0
	res := 1.0
	sum := 0.0
	for res > thres && i < maxIter {
		preRowData := mat64.DenseCopyOf(rowData)
		for j := 0; j < nCol; j++ {
			prob := 0.0
			for k := 0; k < nCol; k++ {
				if k != j {
					//transition probability from label k to label j
					prob += alpha / (float64(nCol) - 1) * transLabels.At(j, k) * preRowData.At(0, k)
				} else {
					prob += (1 - alpha) * initRowData.At(0, j)
				}
			}
			rowData.Set(0, j, prob)
		}
		res = 0.0
		sum = 0.0
		for j := 0; j < nCol; j++ {
			res += math.Abs(rowData.At(0, j) - preRowData.At(0, j))
			sum += rowData.At(0, j)
		}
		res = res / sum
		i += 1
	}
	//update
	mutex.Lock()
	sPriorData.SetRow(idx, rowData.RawRowView(0))
	mutex.Unlock()
}

func colNorm(network *mat64.Dense) (normNet *mat64.Dense, n int) {
	n, _ = network.Caps()
	normNet = mat64.NewDense(n, n, nil)
	for j := 0; j < n; j++ {
		s := mat64.Sum(network.ColView(j))
		if s > 0.0 {
			for i := 0; i < n; i++ {
				normNet.Set(i, j, network.At(i, j)/s)
			}
		}
	}
	return normNet, n
}

func dNorm(network *mat64.Dense) (normNet *mat64.Dense, n int) {
	n, _ = network.Caps()
	d := mat64.NewDense(n, n, nil)
	normNet = mat64.NewDense(0, 0, nil)
	for j := 0; j < n; j++ {
		s := math.Sqrt(mat64.Sum(network.RowView(j)))
		if s > 0.0 {
			d.Set(j, j, 1.0/s)
		}
	}
	term1 := mat64.NewDense(0, 0, nil)
	term1.Mul(d, network)
	normNet.Mul(term1, d)
	return normNet, n
}
func rwrNetwork(network *mat64.Dense, alpha float64) *mat64.Dense {
	n, _ := network.Caps()
	d := mat64.NewDense(n, n, nil)
	t := mat64.NewDense(n, n, nil)
	rwrNet := mat64.NewDense(0, 0, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				t.Set(i, j, 1.0-(1.0-alpha)*network.At(i, j))
				d.Set(i, j, alpha)
			} else {
				t.Set(i, j, (alpha-1.0)*network.At(i, j))
			}
		}
	}
	invT := mat64.NewDense(0, 0, nil)
	invT.Inverse(t)
	rwrNet.Mul(invT, d)
	return rwrNet
}
func TransitionMatrixByK(network *mat64.Dense, k int) (transP *mat64.Dense, n int) {
	n, _ = network.Caps()
	if k >= n {
		k = n - 1
	}
	transP = mat64.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		//if network.At(i, i) < 1.0 {
		//	network.Set(i, i, 1.0)
		//}
		var sortMap []kv
		//top K values at row
		for j := 0; j < n; j++ {
			sortMap = append(sortMap, kv{j, network.At(i, j)})
		}
		sort.Slice(sortMap, func(i, j int) bool {
			return sortMap[i].Value > sortMap[j].Value
		})
		thres := sortMap[k].Value
		sum := 0.0
		for j := 0; j < n; j++ {
			if network.At(i, j) >= thres {
				sum += network.At(i, j)
			}
		}
		if sum == 0.0 {
			log.Print("row sum zero for row ", i)
			sum = 1.0
		}
		//set transP
		for j := 0; j < n; j++ {
			if network.At(i, j) > thres {
				transP.Set(i, j, network.At(i, j)/sum)
			}
		}
	}
	return transP, n
}

func trYdataFilter(trYdata *mat64.Dense, idArr []string, idIdx map[string]int, trGeneMap map[string]int, nNetWorkGene int) (trY *mat64.Dense, ind []int, nOutLabel int) {
	nTrGene, nTrLabel := trYdata.Caps()
	//ind for prior/label gene set mapping at least one gene to the network
	ind = make([]int, nTrLabel)
	for j := 0; j < nTrLabel; j++ {
		inGene := make([]int, 0)
		for i := 0; i < nTrGene; i++ {
			_, exist := idIdx[idArr[i]]
			_, existTr := trGeneMap[idArr[i]]
			if trYdata.At(i, j) > 0 && exist && existTr {
				ind[j] += 1
				inGene = append(inGene, idIdx[idArr[i]])
			}
		}
	}
	nOutLabel = 0
	for i := 0; i < nTrLabel; i++ {
		if ind[i] > 1 {
			nOutLabel += 1
		}
	}
	//trY
	trY = mat64.NewDense(nNetWorkGene, nOutLabel, nil)
	tCol := 0
	for j := 0; j < nTrLabel; j++ {
		if ind[j] > 1 {
			for i := 0; i < nTrGene; i++ {
				_, exist := idIdx[idArr[i]]
				_, existTr := trGeneMap[idArr[i]]
				if trYdata.At(i, j) > 0 && exist && existTr {
					trY.Set(idIdx[idArr[i]], tCol, trYdata.At(i, j))
				}
			}
			tCol += 1
		}
	}
	return trY, ind, nOutLabel
}

package main

import (
	"github.com/gonum/matrix/mat64"
	"math"
	//"sort"
)

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

func propagateSet(network *mat64.Dense, priorData *mat64.Dense, idIdx map[string]int, idArr []string, trGeneMap map[string]int) (sPriorData *mat64.Dense) {
	network, nNetworkGene := colNorm(network)
	nPriorGene, nPriorLabel := priorData.Caps()
	//ind for prior/label gene set mapping at least one gene to the network
	ind := make([]int, nPriorLabel)
	for j := 0; j < nPriorLabel; j++ {
		for i := 0; i < nPriorGene; i++ {
			_, exist := idIdx[idArr[i]]
			_, existTr := trGeneMap[idArr[i]]
			if priorData.At(i, j) > 0 && exist && existTr {
				ind[j] += 1
			}
		}
	}
	nOutLabel := 0
	for i := 0; i < nPriorLabel; i++ {
		if ind[i] > 0 {
			nOutLabel += 1
		}
	}
	sPriorData = mat64.NewDense(nNetworkGene, nOutLabel, nil)
	c := 0
	wg.Add(nOutLabel)
	for j := 0; j < nPriorLabel; j++ {
		if ind[j] > 0 {
			prior := mat64.NewDense(nNetworkGene, 1, nil)
			for i := 0; i < nPriorGene; i++ {
				_, exist := idIdx[idArr[i]]
				_, existTr := trGeneMap[idArr[i]]
				if priorData.At(i, j) > 0 && exist && existTr {
					prior.Set(idIdx[idArr[i]], 0, priorData.At(i, j))
				}
			}
			go single_sPriorData(network, sPriorData, prior, nNetworkGene, c)
			c += 1
		}
	}
	wg.Wait()
	return sPriorData
}

func single_sPriorData(network *mat64.Dense, sPriorData *mat64.Dense, prior *mat64.Dense, nNetworkGene int, c int) {
	defer wg.Done()
	//n by 1 matrix
	sPrior1 := propagate(network, 0.7, prior)
	sPrior2 := propagate(network, 1.0, prior)
	mutex.Lock()
	for i := 0; i < nNetworkGene; i++ {
		value := sPrior1.At(i, 0) / sPrior2.At(i, 0)
		if math.IsInf(value, 1) {
			sPriorData.Set(i, c, 1.0)
		} else if math.IsNaN(value) {
			sPriorData.Set(i, c, 0.0)
		} else {
			sPriorData.Set(i, c, value)
		}
	}
	mutex.Unlock()
}
func propagate(network *mat64.Dense, alpha float64, inPrior *mat64.Dense) (smoothPrior *mat64.Dense) {
	sum := mat64.Sum(inPrior)
	r, _ := inPrior.Caps()
	restart := mat64.NewDense(r, 1, nil)
	prior := mat64.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		restart.Set(i, 0, inPrior.At(i, 0)/sum)
		prior.Set(i, 0, inPrior.At(i, 0)/sum)
	}
	thres := 0.0000000001
	maxIter := 1000
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
	//var sortMap []kv
	//for i := 0; i < r; i++ {
	//	sortMap = append(sortMap, kv{i, prior.At(i, 0)})
	//}
	//sort.Slice(sortMap, func(i, j int) bool {
	//	return sortMap[i].Value > sortMap[j].Value
	//})

	//thres = sortMap[50].Value
	max := 0.0
	for i := 0; i < r; i++ {
		if prior.At(i, 0) > max {
			max = prior.At(i, 0)
		}
	}

	for i := 0; i < r; i++ {
		//	if prior.At(i, 0) < thres {
		prior.Set(i, 0, prior.At(i, 0)/max)
		//	}
	}
	return prior
}

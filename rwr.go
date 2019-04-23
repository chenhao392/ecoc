package main

import (
	"flag"
	"fmt"
	"github.com/gonum/matrix/mat64"
)

func main() {
	var inNetworkFile *string = flag.String("n", "data/hs_fus_net.txt", "network file")
	var priorMatrix *string = flag.String("p", "data/human.bp.level1.set1.trMatrix.txt", "prior/known gene file")
	flag.Parse()
	//idIdx as gene -> idx in net
	//network, idIdx, idxToId := readNetwork(*inNetworkFile)
	network, idIdx, _ := readNetwork(*inNetworkFile)
	network, n := colNorm(network)
	//idArr  gene index as in prior file
	priorData, idArr, _ := readFile(*priorMatrix, true)
	nGene, nLabel := priorData.Caps()
	r, c := priorData.Caps()
	sPriorData := mat64.NewDense(r, c, nil)
	//nGene, _ := priorData.Caps()
	for j := 0; j < nLabel; j++ {
		prior := mat64.NewDense(n, 1, nil)
		for i := 0; i < nGene; i++ {
			if priorData.At(i, j) > 0 {
				prior.Set(idIdx[idArr[i]], 0, priorData.At(i, j))
			}
		}
		sPrior := propagate(network, 0.3, prior)
		//sort and print sPrior
		for i := 0; i < n; i++ {
			//fmt.Println(idxToId[i], sPrior.At(i, 0))
			sPriorData.Set(i, j, sPrior.At(i, 0))
		}
	}
	//test out
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			fmt.Println(sPriorData.RawRowView(i))
		}
	}

}

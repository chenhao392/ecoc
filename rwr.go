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
	network, idIdx := readNetwork(*inNetworkFile)
	network, n := colNorm(network)
	priorData, idArr, _ := readFile(*priorMatrix, true)
	//nGene, nLabel := priorData.Caps()
	nGene, _ := priorData.Caps()
	for j := 2; j < 3; j++ {
		prior := mat64.NewDense(n, 1, nil)
		for i := 0; i < nGene; i++ {
			if priorData.At(i, j) > 0 {
				prior.Set(idIdx[idArr[i]], 0, priorData.At(i, j))
			}
		}
		sPrior := propagate(network, 0.4, prior)
		//sort and print sPrior
		for i := 0; i < n; i++ {
			fmt.Println(idArr[i], sPrior.At(i, 0))
		}

	}
}

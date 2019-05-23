package main

import (
	"flag"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"os"
	"strings"
	"sync"
)

var wg sync.WaitGroup
var mutex sync.Mutex

func main() {
	var inNetworkFiles *string = flag.String("n", "data/sc_coe_net.txt", "network file")
	var trYFile *string = flag.String("trY", "data/yeast.level1.set1.trMatrix.txt", "prior/known gene file")
	var priorMatrixFiles *string = flag.String("p", "data/sc_exp_net.txt", "prior/known gene file")
	flag.Parse()
	inNetworkFile := strings.Split(*inNetworkFiles, ",")
	priorMatrixFile := strings.Split(*priorMatrixFiles, ",")
	trYdata, trRowName, _, _ := readFile(*trYFile, true, true)
	for i := 0; i < len(inNetworkFile); i++ {
		fmt.Println(inNetworkFile[i])
		for j := 0; j < len(priorMatrixFile); j++ {
			//idIdx as gene -> idx in net
			network, idIdx, idxToId := readNetwork(inNetworkFile[i])
			network, n := dNorm(network)
			//idArr  gene index as in prior file
			priorData, priorGeneID, priorIdxToId := readNetwork(priorMatrixFile[j])
			//for id, i := range priorGeneID {
			//	fmt.Println(id, i)
			//}
			//os.Exit(0)
			//sPriorData := propagateSet(network, priorData, idIdx, idArr)
			//n, _ := network.Caps()
			trY := mat64.NewDense(n, 1, nil)
			tsY := mat64.NewDense(n, 1, nil)
			prior := mat64.NewDense(n, 1, nil)
			for i := 0; i < len(trRowName); i++ {
				if i%2 == 0 {
					trY.Set(idIdx[trRowName[i]], 0, trYdata.At(i, 3))
					if trYdata.At(i, 3) == 1.0 {
						//fmt.Println("trY", i, trRowName[i])
					}
				} else {
					tsY.Set(idIdx[trRowName[i]], 0, trYdata.At(i, 3))
					if trYdata.At(i, 3) == 1.0 {
						//fmt.Println("tsY", i, trRowName[i])
					}
				}
			}
			//for i := 0; i < n; i++ {
			//	fmt.Println(idxToId[i], prior.At(i, 0))
			//}
			//fmt.Println(idxToId[7950], trY.At(7950, 0))
			for k := 0; k < 30; k += 2 {
				prior = addPrior(priorData, priorGeneID, priorIdxToId, trY, idIdx, idxToId, k, n)

				//sPrior := propagate(network, 0.6, prior)
				//sPrior2 := propagate(network, 0.6, trY)
				//for i := 0; i < n; i++ {
				//	fmt.Println(i, idxToId[i], trY.At(i, 0), tsY.At(i, 0), prior.At(i, 0), sPrior.At(i, 0), sPrior2.At(i, 0))
				//}
				aupr, aupr2 := featureAupr(network, prior, tsY, trY)
				aupr3, aupr4 := featureAupr(network, trY, tsY, trY)
				fmt.Println(aupr, aupr2, aupr3, aupr4)
			}
			os.Exit(0)
			//sPrior := propagate(network, 0.6, prior)
			//sPrior2 := propagate(network, 1.0, prior)
			//for i := 0; i < n; i++ {
			//	fmt.Println(idxToId[i], sPrior.At(i, 0)/sPrior2.At(i, 0))
			//fmt.Println(idxToId[i], sPrior.At(i, 0))
			//}
		}
	}

}

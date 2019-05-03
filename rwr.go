package main

import (
	"flag"
	"fmt"
	"github.com/gonum/matrix/mat64"
	//"os"
	"strings"
	"sync"
)

var wg sync.WaitGroup
var mutex sync.Mutex

func main() {
	var inNetworkFiles *string = flag.String("n", "data/hs_coe_net.txt", "network file")
	var priorMatrixFiles *string = flag.String("p", "data/human.bp.level1.set1.trMatrix.txt", "prior/known gene file")
	flag.Parse()
	inNetworkFile := strings.Split(*inNetworkFiles, ",")
	priorMatrixFile := strings.Split(*priorMatrixFiles, ",")
	for i := 0; i < len(inNetworkFile); i++ {
		fmt.Println(inNetworkFile[i])
		for j := 0; j < len(priorMatrixFile); j++ {
			//idIdx as gene -> idx in net
			network, idIdx, idxToId := readNetwork(inNetworkFile[i])
			network, n := colNorm(network)
			//r, c := network.Caps()
			//for k := 0; k < 10; k++ {
			//	for l := 0; l < 10; l++ {
			//		fmt.Printf("\t%.3f", network.At(k, l))
			//	}
			//	fmt.Println("")
			//}
			//os.Exit(0)
			//network, idIdx, idxToId := readNetwork(*inNetworkFile)
			//idArr  gene index as in prior file
			priorData, idArr, _, _ := readFile(priorMatrixFile[j], true, true)
			//sPriorData := propagateSet(network, priorData, idIdx, idArr)
			//n, _ := network.Caps()
			prior := mat64.NewDense(n, 1, nil)
			for i := 0; i < len(idArr); i++ {
				prior.Set(idIdx[idArr[i]], 0, priorData.At(i, 2))
			}
			//for i := 0; i < n; i++ {
			//	fmt.Println(idxToId[i], prior.At(i, 0))
			//}
			sPrior := propagate(network, 0.7, prior)
			sPrior2 := propagate(network, 1.0, prior)
			for i := 0; i < n; i++ {
				fmt.Println(idxToId[i], sPrior.At(i, 0)/sPrior2.At(i, 0))
				//fmt.Println(idxToId[i], sPrior.At(i, 0))
			}
		}
	}

}

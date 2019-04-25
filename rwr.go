package main

import (
	"flag"
	"fmt"
	//"github.com/gonum/matrix/mat64"
	//"os"
	"strings"
)

func main() {
	var inNetworkFiles *string = flag.String("n", "data/hs_fus_net.txt,data/hs_pp_net.txt", "network file")
	var priorMatrixFiles *string = flag.String("p", "data/human.bp.level1.set1.trMatrix.txt,data/human.bp.level1.set2.trMatrix.txt", "prior/known gene file")
	flag.Parse()
	inNetworkFile := strings.Split(*inNetworkFiles, ",")
	priorMatrixFile := strings.Split(*priorMatrixFiles, ",")
	for i := 0; i < len(inNetworkFile); i++ {
		fmt.Println(inNetworkFile[i])
		for j := 0; j < len(priorMatrixFile); j++ {
			//idIdx as gene -> idx in net
			network, idIdx, _ := readNetwork(inNetworkFile[i])
			//network, idIdx, idxToId := readNetwork(*inNetworkFile)
			//idArr  gene index as in prior file
			priorData, idArr, _, _ := readFile(priorMatrixFile[j], true, true)
			sPriorData := propagateSet(network, priorData, idIdx, idArr)
			a, b := sPriorData.Caps()
			fmt.Println(a, b)
		}
	}

}

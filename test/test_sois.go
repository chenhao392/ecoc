package main

import (
	"flag"
	"fmt"
	"github.com/chenhao392/ecoc/src"
	"github.com/gonum/matrix/mat64"
)

func main() {
	var trY *string = flag.String("i", "data/human.bp.level1.set1.trMatrix.txt", "tr matrix")
	flag.Parse()
	trYdata, _, _, _ := src.ReadFile(*trY, true, true)
	folds := src.SOIS(trYdata, 5)
	//fmt.Println(folds[0])
	_, nCol := trYdata.Caps()
	cvCount := mat64.NewDense(5, nCol, nil)
	for i := 0; i < 5; i++ {
		for j := 0; j < nCol; j++ {
			for k := 0; k < len(folds[i]); k++ {
				cvCount.Set(i, j, trYdata.At(folds[i][k], j)+cvCount.At(i, j))
			}
		}
		fmt.Println(cvCount.RawRowView(i))
	}

}

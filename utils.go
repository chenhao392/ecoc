package main

import (
	"github.com/gonum/matrix/mat64"
)

func Shift(pToSlice *[]string) string {
	sValue := (*pToSlice)[0]
	*pToSlice = (*pToSlice)[1:len(*pToSlice)]
	return sValue
}

func Pop(pToSlice *[]float64) float64 {
	pValue := (*pToSlice)[len(*pToSlice)-1]
	*pToSlice = (*pToSlice)[0 : len(*pToSlice)-1]
	return pValue
}

//only row stacking available in mat64, the matlab code use "cbind"
func colStack(X *mat64.Dense, oneSlice []float64) (X2 *mat64.Dense) {
	X = mat64.DenseCopyOf(X.T())
	_, n := X.Caps()
	ones := mat64.DenseCopyOf(mat64.NewDense(1, n, oneSlice))
	X2 = mat64.NewDense(0, 0, nil)
	X2.Stack(ones, X)
	X2 = mat64.DenseCopyOf(X2.T())
	return X2
}

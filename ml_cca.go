package main

import (
	"fmt"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
	"log"
	"math"
	//"os"
)

func ccaProjectTwoMatrix(X *mat64.Dense, Y *mat64.Dense) (W_x *mat64.Dense, W_y *mat64.Dense) {
	var Xsvd, Ysvd, Bsvd mat64.SVD
	var uXFull, vXFull, uYFull, vYFull, pBFull mat64.Dense
	//init SVD
	ok := Xsvd.Factorize(X.T(), matrix.SVDThin)
	if !ok {
		log.Fatal("SVD for X factorization failed!")
	}
	ok = Ysvd.Factorize(Y.T(), matrix.SVDThin)
	if !ok {
		log.Fatal("SVD for Y factorization failed!")
	}
	uXFull.UFromSVD(&Xsvd)
	vXFull.VFromSVD(&Xsvd)
	uYFull.UFromSVD(&Ysvd)
	vYFull.VFromSVD(&Ysvd)
	//ranks and sigma values
	nR, nC := X.Caps()
	nsX := int(math.Min(float64(nR), float64(nC)))
	nR, nC = Y.Caps()
	nsY := int(math.Min(float64(nR), float64(nC)))
	sValuesFullX := Xsvd.Values(nil)
	sValuesX := make([]float64, 0)
	for i := 0; i < nsX; i++ {
		if sValuesFullX[i] > 0.000001 {
			sValuesX = append(sValuesX, sValuesFullX[i])
		}
	}
	sValuesFullY := Ysvd.Values(nil)
	sValuesY := make([]float64, 0)
	for i := 0; i < nsY; i++ {
		if sValuesFullY[i] > 0.000001 {
			sValuesY = append(sValuesY, sValuesFullY[i])
		}
	}
	Xrank := len(sValuesX)
	Yrank := len(sValuesY)
	fmt.Println(Xrank, Yrank)
	//resize matrix according to ranks
	a, _ := uXFull.Caps()
	uX := uXFull.Slice(0, a, 0, Xrank)
	a, _ = vXFull.Caps()
	vX := vXFull.Slice(0, a, 0, Xrank)

	a, _ = uYFull.Caps()
	uY := uYFull.Slice(0, a, 0, Yrank)
	a, _ = vYFull.Caps()
	vY := vYFull.Slice(0, a, 0, Yrank)
	H := mat64.NewDense(0, 0, nil)
	H.Mul(vY, uY.T())
	//for i := 0; i < 10; i++ {
	//	fmt.Println(mat64.DenseCopyOf(uY).RawRowView(i))
	//}
	//fmt.Println("~~~~~")
	//H is correct
	//for i := 0; i < 20; i++ {
	//	fmt.Println(H.RawRowView(i))
	//}
	//fmt.Println("~~~~~")
	//os.Exit(0)
	sValues := Ysvd.Values(nil)
	Y_Sigma := mat64.NewDense(Yrank, Yrank, nil)
	Y_Sigma2 := mat64.NewDense(Yrank, Yrank, nil)
	for i := 0; i < Yrank; i++ {
		Y_Sigma.Set(i, i, sValues[i])
		Y_Sigma2.Set(i, i, 1/sValues[i])
	}
	sValues = Xsvd.Values(nil)
	X_Sigma := mat64.NewDense(Xrank, Xrank, nil)
	X_Sigma2 := mat64.NewDense(Xrank, 1, nil)
	X_Sigma3 := mat64.NewDense(Xrank, Xrank, nil)
	X_SigmaB := mat64.NewDense(Xrank, Xrank, nil)
	for i := 0; i < Xrank; i++ {
		X_Sigma.Set(i, i, sValues[i])
		X_Sigma2.Set(i, 0, 1/sValues[i])
		X_Sigma3.Set(i, i, 1/sValues[i])
		//fmt.Println(i, 1/sValues[i])
		X_SigmaB.Set(i, i, 1.0)
	}

	//[W, eigenList] =solve_eigen(X_U,X_Sigma,X_V,H,X_reg)
	term1 := mat64.NewDense(0, 0, nil)
	B := mat64.NewDense(0, 0, nil)
	term1.Mul(X_SigmaB, vX.T())
	B.Mul(term1, H)
	//B is correct
	//for i := 0; i < 14; i++ {
	//	fmt.Println(B.RawRowView(i))
	//}
	//fmt.Println("~~~~")
	ok = Bsvd.Factorize(B, matrix.SVDThin)
	if !ok {
		log.Fatal("SVD for B factorization failed!")
	}
	pBFull.UFromSVD(&Bsvd)

	bValuesFull := Bsvd.Values(nil)
	Brank := 0
	for i := 0; i < len(bValuesFull); i++ {
		if bValuesFull[i] > 0.000001 {
			Brank += 1
		}
	}
	a, _ = pBFull.Caps()
	//fmt.Println(a, Brank)
	pB := pBFull.Slice(0, a, 0, Brank)
	term2 := mat64.NewDense(0, 0, nil)
	W_x = mat64.NewDense(0, 0, nil)
	W_y = mat64.NewDense(0, 0, nil)
	//term2.Mul(uX, X_Sigma3)
	term2.Mul(X_Sigma3, pB)
	//W_x.Mul(term2, pB)
	W_x.Mul(uX, term2)
	//fmt.Println(W_x.At(1, 0))
	//for i := 0; i < 13; i++ {
	//	fmt.Println(mat64.DenseCopyOf(pB).RawRowView(i))
	//}
	//os.Exit(0)
	//W_y after solve eigen
	term3 := mat64.NewDense(0, 0, nil)
	term4 := mat64.NewDense(0, 0, nil)
	term5 := mat64.NewDense(0, 0, nil)
	term3.Mul(uY, Y_Sigma2)
	term4.Mul(term3, vY.T())
	//was not Transposed for now
	term5.Mul(term4, X)
	W_y.Mul(term5, W_x)
	//normalize W_y
	m, n := W_y.Caps()
	nFactor := make([]float64, n)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			nFactor[j] = nFactor[j] + W_y.At(i, j)*W_y.At(i, j)
		}
		nFactor[j] = math.Sqrt(nFactor[j])
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			W_y.Set(i, j, W_y.At(i, j)/nFactor[j])
		}
	}

	return W_x, W_y
}

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
	ok := Xsvd.Factorize(X.T(), matrix.SVDFull)
	if !ok {
		log.Fatal("SVD for X factorization failed!")
	}
	ok = Ysvd.Factorize(Y.T(), matrix.SVDFull)
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
		//0.000001 is the default cut-off for ranks in matlab
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
	ok = Bsvd.Factorize(B, matrix.SVDFull)
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
	//fmt.Println(mat64.DenseCopyOf(pB).RawRowView(i))
	//	fmt.Println(W_x.RawRowView(i))
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
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			nFactor[j] = nFactor[j] + W_y.At(i, j)*W_y.At(i, j)
		}
	}
	for j := 0; j < n; j++ {
		nFactor[j] = math.Sqrt(nFactor[j])
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			W_y.Set(i, j, W_y.At(i, j)/nFactor[j])
		}
	}

	return W_x, W_y
}

func ccaProject(X *mat64.Dense, Y *mat64.Dense) (W_y *mat64.Dense) {
	var Xsvd, Ysvd, Bsvd mat64.SVD
	var uXFull, vXFull, uYFull, vYFull, pBFull mat64.Dense
	Yreg := 0.0
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
		//0.000001 is the default cut-off for ranks in matlab
		if sValuesFullX[i] > 0.000001 {
			sValuesX = append(sValuesX, sValuesFullX[i])
			//fmt.Println(sValuesFullX[i])
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
	//fmt.Println(a, b, Yrank)
	vY := vYFull.Slice(0, a, 0, Yrank)
	H := mat64.NewDense(0, 0, nil)
	H.Mul(vX, uX.T())
	//for i := 0; i < 10; i++ {
	//	fmt.Println(mat64.DenseCopyOf(vX).RawRowView(i))
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
	Y_SigmaReg := mat64.NewDense(Yrank, Yrank, nil)
	Y_SigmaRegInv := mat64.NewDense(Yrank, Yrank, nil)
	Y_SigmaB := mat64.NewDense(Yrank, Yrank, nil)
	for i := 0; i < Yrank; i++ {
		Y_Sigma.Set(i, i, sValues[i])
		Y_SigmaReg.Set(i, i, sValues[i]*sValues[i]+Yreg)
		Y_SigmaRegInv.Set(i, i, 1/(sValues[i]*sValues[i]+Yreg))
		Y_SigmaB.Set(i, i, sValues[i]/Y_SigmaRegInv.At(i, i))
	}
	sValues = Xsvd.Values(nil)
	X_Sigma := mat64.NewDense(Xrank, Xrank, nil)
	X_Sigma2 := mat64.NewDense(Xrank, 1, nil)
	for i := 0; i < Xrank; i++ {
		X_Sigma.Set(i, i, sValues[i])
		X_Sigma2.Set(i, 0, 1/sValues[i])
		//fmt.Println(i, 1/sValues[i])
	}

	//[W, eigenList] =solve_eigen(X_U,X_Sigma,X_V,H,X_reg)
	term1 := mat64.NewDense(0, 0, nil)
	B := mat64.NewDense(0, 0, nil)
	term1.Mul(Y_SigmaB, vY.T())
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
	//W_x = mat64.NewDense(0, 0, nil)
	W_y = mat64.NewDense(0, 0, nil)
	//term2.Mul(uX, X_Sigma3)
	term2.Mul(Y_SigmaRegInv, pB)
	//W_x.Mul(term2, pB)
	W_y.Mul(uY, term2)
	//fmt.Println(W_x.At(1, 0))
	for i := 0; i < 1; i++ {
		fmt.Println(mat64.DenseCopyOf(pB).RawRowView(i))
		//	//fmt.Println(W_x.RawRowView(i))
	}
	//os.Exit(0)
	//W_y after solve eigen
	//term3 := mat64.NewDense(0, 0, nil)
	//term4 := mat64.NewDense(0, 0, nil)
	//term5 := mat64.NewDense(0, 0, nil)
	//term3.Mul(uY, Y_Sigma2)
	//term4.Mul(term3, vY.T())
	//was not Transposed for now
	//term5.Mul(term4, X)
	//W_y.Mul(term5, W_x)
	//normalize W_y
	//m, n := W_y.Caps()
	//nFactor := make([]float64, n)
	//for i := 0; i < m; i++ {
	//	for j := 0; j < n; j++ {
	//		nFactor[j] = nFactor[j] + W_y.At(i, j)*W_y.At(i, j)
	//	}
	//}
	//for j := 0; j < n; j++ {
	//	nFactor[j] = math.Sqrt(nFactor[j])
	//}
	//for i := 0; i < m; i++ {
	//	for j := 0; j < n; j++ {
	//		W_y.Set(i, j, W_y.At(i, j)/nFactor[j])
	//	}
	//}

	return W_y
}

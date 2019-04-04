package main

import (
	//"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
)

//for cross validation
type cvFold struct {
	X *mat64.Dense
	Y *mat64.Dense
}

func (f *cvFold) setXY(pos []int, neg []int, matX *mat64.Dense, vecY *mat64.Vector) {
	_, nColX := matX.Caps()
	nRowPos := len(pos)
	nRowNeg := len(neg)
	f.X = mat64.NewDense(nRowPos+nRowNeg, nColX, nil)
	f.Y = mat64.NewDense(nRowPos+nRowNeg, 1, nil)
	for i := 0; i < nRowPos; i++ {
		f.X.SetRow(i, matX.RawRowView(pos[i]))
		f.Y.Set(i, 0, vecY.At(pos[i], 0))
	}
	for i := nRowPos; i < nRowPos+nRowNeg; i++ {
		f.X.SetRow(i, matX.RawRowView(neg[i-nRowPos]))
		f.Y.Set(i, 0, vecY.At(neg[i-nRowPos], 0))
	}
}
func (f *cvFold) setXYinDecoding(idxArr []int, matX *mat64.Dense, vecY *mat64.Vector) {
	_, nColX := matX.Caps()
	nRow := len(idxArr)
	f.X = mat64.NewDense(nRow, nColX, nil)
	f.Y = mat64.NewDense(nRow, 1, nil)
	for i := 0; i < nRow; i++ {
		f.X.SetRow(i, matX.RawRowView(idxArr[i]))
		f.Y.Set(i, 0, vecY.At(idxArr[i], 0))
	}
}

func minIdx(inArray []float64) (idx int) {
	m := inArray[0]
	minSet := make([]int, 0)
	for _, e := range inArray {
		if e < m {
			m = e
		}
	}

	for i, e := range inArray {
		if e == m {
			minSet = append(minSet, i)
		}
	}
	roundIdx := int(math.Round(float64(len(minSet)) / 2.0))
	idx = minSet[roundIdx-1]
	return idx
}

func computeF1(X *mat64.Dense, Y *mat64.Dense, beta *mat64.Dense) (F1 float64) {
	n, _ := X.Caps()
	onesSlice := make([]float64, n)
	for i := range onesSlice {
		onesSlice[i] = 1
	}
	//Y*2-1, tmp fix
	r, c := Y.Caps()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			Y.Set(i, j, Y.At(i, j)*2-1)
		}
	}
	X2 := colStack(X, onesSlice)
	Yh := mat64.NewDense(0, 0, nil)
	Yh.Mul(X2, beta)
	var tp int
	var fp int
	var fn int
	var tn int
	for i := 0; i < n; i++ {
		y := Y.At(i, 0)
		yh := Yh.At(i, 0)
		if y > 0 && yh > 0 {
			tp += 1
		} else if y <= 0 && yh > 0 {
			fp += 1
		} else if y > 0 && yh <= 0 {
			fn += 1
		} else if y <= 0 && yh <= 0 {
			tn += 1
		}
	}
	var prec float64
	var rec float64
	//P and R
	if tp+fp == 0 {
		prec = 0
	} else {
		prec = float64(tp) / (float64(tp) + float64(fp))
	}
	if tp+fn == 0 {
		rec = 0.5
	} else {
		rec = float64(tp) / (float64(tp) + float64(fn))
	}
	//F1
	if prec+rec == 0 {
		F1 = 0
	} else {
		F1 = 2 * float64(prec) * float64(rec) / (float64(prec) + float64(rec))
	}
	return F1
}

func computeF1_2(Y *mat64.Vector, Yh *mat64.Vector, thres float64) (F1 float64) {
	n := Y.Len()
	var tp int
	var fp int
	var fn int
	var tn int
	for i := 0; i < n; i++ {
		y := Y.At(i, 0)
		yh := Yh.At(i, 0)
		if y > 0 && yh >= thres {
			tp += 1
		} else if y <= 0 && yh >= thres {
			fp += 1
		} else if y > 0 && yh < thres {
			fn += 1
		} else if y <= 0 && yh < thres {
			tn += 1
		}
	}
	var prec float64
	var rec float64
	//P and R
	if tp+fp == 0 {
		prec = 0
	} else {
		prec = float64(tp) / (float64(tp) + float64(fp))
	}
	if tp+fn == 0 {
		rec = 0.5
	} else {
		rec = float64(tp) / (float64(tp) + float64(fn))
	}
	//F1
	if prec+rec == 0 {
		F1 = 0
	} else {
		F1 = 2 * float64(prec) * float64(rec) / (float64(prec) + float64(rec))
	}
	return F1
}

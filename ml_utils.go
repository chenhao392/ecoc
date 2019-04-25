package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
	"sort"
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
func (f *cvFold) setXYinNestedTraining(idxArr []int, matX *mat64.Dense, matY *mat64.Dense) {
	_, nColX := matX.Caps()
	_, nColY := matY.Caps()
	nRow := len(idxArr)
	f.X = mat64.NewDense(nRow, nColX, nil)
	f.Y = mat64.NewDense(nRow, nColY, nil)
	for i := 0; i < nRow; i++ {
		f.X.SetRow(i, matX.RawRowView(idxArr[i]))
		f.Y.SetRow(i, matY.RawRowView(idxArr[i]))
	}
}
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

func colStackMatrix(X *mat64.Dense, addX *mat64.Dense) (X2 *mat64.Dense) {
	X = mat64.DenseCopyOf(X.T())
	X2 = mat64.NewDense(0, 0, nil)
	a, b := X.Caps()
	c, d := addX.Caps()
	fmt.Println(a, b, d, c)
	X2.Stack(addX.T(), X)
	X2 = mat64.DenseCopyOf(X2.T())
	return X2
}

func posFilter(trYdata *mat64.Dense) (colSum *mat64.Vector, trYdataFilter *mat64.Dense) {
	r, c := trYdata.Caps()
	colSum = mat64.NewVector(c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			colSum.SetVec(j, colSum.At(j, 0)+trYdata.At(i, j))
		}
	}
	nCol := 0
	for j := 0; j < c; j++ {
		if colSum.At(j, 0) >= 1.0 {
			colSum.SetVec(j, 1.0)
			nCol += 1
		}
	}
	trYdataFilter = mat64.NewDense(r, nCol, nil)
	tC := 0
	for j := 0; j < c; j++ {
		if colSum.At(j, 0) == 1.0 {
			for i := 0; i < r; i++ {
				trYdataFilter.Set(i, tC, trYdata.At(i, j))
			}
			tC += 1
		}
	}
	return colSum, trYdataFilter
}

func posSelect(tsYdata *mat64.Dense, colSum *mat64.Vector) (tsYdataFilter *mat64.Dense) {
	r, c := tsYdata.Caps()
	nCol := 0
	for j := 0; j < c; j++ {
		if colSum.At(j, 0) == 1.0 {
			nCol += 1
		}
	}
	tsYdataFilter = mat64.NewDense(r, nCol, nil)
	tC := 0
	for j := 0; j < c; j++ {
		if colSum.At(j, 0) == 1.0 {
			for i := 0; i < r; i++ {
				tsYdataFilter.Set(i, tC, tsYdata.At(i, j))
			}
			tC += 1
		}
	}
	return tsYdataFilter
}

func cvSplit(nElement int, nFold int) (cvSet map[int][]int) {
	//rand.Seed(2)
	cvSet = make(map[int][]int)
	idxPerm := rand.Perm(nElement)
	j := 0
	if nElement >= nFold {
		for i := 0; i < nElement; i++ {
			j = i % nFold
			cvSet[j] = append(cvSet[j], idxPerm[i])
		}
	} else {
		nDiff := nFold - nElement
		for i := 0; i < nDiff; i++ {
			idxPermTemp := rand.Perm(nElement)
			idxPerm = append(idxPerm, idxPermTemp[0])
		}
		for i := 0; i < nElement; i++ {
			j = i % nFold
			cvSet[j] = append(cvSet[j], idxPerm[i])
		}
	}
	return cvSet
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
func computeAupr(Y *mat64.Vector, Yh *mat64.Vector) (aupr float64) {
	type kv struct {
		Key   int
		Value float64
	}
	n := Y.Len()
	mapY := make(map[int]int)
	var sortYh []kv
	for i := 0; i < n; i++ {
		if Y.At(i, 0) == 1.0 {
			mapY[i] = 1
		}
		ele := Yh.At(i, 0)
		if math.IsNaN(ele) {
			sortYh = append(sortYh, kv{i, 0.0})
		} else {
			sortYh = append(sortYh, kv{i, Yh.At(i, 0)})
		}
	}
	sort.Slice(sortYh, func(i, j int) bool {
		return sortYh[i].Value > sortYh[j].Value
	})

	all := 0.0
	p := 0.0
	tp := 0.0
	total := float64(len(mapY))
	prData := make([]float64, 0)
	for _, kv := range sortYh {
		//fmt.Println(kv.Key, kv.Value)
		all += 1.0
		p += 1.0
		_, ok := mapY[kv.Key]
		if ok {
			tp += 1.0
			pr := tp / p
			re := tp / total
			prData = append(prData, pr)
			prData = append(prData, re)
		}
	}

	aupr = 0.0
	for i := 2; i < len(prData)-1; i += 2 {
		//fmt.Println("AUPR:", aupr, prData[i-2], prData[i], prData[i+1], prData[i-1])
		aupr += (prData[i] + prData[i-2]) * (prData[i+1] - prData[i-1])
	}
	aupr = aupr / 2
	//fmt.Println("AUPR: ", aupr)
	return aupr
}

func computeF1_3(Y *mat64.Vector, Yh *mat64.Vector, rankCut int) (F1 float64) {
	type kv struct {
		Key   int
		Value float64
	}
	n := Y.Len()
	mapY := make(map[int]int)
	var sortYh []kv
	for i := 0; i < n; i++ {
		if Y.At(i, 0) == 1.0 {
			mapY[i] = 1
		}
		ele := Yh.At(i, 0)
		if math.IsNaN(ele) {
			sortYh = append(sortYh, kv{i, 0.0})
		} else {
			sortYh = append(sortYh, kv{i, Yh.At(i, 0)})
		}
	}
	sort.Slice(sortYh, func(i, j int) bool {
		return sortYh[i].Value > sortYh[j].Value
	})
	//o based index, thus -1
	thres := sortYh[rankCut-1].Value
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

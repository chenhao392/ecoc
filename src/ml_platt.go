package src

import (
	//	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	//	"os"
	"sort"
)

func SelectPlattAB(cBest int, plattASet *mat64.Dense, plattBSet *mat64.Dense, plattCountSet *mat64.Dense) (plattAB *mat64.Dense) {
	_, nCol := plattASet.Caps()
	plattAB = mat64.NewDense(2, nCol, nil)
	for j := 0; j < nCol; j++ {
		if plattCountSet.At(cBest, j) > 0 {
		} else {
			plattCountSet.Set(cBest, j, 1.0)
		}

		plattAB.Set(0, j, plattASet.At(cBest, j)/plattCountSet.At(cBest, j))
		plattAB.Set(1, j, plattBSet.At(cBest, j)/plattCountSet.At(cBest, j))

	}
	return plattAB
}
func AccumPlatt(c int, colSum *mat64.Vector, plattAB *mat64.Dense, plattASet *mat64.Dense, plattBSet *mat64.Dense, plattCountSet *mat64.Dense) {
	tC := 0
	_, nCol := plattASet.Caps()
	for j := 0; j < nCol; j++ {
		if colSum.At(j, 0) == 1.0 {
			plattASet.Set(c, j, plattAB.At(0, tC)+plattASet.At(c, j))
			plattBSet.Set(c, j, plattAB.At(1, tC)+plattBSet.At(c, j))
			plattCountSet.Set(c, j, plattCountSet.At(c, j)+1.0)
			tC += 1
		}
	}
}
func Platt(trYhat *mat64.Dense, trY *mat64.Dense, tsYhat *mat64.Dense) (tsYhh *mat64.Dense, plattAB *mat64.Dense) {
	nRow, nCol := tsYhat.Caps()
	tsYhh = mat64.NewDense(nRow, nCol, nil)
	plattAB = mat64.NewDense(2, nCol, nil)
	for i := 0; i < nCol; i++ {
		trYhatCol, trYcol := minusValueFilterForPlatt(trYhat.ColView(i), trY.ColView(i))
		//trYhatColTMM, trYcolTMM := TmmFilterForPlatt(trYhatCol, trYcol)
		//fmt.Println(i, trYhat.ColView(i))
		//fmt.Println(i, trY.ColView(i))
		//trYhatColTMM, trYcolTMM := TmmFilterForPlatt(trYhat.ColView(i), trY.ColView(i))
		A, B := PlattParameterEst(trYhatCol, trYcol)
		//fmt.Println(i, A, B)
		plattAB.Set(0, i, A)
		plattAB.Set(1, i, B)
		yhh := PlattScale(tsYhat.ColView(i), A, B)
		tsYhh.SetCol(i, yhh)
	}
	return tsYhh, plattAB
}

func TmmFilterForPlatt(inTrYhat *mat64.Vector, inTrY *mat64.Vector) (trYhat *mat64.Vector, trY *mat64.Vector) {
	type kv struct {
		Key   int
		Value float64
	}
	n := inTrYhat.Len()
	var sortYh []kv
	for i := 0; i < n; i++ {
		ele := inTrYhat.At(i, 0)
		if math.IsNaN(ele) {
			sortYh = append(sortYh, kv{i, 0.0})
		} else {
			sortYh = append(sortYh, kv{i, inTrYhat.At(i, 0)})
		}
	}
	sort.Slice(sortYh, func(i, j int) bool {
		return sortYh[i].Value > sortYh[j].Value
	})

	up := int(float64(n) * 0.01)
	lp := int(float64(n) * 0.99)
	upThres := sortYh[up].Value
	lpThres := sortYh[lp].Value

	tmpTrYhat := make([]float64, 0)
	tmpTrY := make([]float64, 0)

	indexUP := 0
	sumLabel := 0.0
	tick := 0
	for i := 0; i < inTrYhat.Len(); i++ {
		if inTrYhat.At(i, 0) <= upThres && inTrYhat.At(i, 0) >= lpThres {
			tmpTrYhat = append(tmpTrYhat, inTrYhat.At(i, 0))
			tmpTrY = append(tmpTrY, inTrY.At(i, 0))
			if inTrYhat.At(i, 0) == upThres {
				indexUP = tick
			}
			tick += 1
			sumLabel += inTrY.At(i, 0)
		}
	}
	if sumLabel == 0.0 && upThres > 0 {
		//fmt.Println("tick: ", tick, up, lp, upThres, lpThres)
		//fmt.Println(inTrYhat)
		//fmt.Println(sortYh)
		//os.Exit(0)
		tmpTrY[indexUP] = 1.0
	}
	trYhat = mat64.NewVector(len(tmpTrYhat), tmpTrYhat)
	trY = mat64.NewVector(len(tmpTrY), tmpTrY)
	if upThres > 0 {
		return trYhat, trY
	} else {
		return inTrYhat, inTrY
	}
}

func minusValueFilterForPlatt(inTrYhat *mat64.Vector, inTrY *mat64.Vector) (trYhat *mat64.Vector, trY *mat64.Vector) {
	tmpTrYhat := make([]float64, 0)
	tmpTrY := make([]float64, 0)
	for i := 0; i < inTrYhat.Len(); i++ {
		if inTrYhat.At(i, 0) == -1.0 && inTrY.At(i, 0) == -1.0 {
			//do nothing
		} else {
			tmpTrYhat = append(tmpTrYhat, inTrYhat.At(i, 0))
			tmpTrY = append(tmpTrY, inTrY.At(i, 0))
		}
	}
	trYhat = mat64.NewVector(len(tmpTrYhat), tmpTrYhat)
	trY = mat64.NewVector(len(tmpTrY), tmpTrY)
	return trYhat, trY
}

func PlattScaleSet(Yh *mat64.Dense, plattAB *mat64.Dense) (Yhh *mat64.Dense) {
	nRow, nCol := Yh.Caps()
	Yhh = mat64.NewDense(nRow, nCol, nil)
	for i := 0; i < nCol; i++ {
		yhh := PlattScale(Yh.ColView(i), plattAB.At(0, i), plattAB.At(1, i))
		Yhh.SetCol(i, yhh)
	}
	return Yhh
}

func PlattScaleSetPseudoLabel(tsYhat *mat64.Dense, trYdata *mat64.Dense, thres *mat64.Dense) (tsYhat2 *mat64.Dense, thres2 *mat64.Dense) {
	nRow, nCol := trYdata.Caps()
	//trY sum
	trYcolSum := make([]float64, nCol)
	trYsum := 0.0
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if trYdata.At(i, j) == 1.0 {
				trYcolSum[j] += 1.0
				trYsum += 1.0
			}
		}
	}
	//rescale to max as 1
	max := 0.0
	for j := 0; j < nCol; j++ {
		if trYcolSum[j] > max {
			max = trYcolSum[j]
		}
	}
	for j := 0; j < nCol; j++ {
		trYcolSum[j] = trYcolSum[j] / max
	}
	//tsYsum
	nRow, nCol = tsYhat.Caps()
	tsYhat2 = mat64.NewDense(nRow, nCol, nil)
	tsYhatColSum := make([]float64, nCol)
	tsYhatSum := 0.0
	thres2 = mat64.NewDense(1, nCol, nil)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if tsYhat.At(i, j) > 0.5 {
				tsYhatColSum[j] += 1.0
				tsYhatSum += 1.0
			}
		}
	}

	//ratio and scale
	ratio := tsYhatSum / trYsum
	for j := 0; j < nCol; j++ {
		nPos := ratio * tsYhatColSum[j]
		if nPos < 1.0 {
			nPos = 1.0
		}
		yLabelVec := pseudoLabel(int(nPos+0.5), tsYhat.ColView(j))
		A, B := PlattParameterEst(tsYhat.ColView(j), yLabelVec)
		yhh := PlattScale(tsYhat.ColView(j), A, B)
		tsYhat2.SetCol(j, yhh)
		thres2.Set(0, j, 1.0/(1.0+math.Exp(A*thres.At(0, j)+B)))
	}
	return tsYhat2, thres2
}

func pseudoLabel(nPos int, Yh *mat64.Vector) (pseudoY *mat64.Vector) {
	type kv struct {
		Key   int
		Value float64
	}
	n := Yh.Len()
	var sortYh []kv
	for i := 0; i < n; i++ {
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

	thres := sortYh[nPos].Value
	pYSlice := make([]float64, n)
	for i := 0; i < n; i++ {
		if Yh.At(i, 0) >= thres {
			pYSlice[i] = 1.0
		}
	}
	pseudoY = mat64.NewVector(n, pYSlice)
	return pseudoY
}

func PlattScale(Yh *mat64.Vector, A float64, B float64) (Yhh []float64) {
	len := Yh.Len()
	Yhh = make([]float64, 0)
	//max := 0.0
	//min := 0.0
	for i := 0; i < len; i++ {
		ele := 1.0 / (1.0 + math.Exp(A*Yh.At(i, 0)+B))
		Yhh = append(Yhh, ele)
		//if ele > max {
		//	max = ele
		//}
		//if ele < min {
		//	min = ele
		//}
	}

	//if max < 0.99999 {
	//	scale := max - min
	//	for i := 0; i < len; i++ {
	//		Yhh[i] = (Yhh[i] - min) / scale
	//	}
	//
	//}
	return Yhh
}

func PlattParameterEst(Yh *mat64.Vector, Y *mat64.Vector) (A float64, B float64) {
	//init
	p1 := 0.0
	p0 := 0.0
	iter := 0
	maxIter := 100
	minStep := 0.0000000001
	sigma := 0.000000000001
	len := Y.Len()
	fApB := 0.0
	t := mat64.NewVector(len, nil)
	for i := 0; i < len; i++ {
		if Y.At(i, 0) == 1 {
			p1 += 1
		} else {
			p0 += 1
		}
	}
	hiTarget := (p1 + 1.0) / (p1 + 2.0)
	loTarget := 1 / (p0 + 2.0)
	//init parameter
	A = 0.0
	B = math.Log((p0 + 1.0) / (p1 + 1.0))
	fval := 0.0
	for i := 0; i < len; i++ {
		if Y.At(i, 0) == 1 {
			t.SetVec(i, hiTarget)
		} else {
			t.SetVec(i, loTarget)
		}
	}
	for i := 0; i < len; i++ {
		fApB = Yh.At(i, 0)*A + B
		if fApB >= 0 {
			fval += t.At(i, 0)*fApB + math.Log(1+math.Exp(0-fApB))
		} else {
			fval += (t.At(i, 0)-1)*fApB + math.Log(1+math.Exp(fApB))
		}
	}
	//iterations
	for iter <= maxIter {
		h11, h22, h21, g1, g2 := sigma, sigma, 0.0, 0.0, 0.0
		p, q, d1, d2 := 0.0, 0.0, 0.0, 0.0
		//Gradient and Hessian
		for i := 0; i < len; i++ {
			fApB = Yh.At(i, 0)*A + B
			if fApB >= 0 {
				p = math.Exp(0-fApB) / (1.0 + math.Exp(0-fApB))
				q = 1.0 / (1.0 + math.Exp(0-fApB))
			} else {
				p = 1.0 / (1.0 + math.Exp(fApB))
				q = math.Exp(fApB) / (1.0 + math.Exp(fApB))
			}
			d2 = p * q
			h11 += Yh.At(i, 0) * Yh.At(i, 0) * d2
			h22 += d2
			h21 += Yh.At(i, 0) * d2
			d1 = t.At(i, 0) - p
			g1 += Yh.At(i, 0) * d1
			g2 += d1
		}
		//stop criteria
		if math.Abs(g1) < 0.00001 && math.Abs(g2) < 0.00001 {
			break
		}
		//Compute modified Newton directions
		det := h11*h22 - h21*h21
		dA := 0 - (h22*g1-h21*g2)/det
		dB := 0 - (0-h21*g1+h11*g2)/det
		gd := g1*dA + g2*dB
		stepSize := 1.0
		for stepSize >= minStep {
			newA := A + stepSize*dA
			newB := B + stepSize*dB
			newf := 0.0
			for i := 0; i < len; i++ {
				fApB = Yh.At(i, 0)*newA + newB
				if fApB >= 0 {
					newf += t.At(i, 0)*fApB + math.Log(1.0+math.Exp(0-fApB))
				} else {
					newf += (t.At(i, 0)-1.0)*fApB + math.Log(1.0+math.Exp(1+fApB))
				}
			}
			if newf < (fval + 0.0001*stepSize*gd) {
				A = newA
				B = newB
				fval = newf
				break
			} else {
				stepSize /= 2.0
			}
		}
		if stepSize < minStep {
			//fail
			break
		}
		iter++
	}
	if iter >= maxIter {
		//reach max
	}
	return A, B
}

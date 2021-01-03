package src

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"github.com/wangjohn/quickselect"
	"log"
	"math"
	"runtime"
	"sort"
	"sync"
)

type kv struct {
	Key   int
	Value float64
}

func PerLabelBestCalibrated(cBestArr []int, YhPlattSetCalibrated map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense) (YhPlattCalibrated *mat64.Dense, yPlatt *mat64.Dense) {
	nRow, nCol := YhPlattSetCalibrated[0].Caps()
	YhPlattCalibrated = mat64.NewDense(nRow, nCol, nil)
	yPlatt = mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < len(cBestArr); j++ {
		cBest := cBestArr[j]
		for i := 0; i < nRow; i++ {
			YhPlattCalibrated.Set(i, j, YhPlattSetCalibrated[cBest].At(i, j))
			yPlatt.Set(i, j, yPlattSet[cBest].At(i, j))
		}
	}
	return YhPlattCalibrated, yPlatt
}
func PerlLabelQuantileNorm(YhSet map[int]*mat64.Dense, cBestArr []int) (tsYhat *mat64.Dense) {
	nRow, nCol := YhSet[0].Caps()
	tsYhat = mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < len(cBestArr); j++ {
		cBest := cBestArr[j]
		//quantile norm for each label in their label group, matching the situation in training
		tmpMat, _ := QuantileNorm(YhSet[cBest], mat64.NewDense(0, 0, nil), false)
		for i := 0; i < nRow; i++ {
			tsYhat.Set(i, j, tmpMat.At(i, j))
		}

	}
	return tsYhat
}

func PerLabelScaleSet(YhSet map[int]*mat64.Dense, plattABset map[int]*mat64.Dense, cBestArr []int) (Yhh *mat64.Dense) {
	nRow, nCol := YhSet[0].Caps()
	Yhh = mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < len(cBestArr); j++ {
		cBest := cBestArr[j]
		//plattABset index is for all labels only, not all Cs
		//YhSet[cBest], _ = QuantileNorm(YhSet[cBest], mat64.NewDense(0, 0, nil), false)
		log.Print("platt parameters:", cBest, plattABset[j].At(0, j), plattABset[j].At(1, j))
		LogColSum(YhSet[cBest])
		tsYhatTmp := PlattScaleSet(YhSet[cBest], plattABset[j])
		LogColSum(tsYhatTmp)
		tsYhatTmp, _ = QuantileNorm(tsYhatTmp, mat64.NewDense(0, 0, nil), false)
		//tsYhatTmp, _ := QuantileNorm(YhSet[cBest], mat64.NewDense(0, 0, nil), false)
		LogColSum(tsYhatTmp)
		//yhh := PlattScale(YhSet[cBest].ColView(j), plattAB[j].At(0, j), plattAB[j].At(1, j))
		for i := 0; i < nRow; i++ {
			Yhh.Set(i, j, tsYhatTmp.At(i, j))
		}
	}
	return Yhh
}

func QuantileNorm(data *mat64.Dense, thresData *mat64.Dense, isTransThres bool) (normData *mat64.Dense, normThresData *mat64.Dense) {
	nRow, nCol := data.Caps()
	//colSum
	colsum := make([]float64, nCol)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			colsum[j] += data.At(i, j)
		}
	}
	//init
	normData = mat64.NewDense(nRow, nCol, nil)
	rowTotalData := mat64.NewDense(0, 0, nil)
	if isTransThres {
		rowTotalData = mat64.NewDense(nRow+1, 1, nil)
	} else {
		rowTotalData = mat64.NewDense(nRow, 1, nil)
	}
	normThresData = mat64.NewDense(1, nCol, nil)
	//sort and fill
	for j := 0; j < nCol; j++ {
		//skip all zero value label column
		if AlmostEqual(colsum[j], 0.0) {
			continue
		}
		var sortYh []kv
		for i := 0; i < nRow; i++ {
			sortYh = append(sortYh, kv{i, data.At(i, j)})
		}
		if isTransThres {
			sortYh = append(sortYh, kv{nRow, thresData.At(0, j)})
		}
		sort.Slice(sortYh, func(i, j int) bool {
			return sortYh[i].Value > sortYh[j].Value
		})
		//fill normData with rank, tmp, but keep aggregated raw values
		for i := 0; i <= nRow; i++ {
			if isTransThres && sortYh[i].Key == nRow {
				normThresData.Set(0, j, float64(i))
			} else {
				if !isTransThres && i == nRow {
					//skip last row if no thres data
					//as no nRow index for sortYh and matrix
				} else {
					rowTotalData.Set(i, 0, rowTotalData.At(i, 0)+sortYh[i].Value)
					normData.Set(sortYh[i].Key, j, float64(i))
				}

			}
		}
	}
	//mean value, max, min, mm
	max := -999999999999.9
	min := 999999999999.9
	for i := 0; i <= nRow; i++ {
		if !isTransThres && i == nRow {
			//skip last row if no thres data
			//as no nRow index for sortYh and matrix
		} else {
			value := rowTotalData.At(i, 0)
			if value > max {
				max = value
			}
			if value < min {
				min = value
			}
		}
	}
	mm := max - min
	//scale to 0-1
	for i := 0; i <= nRow; i++ {
		if !isTransThres && i == nRow {
			//skip last row if no thres data
			//as no nRow index for sortYh and matrix
		} else {
			value := rowTotalData.At(i, 0)
			rowTotalData.Set(i, 0, (value-min)/mm)
		}
	}
	//fill normValue by rank

	for j := 0; j < nCol; j++ {
		for i := 0; i <= nRow; i++ {
			if isTransThres && i == nRow {
				if AlmostEqual(colsum[j], 0.0) {
					normThresData.Set(0, j, 1.0)
				} else {
					rank := int(normThresData.At(0, j))
					normThresData.Set(0, j, rowTotalData.At(rank, 0))
				}
			} else {
				if !isTransThres && i == nRow {
					//skip last row if no thres data
					//as no nRow index for sortYh and matrix
				} else if !AlmostEqual(colsum[j], 0.0) {
					rank := int(normData.At(i, j))
					normData.Set(i, j, rowTotalData.At(rank, 0))
				}
			}
		}
	}
	return normData, normThresData
}

func ColMaxMin(data *mat64.Dense) {
	nRow, nCol := data.Caps()
	for j := 0; j < nCol; j++ {
		max := -1.0
		min := 1.0
		for i := 0; i < nRow; i++ {
			value := data.At(i, j)
			if value > max {
				max = value
			}
			if value < min {
				min = value
			}
		}
		fmt.Println(j, "max: ", max, "min: ", min)
	}

}
func LabelRelationship(trYdata *mat64.Dense) (posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, transLabels *mat64.Dense) {
	nRow, nCol := trYdata.Caps()
	posLabelRls = mat64.NewDense(nCol, nCol, nil)
	negLabelRls = mat64.NewDense(nCol, nCol, nil)
	transLabels = mat64.NewDense(nCol, nCol, nil)
	for j := 0; j < nCol; j++ {
		//colSums
		jColSum := 0.0
		for i := 0; i < nRow; i++ {
			if trYdata.At(i, j) == 1.0 {
				jColSum += 1.0
			}
		}
		term2 := jColSum / float64(nRow)
		//comment out, for simplicity in MultiLabelRecalibrate, for k = j + 1; k < nCol; k++ {
		for k := 0; k < nCol; k++ {
			if k != j {
				nPos := 0.0
				nNeg := 0.0
				//conditional positive influence from label k to label j
				nConPos := 0.0
				nConNeg := 0.0
				for i := 0; i < nRow; i++ {
					if trYdata.At(i, k) == 1.0 {
						nPos += 1.0
						if trYdata.At(i, j) == 1.0 {
							nConPos += 1.0
						}
					} else {
						nNeg += 1.0
						if trYdata.At(i, j) == 1.0 {
							nConNeg += 1.0
						}
					}
				}
				//pos and neg calculation
				posRls := nConPos/nPos - term2
				negRls := nConNeg/nNeg - term2
				posLabelRls.Set(k, j, posRls)
				negLabelRls.Set(k, j, negRls)
				//transition probability from label j to label k
				transLabels.Set(k, j, nConPos/jColSum)
			} else {
				//transLabels.Set(k, j, 1.0)
			}
		}
	}
	// note this is an upper right filled matrix and the influence go from row index to col index
	return posLabelRls, negLabelRls, transLabels
}

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
func Platt(trYhat *mat64.Dense, trY *mat64.Dense, tsYhat *mat64.Dense, lamdaArr []float64) (tsYhh *mat64.Dense, plattAB *mat64.Dense, mseArr []float64) {
	nRow, nCol := tsYhat.Caps()
	tsYhh = mat64.NewDense(nRow, nCol, nil)
	mseArr = make([]float64, 0)
	plattAB = mat64.NewDense(2, nCol, nil)
	for i := 0; i < nCol; i++ {
		trYhatCol, trYcol := minusValueFilterForPlatt(trYhat.ColView(i), trY.ColView(i))
		trYhatColTMM, trYcolTMM := TmmFilterForPlatt(trYhatCol, trYcol, lamdaArr[i])
		A, B := PlattParameterEst(trYhatColTMM, trYcolTMM)
		plattAB.Set(0, i, A)
		plattAB.Set(1, i, B)
		tmpCol := minusValueFilter(tsYhat.ColView(i))
		yhh := PlattScale(tmpCol, A, B)
		tsYhh.SetCol(i, yhh)
		mse := MSE(trYcol, trYhatCol)
		mseArr = append(mseArr, mse)
	}
	return tsYhh, plattAB, mseArr
}

func MSE(trY *mat64.Vector, trYh *mat64.Vector) (mse float64) {
	mse = 0.0
	for i := 0; i < trY.Len(); i++ {
		if trYh.At(i, 0) < 1.0 {
			mse += math.Pow(trY.At(i, 0)-trYh.At(i, 0), 2)
		} else {
			//mse add zero
		}
	}
	mse = mse / float64(trY.Len())
	return mse
}

//chop scale to 0 - 1 range
func PlattChopScale(tsY *mat64.Dense, tsYhh *mat64.Dense) (maxArr []float64) {
	nRow, nCol := tsYhh.Caps()
	maxArr = make([]float64, 0)
	//arr := make([]float64, 0)
	//m := preprocessing.NewPowerTransformer()
	for j := 0; j < nCol; j++ {
		//colMat := mat.NewDense(nRow, 1, nil)
		//for i := 0; i < nRow; i++ {
		//	colMat.Set(i, 0, tsYhh.At(i, j))
		//}
		//colMat2, _ := m.FitTransform(colMat, nil)

		max := -100000000.0
		min := 100000000.0
		//var sortYh []kv
		for i := 0; i < nRow; i++ {
			//arr = append(arr, tsYhh.At(i, j))
			ele := tsYhh.At(i, j)
			//fix NaN
			if math.IsNaN(ele) {
				ele = 0.0
				tsYhh.Set(i, j, 0.0)
			}
			//min max
			if tsY.At(i, j) == 1.0 && ele > max {
				max = ele
			}
			//add ele >0, avoiding gap between last non-zero to zero being large
			if ele < min {
				min = ele
			}
			//}

			//sortYh = append(sortYh, kv{i, ele})
		}
		//sort.Slice(sortYh, func(i, j int) bool {
		//	return sortYh[i].Value > sortYh[j].Value
		//})
		//min = sortYh[int(0.95*float64(nRow))].Value
		//med, _ := stats.Median(arr)
		//for i := 0; i < nRow; i++ {
		//	arr[i] = math.Abs(arr[i] - med)
		//}
		//mad, _ := stats.Median(arr)
		//_, pAupr, _, thres := ComputeAupr(tsY.ColView(j), tsYhh.ColView(j), 0.1)
		//if pAupr > 0.1 {
		//	max = thres
		//}
		if max > 1.0 {
			max = 1.0
		}
		maxArr = append(maxArr, max)
		mm := max - min
		for i := 0; i < nRow; i++ {
			//ele := 0.5 + (tsYhh.At(i, j)-med)/(2.9652*mad)
			ele := (tsYhh.At(i, j) - min) / mm
			if ele > 1.0 {
				tsYhh.Set(i, j, 1.0)
			} else {
				//ele = math.Log(ele)
				tsYhh.Set(i, j, ele)
			}
			//} else {
			//	tsYhh.Set(i, j, 0.0)
			//}
			//tsYhh.Set(i, j, colMat2.At(i, 0))
		}
	}
	return maxArr
}

//chop scale to 0 - 1 range
func TestDataPlattChopScale(tsYhh *mat64.Dense, maxArr []float64) {
	nRow, nCol := tsYhh.Caps()
	for j := 0; j < nCol; j++ {

		min := 100000000.0
		max := -100000000.0
		//var sortYh []kv
		for i := 0; i < nRow; i++ {
			ele := tsYhh.At(i, j)
			if ele < min && ele > 0.0 {
				min = ele
			}
			if ele > max {
				max = ele
			}
			//sortYh = append(sortYh, kv{i, ele})

		}
		//if j == 0 {
		//	fmt.Println("train max: ", maxArr[j])
		//}
		if maxArr[j] > max {
			maxArr[j] = max
		}
		//sort.Slice(sortYh, func(i, j int) bool {
		//	return sortYh[i].Value > sortYh[j].Value
		//})
		//min = sortYh[int(0.95*float64(nRow))].Value
		mm := maxArr[j] - min
		//if j == 0 {
		//fmt.Println("test max: ", maxArr[j], min, max)
		//for i := 0; i < nRow; i++ {
		//	fmt.Println("ele:\t", tsYhh.At(i, j))
		//}
		//}
		for i := 0; i < nRow; i++ {
			ele := tsYhh.At(i, j)
			if ele >= maxArr[j] {
				tsYhh.Set(i, j, 1.0)
				//fmt.Println("fromTo:", ele, 1.0)
			} else {
				tsYhh.Set(i, j, (ele-min)/mm)
				//fmt.Println("fromTo:", ele, (ele-min)/mm)
			}
		}
	}
}
func TmmFilterForPlatt(inTrYhat *mat64.Vector, inTrY *mat64.Vector, lamda float64) (trYhat *mat64.Vector, trY *mat64.Vector) {
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

	up := int(float64(n) * lamda)
	//lp := int(float64(n) * 0.99)
	lp := n - 1
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

func minusValueFilter(inData *mat64.Vector) (outData *mat64.Vector) {
	tmpOut := make([]float64, 0)
	for i := 0; i < inData.Len(); i++ {
		if inData.At(i, 0) == -1.0 {
			tmpOut = append(tmpOut, 0.0)
		} else {
			tmpOut = append(tmpOut, inData.At(i, 0))
		}
	}
	outData = mat64.NewVector(len(tmpOut), tmpOut)
	return outData
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
	//rescale to 0-1
	//vectorRescale(Yh)
	len := Yh.Len()
	Yhh = make([]float64, 0)
	max := 0.0
	min := 1.0
	for i := 0; i < len; i++ {
		ele := 1.0 / (1.0 + math.Exp(A*Yh.At(i, 0)+B))
		Yhh = append(Yhh, ele)
		if ele > max {
			max = ele
		}
		if ele < min && ele > 0.0 {
			min = ele
		}
	}

	//init scale
	scale := max - min
	//if scale == 0
	if scale == 0.0 {
		scale = max
	}
	for i := 0; i < len; i++ {
		Yhh[i] = (Yhh[i] - min) / scale
	}

	return Yhh
}

func PlattParameterEst(Yh *mat64.Vector, Y *mat64.Vector) (A float64, B float64) {
	//init
	p1 := 0.0
	p0 := 0.0
	iter := 0
	maxIter := 1000
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

func YhPlattSetUpdate(iFold int, c int, YhPlattSetCalibrated map[int]*mat64.Dense, tsYhat *mat64.Dense, iFoldmat *mat64.Dense) {
	nRow := 0
	nRowTotal, nCol := YhPlattSetCalibrated[c].Caps()
	for i := 0; i < nRowTotal; i++ {
		if iFoldmat.At(i, 0) == float64(iFold) {
			for j := 0; j < nCol; j++ {
				YhPlattSetCalibrated[c].Set(i, j, tsYhat.At(nRow, j))
			}
			nRow += 1
		}
	}
}

func SubSetTrain(iFold int, Y *mat64.Dense, Yh *mat64.Dense, predBinY *mat64.Dense, X *mat64.Dense, iFoldmat *mat64.Dense) (yPlattTrain *mat64.Dense, yPredTrain *mat64.Dense, xTrain *mat64.Dense, xTest *mat64.Dense, tsYhat *mat64.Dense, tsYfold *mat64.Dense) {
	nRow := 0
	nRowTotal, nCol := Y.Caps()
	_, nColX := X.Caps()
	for i := 0; i < nRowTotal; i++ {
		if iFoldmat.At(i, 0) != float64(iFold) {
			nRow += 1
		}
	}

	yPlattTrain = mat64.NewDense(nRow, nCol, nil)
	yPredTrain = mat64.NewDense(nRow, nCol, nil)
	xTrain = mat64.NewDense(nRow, nColX, nil)
	xTest = mat64.NewDense(nRowTotal-nRow, nColX, nil)
	tsYhat = mat64.NewDense(nRowTotal-nRow, nCol, nil)
	tsYfold = mat64.NewDense(nRowTotal-nRow, nCol, nil)
	nRow = 0
	nRowTest := 0
	for i := 0; i < nRowTotal; i++ {
		if iFoldmat.At(i, 0) != float64(iFold) {
			for j := 0; j < nCol; j++ {
				yPlattTrain.Set(nRow, j, Y.At(i, j))
				yPredTrain.Set(nRow, j, predBinY.At(i, j))
			}
			for j := 0; j < nColX; j++ {
				xTrain.Set(nRow, j, X.At(i, j))
			}
			nRow += 1
		} else {
			for j := 0; j < nCol; j++ {
				tsYhat.Set(nRowTest, j, Yh.At(i, j))
				tsYfold.Set(nRowTest, j, Y.At(i, j))
			}
			for j := 0; j < nColX; j++ {
				xTest.Set(nRowTest, j, X.At(i, j))
			}
			nRowTest += 1
		}
	}
	//from  yPlattSet[c], yPredSet[c], xSet[c],  xSet[c] ,YhPlattSet[c], yPlattSet[c]
	//from  Y(train)    , predBinY   , X(train), X(test) ,Yh,            Y(test)
	return yPlattTrain, yPredTrain, xTrain, xTest, tsYhat, tsYfold
}
func PerlLabelMultiLabelRecalibrate(YhSet map[int]*mat64.Dense, cBestArr []int, kNN int, trainMeasure *mat64.Dense, xTest *mat64.Dense, yPlattSet map[int]*mat64.Dense, yPredSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, plattABset map[int]*mat64.Dense, thresSet map[int]*mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, wg *sync.WaitGroup, mutex *sync.Mutex) (tsYhatCal *mat64.Dense) {

	nRow, nCol := YhSet[cBestArr[0]].Caps()
	nRowTr, nColY := yPlattSet[cBestArr[0]].Caps()
	_, nColX := xSet[cBestArr[0]].Caps()
	tsYhatCal = mat64.NewDense(nRow, nCol, nil)
	yPlattAL := mat64.NewDense(nRowTr, nColY, nil)
	yPredAL := mat64.NewDense(nRowTr, nColY, nil)
	xAL := mat64.NewDense(nRowTr, nColX, nil)
	for j := 0; j < len(cBestArr); j++ {
		cBest := cBestArr[j]
		//scales, plattABset and thresSet is for all label, not all Cs
		tsYhat := PlattScaleSet(YhSet[cBest], plattABset[j])
		tsYhat, _ = QuantileNorm(tsYhat, mat64.NewDense(0, 0, nil), false)
		tsYhat, _ = SoftThresScale(tsYhat, thresSet[j])
		//top labels
		macroAuprSet := make([]float64, 0)
		for p := 0; p < len(cBestArr); p++ {
			idx := 13 + p*2
			macroAuprSet = append(macroAuprSet, trainMeasure.At(cBest, idx))
		}
		kNNidx := TopKnnLabelIdx(macroAuprSet, 0.5)
		//y,yPred and x for this cBest
		for i := 0; i < nRowTr; i++ {
			for l := 0; l < len(cBestArr); l++ {
				yPlattAL.Set(i, l, yPlattSet[cBest].At(i, l))
				yPredAL.Set(i, l, yPredSet[cBest].At(i, l))
			}
			for k := 0; k < nColX; k++ {
				xAL.Set(i, k, xSet[cBest].At(i, k))
			}
		}
		log.Print("col: ", j, ", cBest: ", cBest)
		LogColSum(tsYhat)
		tsYhatCalTmp := MultiLabelRecalibrate(kNN, tsYhat, xTest, yPlattAL, yPredAL, xAL, posLabelRls, negLabelRls, kNNidx, wg, mutex)
		for i := 0; i < nRow; i++ {
			tsYhatCal.Set(i, j, tsYhatCalTmp.At(i, j))
		}
		LogColSum(tsYhatCal)
	}
	return tsYhatCal
}
func MultiLabelRecalibrate(kNN int, tsYhat *mat64.Dense, xTest *mat64.Dense, yPlattTrain *mat64.Dense, yPredTrain *mat64.Dense, xTrain *mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, kNNidx map[int]int, wg *sync.WaitGroup, mutex *sync.Mutex) (tsYhatCal *mat64.Dense) {
	nRow, nCol := tsYhat.Caps()
	tsYhatCal = mat64.NewDense(nRow, nCol, nil)
	nTrainRow, _ := xTrain.Caps()
	if kNN >= nTrainRow {
		kNN = nTrainRow - 1
		log.Print("number of nearest neighbors is less than all training instances. Reducing...")
	}
	wg.Add(nRow)
	for i := 0; i < nRow; i++ {
		go single_MultiLabelRecalibrate(kNN, i, nCol, tsYhatCal, tsYhat, xTest, xTrain, yPlattTrain, yPredTrain, posLabelRls, negLabelRls, kNNidx, wg, mutex)
	}
	wg.Wait()
	runtime.GC()
	return tsYhatCal
}

func single_MultiLabelRecalibrate(kNN int, i int, nCol int, tsYhatCal *mat64.Dense, tsYhat *mat64.Dense, xTest *mat64.Dense, xTrain *mat64.Dense, yPlattTrain *mat64.Dense, yPredTrain *mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, kNNidx map[int]int, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	idxArr := DistanceTopK(kNN, i, xTest, xTrain)
	weight1 := 0.0
	weight2 := 0.0
	labelRls := 0.0
	for j := 0; j < nCol; j++ {
		for k := 0; k < kNN; k++ {
			if yPlattTrain.At(idxArr[k], j) == yPredTrain.At(idxArr[k], j) {
				weight1 += 1.0
			}
		}
		weight1 = weight1 / float64(kNN)
		weight2 = 1.0 - weight1
		//label influence
		labelRls = 0.0
		for m := 0; m < nCol; m++ {
			_, ok := kNNidx[m]
			if ok {
				// m == j cases are set as zero in LabelRelationship
				labelRls += tsYhat.At(i, m)*posLabelRls.At(m, j) + (1-tsYhat.At(i, m))*negLabelRls.At(m, j)
			}
		}
		prob := weight1*tsYhat.At(i, j) + weight2*labelRls/float64(len(kNNidx)-1)
		if prob < 0.0 {
			prob = 0.0
		}
		mutex.Lock()
		tsYhatCal.Set(i, j, prob)
		mutex.Unlock()
	}
}
func MultiLabelRecalibrate_SingleThread(kNN int, tsYhat *mat64.Dense, xTest *mat64.Dense, yPlattTrain *mat64.Dense, yPredTrain *mat64.Dense, xTrain *mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, kNNidx map[int]int) (tsYhatCal *mat64.Dense) {
	nRow, nCol := tsYhat.Caps()
	tsYhatCal = mat64.NewDense(nRow, nCol, nil)
	nTrainRow, _ := xTrain.Caps()
	if kNN >= nTrainRow {
		kNN = nTrainRow - 1
		log.Print("number of nearest neighbors is less than all training instances. Reducing...")
	}
	for i := 0; i < nRow; i++ {
		//go single_MultiLabelRecalibrate(kNN, i, nCol, tsYhatCal, tsYhat, xTest, xTrain, yPlattTrain, yPredTrain, posLabelRls, negLabelRls, wg, mutex)
		idxArr := DistanceTopK(kNN, i, xTest, xTrain)
		weight1 := 0.0
		weight2 := 0.0
		labelRls := 0.0
		for j := 0; j < nCol; j++ {
			for k := 0; k < kNN; k++ {
				if yPlattTrain.At(idxArr[k], j) == yPredTrain.At(idxArr[k], j) {
					weight1 += 1.0
				}
			}
			weight1 = weight1 / float64(kNN)
			weight2 = 1.0 - weight1
			//label influence
			labelRls = 0.0
			for m := 0; m < nCol; m++ {
				_, ok := kNNidx[m]
				if ok {
					// m == j cases are set as zero in LabelRelationship
					labelRls += tsYhat.At(i, m)*posLabelRls.At(m, j) + (1-tsYhat.At(i, m))*negLabelRls.At(m, j)
				}
			}
			prob := weight1*tsYhat.At(i, j) + weight2*labelRls/float64(len(kNNidx)-1)
			if prob < 0.0 {
				prob = 0.0
			}
			tsYhatCal.Set(i, j, prob)
		}
	}
	return tsYhatCal
}

func TopKnnLabelIdx(macroAuprSet []float64, ratio float64) (kNNidx map[int]int) {
	var sortAupr []kv
	nIdx := int(ratio * float64(len(macroAuprSet)))
	kNNidx = make(map[int]int, nIdx)
	for i := 0; i < len(macroAuprSet); i++ {
		sortAupr = append(sortAupr, kv{i, macroAuprSet[i]})
	}
	sort.Slice(sortAupr, func(i, j int) bool {
		return sortAupr[i].Value > sortAupr[j].Value
	})

	for i := 0; i < len(macroAuprSet); i++ {
		if nIdx >= 0 {
			kNNidx[sortAupr[i].Key] = i
			nIdx -= 1
		}
	}
	return kNNidx
}

//select top k rows in yPredTrain that is closed to row rowIdx in tsYhat
//hassanat distance, both yPredTrain and tsYhat contain same number of cols
type ByValue []kv

func (a ByValue) Len() int           { return len(a) }
func (a ByValue) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByValue) Less(i, j int) bool { return a[i].Value < a[j].Value }
func DistanceTopK(k int, rowIdx int, tsYhat *mat64.Dense, yProbTrain *mat64.Dense) (idxArr []int) {

	var selectDis ByValue
	nRow, nCol := yProbTrain.Caps()
	//a and b init out for better mem efficient
	a := 0.0
	b := 0.0
	for i := 0; i < nRow; i++ {
		dis := 0.0
		for j := 0; j < nCol; j++ {
			//hassanat distance, note all value after propagation are larger than 0, thus no negative value considerred
			a = tsYhat.At(rowIdx, j)
			b = yProbTrain.At(i, j)
			dis = dis + 1.0 - (1.0+math.Min(a, b))/(1.0+math.Max(a, b))
		}
		selectDis = append(selectDis, kv{i, dis})
	}
	quickselect.QuickSelect(ByValue(selectDis), k)

	//reorder for minDis idx at first
	var sortMap []kv
	for i := 0; i < k; i++ {
		sortMap = append(sortMap, kv{selectDis[i].Key, selectDis[i].Value})
	}
	sort.Slice(sortMap, func(i, j int) bool {
		return sortMap[i].Value < sortMap[j].Value
	})
	//disArr normalized by maxDis
	for i := 0; i < k; i++ {
		idxArr = append(idxArr, sortMap[i].Key)
	}
	return idxArr
}

//rescale to 0-1
func vectorRescale(data *mat64.Vector) {
	len := data.Len()
	max := 0.0
	for i := 0; i < len; i++ {
		if data.At(i, 0) > max {
			max = data.At(i, 0)
		}
	}
	if max == 0.0 {
		max = 1.0
	}
	for i := 0; i < len; i++ {
		ele := data.At(i, 0)
		data.SetVec(i, ele/max)
	}
}

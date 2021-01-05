package src

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"gonum.org/v1/gonum/stat"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
)

//for cross validation
type CvFold struct {
	X        *mat64.Dense
	Y        *mat64.Dense
	IndAccum []int
}

func (f *CvFold) setIndAccum(indAccum []int) {
	f.IndAccum = indAccum
}
func (f *CvFold) setX(X *mat64.Dense) {
	f.X = X
}
func (f *CvFold) setY(Y *mat64.Dense) {
	f.Y = Y
}
func (f *CvFold) setXYinDecoding(idxArr []int, matX *mat64.Dense, vecY *mat64.Vector) {
	_, nColX := matX.Caps()
	nRow := len(idxArr)
	f.X = mat64.NewDense(nRow, nColX, nil)
	f.Y = mat64.NewDense(nRow, 1, nil)
	for i := 0; i < nRow; i++ {
		f.X.SetRow(i, matX.RawRowView(idxArr[i]))
		f.Y.Set(i, 0, vecY.At(idxArr[i], 0))
	}
}
func (f *CvFold) SetXYinNestedTraining(idxArr []int, matX *mat64.Dense, matY *mat64.Dense, indAccum []int) {
	_, nColX := matX.Caps()
	_, nColY := matY.Caps()
	nRow := len(idxArr)
	f.X = mat64.NewDense(nRow, nColX, nil)
	f.Y = mat64.NewDense(nRow, nColY, nil)
	for i := 0; i < nRow; i++ {
		f.X.SetRow(i, matX.RawRowView(idxArr[i]))
		f.Y.SetRow(i, matY.RawRowView(idxArr[i]))
	}
	f.IndAccum = indAccum
}

func ConsistencyIndAccum(trainFold []CvFold, testFold []CvFold, trXdata *mat64.Dense, tsXdata *mat64.Dense, indAccum []int) ([]CvFold, []CvFold, *mat64.Dense, *mat64.Dense, []int) {
	//refill trXdata,tsXdata
	tmpTrXdata := RefillIndCol(trXdata, indAccum)
	tmpTsXdata := RefillIndCol(tsXdata, indAccum)
	//minimum indAccum for all folds
	for f := 0; f < len(testFold); f++ {
		for i := 0; i < len(indAccum); i++ {
			if testFold[f].IndAccum[i] == 0 {
				indAccum[i] = 0
			}
		}
	}
	//colSum and colMax
	colSumData := make([]float64, len(indAccum))
	for i := 0; i < len(indAccum); i++ {
		if indAccum[i] == 0 {
			colSumData[i] = 0.0
		} else {
			colSumData[i] = 1.0
		}
	}
	colSum := mat64.NewVector(len(indAccum), colSumData)
	//modify Xs
	for f := 0; f < len(testFold); f++ {
		//refill empty cols
		tmpTrX := RefillIndCol(trainFold[f].X, testFold[f].IndAccum)
		tmpTsX := RefillIndCol(testFold[f].X, testFold[f].IndAccum)
		tmpTrX = PosSelect(tmpTrX, colSum)
		tmpTsX = PosSelect(tmpTsX, colSum)
		trainFold[f].setX(tmpTrX)
		testFold[f].setX(tmpTsX)
		testFold[f].setIndAccum(indAccum)
	}
	trXdata = PosSelect(tmpTrXdata, colSum)
	tsXdata = PosSelect(tmpTsXdata, colSum)
	return trainFold, testFold, trXdata, tsXdata, indAccum
}

func ConsistencyScale(trainFold []CvFold, testFold []CvFold, trXdata *mat64.Dense, tsXdata *mat64.Dense) ([]CvFold, []CvFold, *mat64.Dense, *mat64.Dense) {
	//colMax, colMin
	_, nCol := tsXdata.Caps()
	colMax := make([]float64, nCol)
	colMin := make([]float64, nCol)
	for j := 0; j < nCol; j++ {
		colMin[j] = 1.0
		nRow, _ := trXdata.Caps()
		for i := 0; i < nRow; i++ {
			if trXdata.At(i, j) > colMax[j] {
				colMax[j] = trXdata.At(i, j)
			}
			if trXdata.At(i, j) > 0.0 && trXdata.At(i, j) < colMin[j] {
				colMin[j] = trXdata.At(i, j)
			}
		}
		nRow, _ = tsXdata.Caps()
		for i := 0; i < nRow; i++ {
			if tsXdata.At(i, j) > colMax[j] {
				colMax[j] = tsXdata.At(i, j)
			}
			if tsXdata.At(i, j) > 0.0 && tsXdata.At(i, j) < colMin[j] {
				colMin[j] = tsXdata.At(i, j)
			}
		}
		for f := 0; f < len(testFold); f++ {
			nRow, _ := trainFold[f].X.Caps()
			for i := 0; i < nRow; i++ {
				if trainFold[f].X.At(i, j) > colMax[j] {
					colMax[j] = trainFold[f].X.At(i, j)
				}
				if trainFold[f].X.At(i, j) > 0.0 && trainFold[f].X.At(i, j) < colMin[j] {
					colMin[j] = trainFold[f].X.At(i, j)
				}
			}
			nRow, _ = testFold[f].X.Caps()
			for i := 0; i < nRow; i++ {
				if testFold[f].X.At(i, j) > colMax[j] {
					colMax[j] = testFold[f].X.At(i, j)
				}
				if testFold[f].X.At(i, j) > 0.0 && testFold[f].X.At(i, j) < colMin[j] {
					colMin[j] = testFold[f].X.At(i, j)
				}
			}
		}
	}
	//rescale
	for j := 0; j < nCol; j++ {
		scale := colMax[j] - colMin[j]
		if scale <= 0.0 {
			scale = 1.0
		}
		nRow, _ := tsXdata.Caps()
		for i := 0; i < nRow; i++ {
			if tsXdata.At(i, j) >= colMin[j] {
				tsXdata.Set(i, j, (tsXdata.At(i, j)-colMin[j])/scale)
			}
		}
		nRow, _ = trXdata.Caps()
		for i := 0; i < nRow; i++ {
			if trXdata.At(i, j) >= colMin[j] {
				trXdata.Set(i, j, (trXdata.At(i, j)-colMin[j])/scale)
			}
		}
		for f := 0; f < len(testFold); f++ {
			nRow, _ = trainFold[f].X.Caps()
			for i := 0; i < nRow; i++ {
				if trainFold[f].X.At(i, j) >= colMin[j] {
					trainFold[f].X.Set(i, j, (trainFold[f].X.At(i, j)-colMin[j])/scale)
				}
			}
			nRow, _ = testFold[f].X.Caps()
			for i := 0; i < nRow; i++ {
				if testFold[f].X.At(i, j) >= colMin[j] {
					testFold[f].X.Set(i, j, (testFold[f].X.At(i, j)-colMin[j])/scale)
				}
			}
		}
	}

	str := ""
	for j := 0; j < nCol; j++ {
		str = str + "\t" + fmt.Sprintf("%g", colMax[j])
		str = str + "/" + fmt.Sprintf("%g", colMin[j])
	}
	log.Print("col Max/Min:")
	log.Print(str)
	return trainFold, testFold, trXdata, tsXdata
}

func LogColSum(data *mat64.Dense) {
	nRow, nCol := data.Caps()
	sum := make([]float64, nCol)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if data.At(i, j) > 0.0 {
				sum[j] += data.At(i, j)
			}
		}
	}
	str := ""
	for j := 0; j < nCol; j++ {
		str = str + "\t" + fmt.Sprintf("%.2f", sum[j])
	}
	log.Print(str)
}
func RatioPosPerLabel(tmpTsYhat *mat64.Dense, tsYdata *mat64.Dense) (idxArr []int) {
	nRow, nCol := tsYdata.Caps()
	nPredPos := make([]float64, nCol)
	nPos := make([]float64, nCol)
	idxArr = make([]int, nCol)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if tsYdata.At(i, j) == 1.0 {
				nPos[j] += 1.0
			}
			if tmpTsYhat.At(i, j) > 0.5 {
				nPredPos[j] += 1.0
			}
		}
	}
	var sortMap []kv
	for i := 0; i < nCol; i++ {
		sortMap = append(sortMap, kv{i, nPredPos[i] / nPos[i]})
	}
	sort.Slice(sortMap, func(i, j int) bool {
		return sortMap[i].Value > sortMap[j].Value
	})
	for i := 0; i < nCol; i++ {
		idxArr[i] = sortMap[i].Key
	}
	return idxArr
}

func SubDataByPos(idx int, colSum map[int]float64, trXdata *mat64.Dense, trYdata *mat64.Dense) (trXsub *mat64.Dense, trYsub *mat64.Dense) {
	_, nColX := trXdata.Caps()
	nRowY, nColY := trYdata.Caps()
	nRow := int(colSum[idx])
	trXsub = mat64.NewDense(nRow, nColX, nil)
	trYsub = mat64.NewDense(nRow, nColY, nil)
	idxRow := 0
	for i := 0; i < nRowY; i++ {
		if trYdata.At(i, idx) == 1.0 {
			for j := 0; j < nColY; j++ {
				trYsub.Set(idxRow, j, trYdata.At(i, j))
			}
			for j := 0; j < nColX; j++ {
				trXsub.Set(idxRow, j, trXdata.At(i, j))
			}
			idxRow += 1
		}
	}
	return trXsub, trYsub
}

func HyperParameterSet(maxDim int, lbL float64, hbL float64, nStep int) (kSet []int, sigmaFctsSet []float64, lamdaSet []float64) {

	//sigmaFctsSet := []float64{4.0, 25.0, 100.0, 400.0, 10000.0}
	sigmaFctsSet = make([]float64, 0)
	lamdaSet = make([]float64, 0)
	kSet = make([]int, 0)
	//kMap := make(map[int]int)
	//for i := 75; i <= 95; i += 10 {
	//	k := maxDim * i / 100
	//	if k > 0 {
	//		_, isDefined := kMap[k]
	//		if !isDefined {
	//			kSet = append(kSet, k)
	//		}
	//		kMap[k] = k
	//	}
	//}
	kSet = append(kSet, maxDim-1)
	step := (hbL - lbL) / float64(nStep)
	for i := 0; i < nStep; i++ {
		lamda := lbL + float64(i)*step
		sigmaFctsSet = append(sigmaFctsSet, 1.0/(lamda*lamda))
		lamdaSet = append(lamdaSet, lamda)
	}
	return kSet, sigmaFctsSet, lamdaSet
}

func lamdaToSigmaFctsSet(lamdaSet []float64) (sigmaFctsSet []float64) {
	sigmaFctsSet = make([]float64, 0)
	for i := 0; i < len(lamdaSet); i++ {
		sigmaFcts := 1.0 / (lamdaSet[i] * lamdaSet[i])
		sigmaFctsSet = append(sigmaFctsSet, sigmaFcts)
	}
	return sigmaFctsSet
}

func (f *CvFold) setXY(pos []int, neg []int, matX *mat64.Dense, vecY *mat64.Vector) {
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

func ColStackMatrix(X *mat64.Dense, addX *mat64.Dense) *mat64.Dense {
	Xt := mat64.DenseCopyOf(X.T())
	X2 := mat64.NewDense(0, 0, nil)
	X2.Stack(addX.T(), Xt)
	X2t := mat64.DenseCopyOf(X2.T())
	return X2t
}

func NanFilter(data *mat64.Dense) (detectNanInf bool) {
	detectNanInf = false
	nRow, nCol := data.Caps()
	for r := 0; r < nRow; r++ {
		for c := 0; c < nCol; c++ {
			ele := data.At(r, c)
			if math.IsInf(ele, 0) || math.IsNaN(ele) {
				data.Set(r, c, 0.0)
				if !detectNanInf {
					detectNanInf = true
				}
			}
		}
	}
	return detectNanInf
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

func PosSelect(data *mat64.Dense, colSum *mat64.Vector) (dataSelected *mat64.Dense) {
	r, c := data.Caps()
	nCol := 0
	for j := 0; j < c; j++ {
		if colSum.At(j, 0) == 1.0 {
			nCol += 1
		}
	}
	dataSelected = mat64.NewDense(r, nCol, nil)
	tC := 0
	for j := 0; j < c; j++ {
		if colSum.At(j, 0) == 1.0 {
			for i := 0; i < r; i++ {
				dataSelected.Set(i, tC, data.At(i, j))
			}
			tC += 1
		}
	}
	return dataSelected
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

func cvSplitNoPerm(nElement int, nFold int) (cvSet map[int][]int) {
	//this is a tmp solution for cvSplit without perm, as number of positive is different for each label in feature filtering
	cvSet = make(map[int][]int)
	j := 0
	if nElement >= nFold {
		for i := 0; i < nElement; i++ {
			j = i % nFold
			cvSet[j] = append(cvSet[j], i)
		}
	} else {
		fmt.Println(nElement, "less than cv folds", nFold)
		os.Exit(1)
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

func NormScale(data *mat64.Dense) {
	nRow, nCol := data.Caps()
	for i := 0; i < nRow; i++ {
		max := 0.0
		for j := 0; j < nCol; j++ {
			if data.At(i, j) > max {
				max = data.At(i, j)
			}
		}
		for j := 0; j < nCol; j++ {
			ele := data.At(i, j) / max
			data.Set(i, j, ele)
		}
	}

}
func ColScale(data *mat64.Dense, rebaData *mat64.Dense) (scaleData *mat64.Dense) {
	nRow, nCol := data.Caps()
	scaleData = mat64.NewDense(nRow, nCol, nil)
	scaleValue := make([]float64, nCol)
	for j := 0; j < nCol; j++ {
		var sortYh []kv
		for i := 0; i < nRow; i++ {
			sortYh = append(sortYh, kv{i, data.At(i, j)})
		}
		sort.Slice(sortYh, func(i, j int) bool {
			return sortYh[i].Value > sortYh[j].Value
		})
		scaleIdx := int(2.0 * rebaData.At(0, j) * float64(nRow))
		for i := scaleIdx; i >= 0; i-- {
			if sortYh[i].Value > 0 {
				scaleValue[j] = sortYh[i].Value
				break
			}
		}
	}
	//incase maxValue as 0
	for i := 0; i < len(scaleValue); i++ {
		if scaleValue[i] == 0.0 {
			scaleValue[i] = 1.0
		}
	}
	//fmt.Println(scaleValue)

	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			value := data.At(i, j) / scaleValue[j]
			//value := data.At(i, j)
			//if value > 1.0 {
			//	scaleData.Set(i, j, 1.0)
			//} else {
			scaleData.Set(i, j, value)
			//}
		}
	}
	return scaleData
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

func CostSeneitiveAupr(Y *mat64.Vector, Yh *mat64.Vector, C *mat64.Vector) (aupr float64) {
	type kv struct {
		Key   int
		Value float64
		Cost  float64
	}
	n := Y.Len()
	mapY := make(map[int]int)
	total := 0.0
	var sortYh []kv
	for i := 0; i < n; i++ {
		if Y.At(i, 0) == 1.0 {
			mapY[i] = 1
			total += C.At(i, 0)
		}
		ele := Yh.At(i, 0)
		if math.IsNaN(ele) {
			sortYh = append(sortYh, kv{i, 0.0, C.At(i, 0)})
		} else {
			ele := Yh.At(i, 0)
			sortYh = append(sortYh, kv{i, ele, C.At(i, 0)})
		}
	}
	sort.Slice(sortYh, func(i, j int) bool {
		return sortYh[i].Value > sortYh[j].Value
	})

	p := 0.0
	tp := 0.0
	pr := 0.0
	re := 0.0
	prData := make([]float64, 0)
	for _, kv := range sortYh {
		cost := kv.Cost
		_, ok := mapY[kv.Key]
		p += cost
		//prediction is positive
		if ok {
			tp += cost
			pr = tp / p
			re = tp / total
			//record Aupr data
			prData = append(prData, pr)
			prData = append(prData, re)
		}
	}
	aupr = 0.0
	for i := 2; i < len(prData)-1; i += 2 {
		aupr += (prData[i] + prData[i-2]) * (prData[i+1] - prData[i-1])
	}
	return aupr
}
func ComputeAupr(Y *mat64.Vector, Yh *mat64.Vector, beta float64) (aupr float64, pAupr float64, maxFscore float64, optThres float64) {
	type kv struct {
		Key   int
		Value float64
	}
	n := Y.Len()
	mapY := make(map[int]int)
	sumZero := 0
	sumOne := 0
	var sortYh []kv
	for i := 0; i < n; i++ {
		if Y.At(i, 0) == 1.0 {
			mapY[i] = 1
		}
		ele := Yh.At(i, 0)
		if math.IsNaN(ele) {
			sortYh = append(sortYh, kv{i, 0.0})
			sumZero += 1
		} else {
			ele := Yh.At(i, 0)
			sortYh = append(sortYh, kv{i, ele})
			if ele >= 1.0 {
				sumOne += 1
			} else if ele <= 0.0 {
				sumZero += 1
			}
		}
	}

	//the all zero/one slice won't be sorted by default, resulting aartifact Aupr that can be large
	nCaseThres := int(float64(n) * 0.99)
	if sumOne > nCaseThres || sumZero > nCaseThres {
		return 0.0, 0.0, 0.0, 1.0
	}
	sort.Slice(sortYh, func(i, j int) bool {
		return sortYh[i].Value > sortYh[j].Value
	})

	all := 0.0
	p := 0.0
	tp := 0.0
	fp := 0.0
	tn := 0.0
	pr := 0.0
	re := 0.0
	beta1 := beta
	beta2 := 1.0 / beta
	invPr := 0.0
	invRe := 0.0
	fscore := 0.0
	invFscore := 0.0
	maxFscore = 0.0
	optThres = 1.0
	//tprAtMax := 0.0
	isThresDetermined := false
	total := float64(len(mapY))
	invTotal := float64(n - len(mapY))
	//inference 4 from qz for fixing init fscore zero with no pos at begin
	k := 0.0
	isFirstPositiveDetected := false
	prData := make([]float64, 0)
	//prDataPartial := make([]float64, 0)
	thresP := 0
	//thresBaseline := 0.0
	for _, kv := range sortYh {
		all += 1.0
		p += 1.0
		_, ok := mapY[kv.Key]
		//prediction is positive
		if ok {
			isFirstPositiveDetected = true
			tp += 1.0
			pr = tp / p
			re = tp / total
			//update Fscore 2
			fscore = (1 + beta1*beta1) * pr * re / (beta1*beta1*pr + re)
			//record Aupr data
			prData = append(prData, pr)
			prData = append(prData, re)
		} else {
			//prediction is negative
			if !isFirstPositiveDetected {
				k += 1
			}
			fp += 1.0
			tn = invTotal - fp
			if float64(n) > p {
				invPr = tn / (float64(n) - p)
				invRe = tn / invTotal
			}
			//update inv Fscore 0.5
			invFscore = (1 + beta2*beta2) * invPr * invRe / (beta2*beta2*invPr + invRe)
		}
		//inference 4 from qz for fixing init fscore zero with no pos at begin
		if !isFirstPositiveDetected && k > total {
			isThresDetermined = true
		}
		//adjusted geometric f-measure
		if !isThresDetermined {
			agFscore := math.Sqrt(fscore * invFscore)
			if agFscore > maxFscore {
				maxFscore = agFscore
				optThres = kv.Value
				thresP = len(prData)
			}
		}

	}
	aupr = 0.0
	pAupr = 0.0
	for i := 2; i < len(prData)-1; i += 2 {
		aupr += (prData[i] + prData[i-2]) * (prData[i+1] - prData[i-1])
	}
	for i := 2; i < thresP-1; i += 2 {
		pAupr += (prData[i] + prData[i-2]) * (prData[i+1] - prData[i-1])
	}
	aupr = aupr / 2
	pAupr = pAupr / 2

	//only one positive
	if len(prData) == 2 {
		aupr = prData[0] * prData[1]
		if thresP == 0 {
			pAupr = 0.0
		} else {
			pAupr = aupr
		}
	}

	if aupr < float64(len(mapY))/float64(n) {
		return aupr, pAupr, maxFscore, 1.0
	}
	return aupr, pAupr, maxFscore, optThres
}

func Flat(Y *mat64.Dense) (vec *mat64.Vector) {
	nR, nC := Y.Caps()
	vec = mat64.NewVector(nR*nC, nil)
	c := 0
	for i := 0; i < nC; i++ {
		for j := 0; j < nR; j++ {
			vec.SetVec(c, Y.At(j, i))
			c += 1
		}
	}
	return vec
}

func RankPred(Yh *mat64.Dense, thres *mat64.Dense) (rankYh *mat64.Dense, rankThres *mat64.Dense) {
	nRow, nCol := Yh.Caps()
	rankYh = mat64.NewDense(nRow, nCol, nil)
	rankThres = mat64.NewDense(1, nCol, nil)
	for c := 0; c < nCol; c++ {
		var sortYh []kv
		for r := 0; r < nRow; r++ {
			ele := Yh.At(r, c)
			if math.IsNaN(ele) {
				sortYh = append(sortYh, kv{r, 0.0})
			} else {
				sortYh = append(sortYh, kv{r, ele})
			}

		}
		sort.Slice(sortYh, func(i, j int) bool {
			return sortYh[i].Value > sortYh[j].Value
		})
		rankThres.Set(0, c, 1.0)
		for r := 0; r < nRow; r++ {
			value := (float64(nRow) - float64(r)) / float64(nRow)
			//largest rank in values abrove thres
			if rankThres.At(0, c) > value && thres.At(0, c) < sortYh[r].Value {
				rankThres.Set(0, c, value)
			}
		}
		for r := 0; r < nRow; r++ {
			value := (float64(nRow) - float64(r)) / float64(nRow)
			if rankThres.At(0, c) >= 1.0 {
				rankYh.Set(sortYh[r].Key, c, -1.0)

			} else if value < rankThres.At(0, c) {
				rankYh.Set(sortYh[r].Key, c, -1.0)
			} else {
				rankYh.Set(sortYh[r].Key, c, (value-rankThres.At(0, c))/(1.0-rankThres.At(0, c)))
			}

		}
		rankThres.Set(0, c, 0.0)

	}
	return rankYh, rankThres
}
func BinPredByAlpha(Yh *mat64.Dense, rankCut int, outBin bool) (binYh *mat64.Dense, detectNanInf bool) {
	detectNanInf = false
	tmpIntC := -1
	nRow, nCol := Yh.Caps()
	binYh = mat64.NewDense(nRow, nCol, nil)
	for r := 0; r < nRow; r++ {
		var sortYh []kv
		for c := 0; c < nCol; c++ {
			ele := Yh.At(r, c)
			if math.IsNaN(ele) {
				sortYh = append(sortYh, kv{c, 0.0})
			} else {
				sortYh = append(sortYh, kv{c, ele})
			}

		}
		sort.Slice(sortYh, func(i, j int) bool {
			return sortYh[i].Value > sortYh[j].Value
		})
		//so that it is rankCut +1 as golang is 0 based
		thres := sortYh[rankCut].Value
		//tied top rank as 1
		if rankCut == 1 && sortYh[rankCut].Value == sortYh[0].Value {
			thres -= 0.000001
		}
		//in case all zero
		if thres < 0.0 {
			thres = 0.0
		}
		for c := 0; c < nCol; c++ {
			tick := 0
			ele := Yh.At(r, c)
			if math.IsNaN(ele) {
				binYh.Set(r, c, 0.0)
				if !detectNanInf {
					detectNanInf = true
					tmpIntC = c
				}
			} else if Yh.At(r, c) > thres && tick < rankCut {
				if outBin {
					binYh.Set(r, c, 1.0)
				} else {
					binYh.Set(r, c, (Yh.At(r, c)))
				}
				tick += 1
			} else {
				binYh.Set(r, c, 0.0)
			}
			if math.IsInf(ele, 0) && !detectNanInf {
				detectNanInf = true
			}
		}
		if detectNanInf {
			log.Print("detectNanInf in col: ", tmpIntC, ", with rankCut: ", rankCut)
			LogColSum(Yh)
		}
	}
	return binYh, detectNanInf
}

func MaskZeroByThres(tsYhat *mat64.Dense, thresData *mat64.Dense) (tsYhat2 *mat64.Dense) {
	nRow, nCol := tsYhat.Caps()
	tsYhat2 = mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < nCol; j++ {
		for i := 0; i < nRow; i++ {
			if tsYhat.At(i, j) >= thresData.At(0, j) {
				tsYhat2.Set(i, j, tsYhat.At(i, j))
			}
		}
	}
	return tsYhat2
}

func PerLabelSoftThresScale(tsYhat *mat64.Dense, thresSet map[int]*mat64.Dense) (tsYhat2 *mat64.Dense, thresData2 *mat64.Dense) {
	_, nCol := tsYhat.Caps()
	thresData := mat64.NewDense(1, nCol, nil)
	for j := 0; j < nCol; j++ {
		thresData.Set(0, j, thresSet[j].At(0, j))
	}
	tsYhat2, thresData2 = SoftThresScale(tsYhat, thresData)
	return tsYhat2, thresData2
}

func SoftThresScale(tsYhat *mat64.Dense, thresData *mat64.Dense) (tsYhat2 *mat64.Dense, thresData2 *mat64.Dense) {
	nRow, nCol := tsYhat.Caps()
	tsYhat2 = mat64.NewDense(nRow, nCol, nil)
	thresData2 = mat64.NewDense(1, nCol, nil)
	for j := 0; j < nCol; j++ {
		max := 0.0
		ele := 0.0
		//zero thres case
		if thresData.At(0, j) < 0.0000001 {
			thresData.Set(0, j, 0.0000001)
		}
		//max ele
		for i := 0; i < nRow; i++ {
			if math.IsInf(thresData.At(0, j), 0) {
				ele = 0.0
			} else if thresData.At(0, j) == 1.0 {
				ele = 0.0
			} else {
				ele = tsYhat.At(i, j) / thresData.At(0, j)
			}
			if ele > max {
				max = ele
			}
		}
		//all zero case
		if max == 0.0 {
			max = 1.0
		}
		//scale ele
		for i := 0; i < nRow; i++ {
			if math.IsInf(thresData.At(0, j), 0) {
				ele = 0.0
			} else if thresData.At(0, j) == 1.0 || max < 1.0 {
				ele = 0.0
			} else {
				ele = tsYhat.At(i, j) / thresData.At(0, j)
			}
			//recale
			//(b-a)(ele-min)/(max-min)+a
			if ele >= 1.0 {
				ele = 0.5*(ele-1.0)/(max-1.0) + 0.5
			} else {
				ele = 0.5 * ele
			}

			tsYhat2.Set(i, j, ele)
		}
		thresData2.Set(0, j, 0.5)
	}
	return tsYhat2, thresData2
}

func Sigmoid(x float64) (y float64) {
	y = math.Exp(x) / (1 + math.Exp(x))
	return y
}

func ComputeF1_3(Y *mat64.Vector, Yh *mat64.Vector, thres float64) (F1 float64, tp int, fp int, fn int, tn int) {
	n := Y.Len()
	mapY := make(map[int]int)
	//skipY := make(map[int]int)
	//var sortYh []kv
	for i := 0; i < n; i++ {
		if Y.At(i, 0) == 1.0 {
			mapY[i] = 1
		}
		//if Ys.At(i, 0) == 1.0 {
		//	skipY[i] = 1
		//}
		//ele := Yh.At(i, 0)
		//if math.IsNaN(ele) {
		//	sortYh = append(sortYh, kv{i, 0.0})
		//} else {
		//	sortYh = append(sortYh, kv{i, Yh.At(i, 0)})
		//}
	}
	//sort.Slice(sortYh, func(i, j int) bool {
	//	return sortYh[i].Value > sortYh[j].Value
	//})
	//o based index, thus -1
	//thres := sortYh[rankCut-1].Value
	//var tp int
	//var fp int
	//var fn int
	//var tn int
	for i := 0; i < n; i++ {
		y := Y.At(i, 0)
		yh := Yh.At(i, 0)
		//_, exist := skipY[i]
		//if !exist {
		if y > 0 && yh > thres {
			tp += 1
		} else if y <= 0 && yh > thres {
			fp += 1
		} else if y > 0 && yh <= thres {
			fn += 1
		} else if y <= 0 && yh <= thres {
			tn += 1
		}
		//}
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
	return F1, tp, fp, fn, tn
}
func ComputeAccuracy(tsYdata *mat64.Dense, Yhat *mat64.Dense, isWeighted bool) (accuracy float64) {
	nRow, nCol := tsYdata.Caps()
	colSum := mat64.NewDense(1, nCol, nil)
	//matching benchmark for no pos but all neg row
	rowSum := mat64.NewDense(1, nRow, nil)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if tsYdata.At(i, j) == 1.0 {
				rowSum.Set(0, i, rowSum.At(0, i)+1)
			}
		}
	}
	nPosRow := 0.0
	for i := 0; i < nRow; i++ {
		if rowSum.At(0, i) > 0.0 {
			nPosRow += 1.0
		}
	}

	if isWeighted {
		for i := 0; i < nRow; i++ {
			for j := 0; j < nCol; j++ {
				if tsYdata.At(i, j) == 1.0 {
					colSum.Set(0, j, colSum.At(0, j)+1)
				}
			}
		}
	}
	count := 0.0
	accuracy = 0.0
	for j := 0; j < nCol; j++ {
		if isWeighted {
			count = 0.0
		}
		for i := 0; i < nRow; i++ {
			if Yhat.At(i, j) == 1.0 && tsYdata.At(i, j) == 1.0 {
				count += 1.0
			} // else if Yhat.At(i, j) == 0.0 && tsYdata.At(i, j) == 0.0 {
			//	count += 1.0
			//}
		}
		if isWeighted {
			//fmt.Println(accuracy)
			accuracy += count / float64(nRow) * math.Log(colSum.At(0, j)/float64(nRow)) * -1.0
		}
	}
	if !isWeighted {
		accuracy = count / nPosRow
	}
	if accuracy > 1.0 {
		accuracy = 1.0
	}
	return accuracy
}

func TopKprec(tsYdata *mat64.Dense, Yhat *mat64.Dense, k int) (kPrec float64) {
	type kv struct {
		Key   int
		Value float64
	}
	nRow, nCol := tsYdata.Caps()
	colSum := mat64.NewDense(1, nCol, nil)
	colCount := mat64.NewDense(1, nCol, nil)
	//nPos := 0.0
	for j := 0; j < nCol; j++ {
		//thres for top k
		var sortYh []kv
		for i := 0; i < nRow; i++ {
			ele := Yhat.At(i, j)
			if math.IsNaN(ele) {
				sortYh = append(sortYh, kv{i, 0.0})
			} else {
				sortYh = append(sortYh, kv{i, ele})
			}
		}
		sort.Slice(sortYh, func(i, j int) bool {
			return sortYh[i].Value > sortYh[j].Value
		})
		thres := sortYh[k].Value
		//calculate kPrec
		for i := 0; i < nRow; i++ {
			ele := Yhat.At(i, j)
			//using > as golang start index at 0
			if tsYdata.At(i, j) == 1.0 {
				colSum.Set(0, j, colSum.At(0, j)+1.0)
			}
			//k > 0 dealing with equally top ranked instance
			if ele > thres && tsYdata.At(i, j) == 1.0 && k > 0 {
				colCount.Set(0, j, colCount.At(0, j)+1.0)
				k -= 1
				//nPos += 1.0
			}
		}
	}

	kPrec = 0.0
	for j := 0; j < nCol; j++ {
		kPrec += float64(nCol) * colCount.At(0, j) / (colSum.At(0, j) * float64(k))
	}
	//kPrec = nPos / float64(k) / float64(nCol)
	return kPrec
}

func Single_compute(tsYdata *mat64.Dense, tsYhat *mat64.Dense, rankCut int) (microF1 float64, accuracy float64, macroAupr float64, microAupr float64) {
	_, nLabel := tsYdata.Caps()
	sumAupr := 0.0
	sumF1 := 0.0
	sumTp := 0
	sumFp := 0
	sumFn := 0
	sumTn := 0
	for i := 0; i < nLabel; i++ {
		aupr, _, _, _ := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i), 1.0)
		sumAupr += aupr
	}
	//y-flat and microAupr
	tsYdataVec := Flat(tsYdata)
	tsYhatVec := Flat(tsYhat)
	microAupr, _, _, _ = ComputeAupr(tsYdataVec, tsYhatVec, 1.0)

	//bin with rankCut
	tsYhat, _ = BinPredByAlpha(tsYhat, rankCut, true)
	for i := 0; i < nLabel; i++ {
		f1, tp, fp, fn, tn := ComputeF1_3(tsYdata.ColView(i), tsYhat.ColView(i), 0.99)
		sumF1 += f1
		sumTp += tp
		sumFp += fp
		sumFn += fn
		sumTn += tn
	}
	p := float64(sumTp) / (float64(sumTp) + float64(sumFp))
	r := float64(sumTp) / (float64(sumTp) + float64(sumFn))
	microF1 = 2.0 * p * r / (p + r)
	accuracy = (float64(sumTp) + float64(sumTn)) / (float64(sumTp) + float64(sumFp) + float64(sumFn) + float64(sumTn))
	macroAupr = sumAupr / float64(nLabel)

	return microF1, accuracy, macroAupr, microAupr
}

func Report(tsYdata *mat64.Dense, tsYhat *mat64.Dense, thresData *mat64.Dense, rankCut int, isVerbose bool) (microF1 float64, accuracy float64, macroAupr float64, microAupr float64, agMicroF1 float64, optScore float64, macroAuprSet []float64) {
	//F1 score
	_, nLabel := tsYdata.Caps()
	sumAupr := 0.0
	sumF1 := 0.0
	sumTp := 0
	sumFp := 0
	sumFn := 0
	sumTn := 0
	tpSet := make([]int, 0)
	fpSet := make([]int, 0)
	fnSet := make([]int, 0)
	tnSet := make([]int, 0)
	macroAuprSet = make([]float64, 0)
	microF1Set := make([]float64, 0)
	//macroAupr
	for i := 0; i < nLabel; i++ {
		aupr, _, _, _ := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i), 1.0)
		//if i == 0 {
		//	firstAupr = aupr
		//}
		macroAuprSet = append(macroAuprSet, aupr)
		sumAupr += aupr
	}
	macroAupr = sumAupr / float64(nLabel)
	//top k precision
	//nPrecCut := 10
	//nRow, _ := tsYdata.Caps()
	//if nPrecCut > nRow {
	//	nPrecCut = nRow - 1
	//}
	//kPrec = TopKprec(tsYdata, tsYhat, nPrecCut)
	tsYhatAccuracy, detectNanInf := BinPredByAlpha(tsYhat, 1, true)
	accuracy = ComputeAccuracy(tsYdata, tsYhatAccuracy, false)
	//y-flat and microAupr
	tsYdataVec := Flat(tsYdata)
	tsYhatVec := Flat(tsYhat)
	microAupr, _, _, _ = ComputeAupr(tsYdataVec, tsYhatVec, 1.0)
	//agMicroF1
	agMicroF1 = MicroF1WithThres(tsYdata, tsYhat, thresData, rankCut)
	//microF1
	tsYhatMicro, _ := BinPredByAlpha(tsYhat, rankCut, true)
	for i := 0; i < nLabel; i++ {
		f1, tp, fp, fn, tn := ComputeF1_3(tsYdata.ColView(i), tsYhatMicro.ColView(i), 0.99)
		if isVerbose {
			tpSet = append(tpSet, tp)
			fpSet = append(fpSet, fp)
			fnSet = append(fnSet, fn)
			tnSet = append(tnSet, tn)
		}
		microF1Set = append(microF1Set, f1)
		sumF1 += f1
		sumTp += tp
		sumFp += fp
		sumFn += fn
		sumTn += tn
	}
	p := float64(sumTp) / (float64(sumTp) + float64(sumFp))
	r := float64(sumTp) / (float64(sumTp) + float64(sumFn))
	microF1 = 2.0 * p * r / (p + r)
	if isVerbose {
		fmt.Printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "label", "tp", "fp", "fn", "tn", "F1", "AUPR")
		for i := 0; i < nLabel; i++ {
			fmt.Printf("%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\n", i, tpSet[i], fpSet[i], fnSet[i], tnSet[i], microF1Set[i], macroAuprSet[i])
		}
	}
	//optimal score
	optScore = CostSensitiveMicroAupr(tsYdata, tsYhat)
	//optScore = math.Sqrt(agMicroF1 * microAupr)
	if detectNanInf {
		log.Print("NanInf found, mask these score to zero: ", accuracy, microF1, microAupr, macroAupr, agMicroF1)
		tmp := make([]float64, nLabel)
		return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tmp
	}
	return accuracy, microF1, microAupr, macroAupr, agMicroF1, optScore, macroAuprSet
}

func RescaleData(data *mat64.Dense, thresData *mat64.Dense) (scaleData *mat64.Dense) {
	nRow, nCol := data.Caps()
	scaleData = mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < nCol; j++ {
		for i := 0; i < nRow; i++ {
			//if data.At(i, j) >= thresData.At(0, j) {
			scaleData.Set(i, j, data.At(i, j)/thresData.At(0, j))
			//}
			//} else {
			//	scaleData.Set(i, j, 0)
			//}
		}
	}
	return scaleData
}

func RebalanceData(trYdata *mat64.Dense) (rebaData *mat64.Dense) {
	nRow, nCol := trYdata.Caps()
	rebaData = mat64.NewDense(1, nCol, nil)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if trYdata.At(i, j) == 1.0 {
				rebaData.Set(0, j, rebaData.At(0, j)+1.0)
			}
		}
	}
	for i := 0; i < nCol; i++ {
		rebaData.Set(0, i, rebaData.At(0, i)/float64(nRow))
	}
	return rebaData
}
func Zscore(data *mat64.Dense) (scaleData *mat64.Dense) {
	nRow, nCol := data.Caps()
	scaleData = mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < nCol; j++ {
		colData := make([]float64, nRow)
		max := -9999999999.9
		min := 9999999999.9
		for i := 0; i < nRow; i++ {
			colData[i] = data.At(i, j)
			if max < colData[i] {
				max = colData[i]
			}
			if min > colData[i] {
				min = colData[i]
			}
		}
		mean, sd := stat.MeanStdDev(colData, nil)
		max = stat.StdScore(max, mean, sd)
		min = stat.StdScore(min, mean, sd)
		for i := 0; i < nRow; i++ {
			value := stat.StdScore(data.At(i, j), mean, sd)
			//scaleData.Set(i, j, stat.StdScore(data.At(i, j), mean, sd)/5+0.5)
			scaleData.Set(i, j, (value-min)/(max-min))
		}
	}
	return scaleData
}

func SigmoidMatrix(data *mat64.Dense) (scaleData *mat64.Dense) {
	nRow, nCol := data.Caps()
	scaleData = mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < nCol; j++ {
		for i := 0; i < nRow; i++ {
			scaleData.Set(i, j, (Sigmoid(data.At(i, j)-0.5) / 0.5))
		}
	}
	return scaleData
}
func EleCopy(data *mat64.Dense) (data2 *mat64.Dense) {
	nRow, nCol := data.Caps()
	data2 = mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < nCol; j++ {
		for i := 0; i < nRow; i++ {
			data2.Set(i, j, data.At(i, j))
		}
	}
	return data2
}

func TprBeta(tsYvec *mat64.Vector, tsYhatVec *mat64.Vector, tprThres float64) (beta float64) {
	tprMax := 0.0
	betaMax := 0.0
	betaFinal := 0.0
	for beta := 0.1; beta < 2; beta += 0.1 {
		_, _, tpr, _ := ComputeAupr(tsYvec, tsYhatVec, beta)
		//best possible tpr
		if tpr > tprMax {
			tprMax = tpr
			betaMax = beta
		}

		if tpr > tprThres && beta > betaFinal {
			betaFinal = beta
		}
	}
	if betaFinal > 0.0 {
		return betaFinal
	} else {
		return betaMax
	}
}

func FscoreBeta(tsYdata *mat64.Dense, tsYhat *mat64.Dense, fBetaThres float64, isAutoBeta bool) (beta *mat64.Dense) {
	//init vars
	rateSet := []float64{4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2}
	//rateSet := []float64{100.0, 0.01}
	nRow, nCol := tsYdata.Caps()
	//rankCut := nCol - 1
	thresData := mat64.NewDense(1, nCol, nil)
	macroAupr := []float64{}
	beta = mat64.NewDense(1, nCol, nil)

	if !isAutoBeta {
		for i := 0; i < nCol; i++ {
			beta.Set(0, i, fBetaThres)
		}
		return beta
	}

	//init aScore with midway aRate and aupr -baseline
	aRate := 1.0
	for i := 0; i < nCol; i++ {
		nPos := 0
		for k := 0; k < nRow; k++ {
			if tsYdata.At(k, i) == 1.0 {
				nPos += 1

			}
		}
		//baseline := float64(nPos) / float64(nRow)
		_, pAupr, _, _ := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i), 1)
		//betaValue := TprBeta(tsYdata.ColView(i), tsYhat.ColView(i), 0.7)
		//aupr = aupr - baseline
		macroAupr = append(macroAupr, pAupr)
		_, _, _, optThres := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i), pAupr*aRate)
		beta.Set(0, i, pAupr*aRate)
		thresData.Set(0, i, optThres)
	}
	//Accuracy := AccuracyWithThres(tsYdata, tsYhat, thresData)
	tmpTsYhat, _ := SoftThresScale(tsYhat, thresData)
	//totLoss := CrossEntropy(tsYdata, tmpTsYhat)
	//totLoss := CostSensitiveLoss(tsYdata, tmpTsYhat, scaleThresData)
	//microAupr := MicroAuprWithThres(tsYdata, tmpTsYhat, scaleThresData)
	//microAupr := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
	//microF1 := MicroF1WithThres(tsYdata, tmpTsYhat, scaleThresData, rankCut)
	//aScore := math.Sqrt(microAupr * microF1)
	//aScore := 0 - totLoss
	aScore := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
	bScore := 0.0
	bRate := 0.0
	tmpScore := 0.0
	//init bRate and bScore with grid search for sub optimal beta
	for j := 0; j < len(rateSet); j++ {
		tmpThresData := thresData
		for i := 0; i < nCol; i++ {
			_, _, _, optThres := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i), rateSet[j]*macroAupr[i])
			tmpThresData.Set(0, i, optThres)
		}
		//tmpAccuracy := AccuracyWithThres(tsYdata, tsYhat, thresData)
		tmpTsYhat, _ := SoftThresScale(tsYhat, tmpThresData)
		//totLoss := CrossEntropy(tsYdata, tmpTsYhat)
		//totLoss := CostSensitiveLoss(tsYdata, tmpTsYhat, scaleThresData)
		//tmpMicroAupr := MicroAuprWithThres(tsYdata, tmpTsYhat, scaleThresData)
		//tmpMicroAupr := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
		//tmpMicroF1 := MicroF1WithThres(tsYdata, tmpTsYhat, scaleThresData, rankCut)
		//bScore := math.Sqrt((0 - totLoss) * tmpMicroF1)
		//bScore := math.Sqrt(tmpMicroAupr * tmpMicroF1)
		//bScore := 0 - totLoss
		bScore := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
		if bScore > tmpScore {
			for i := 0; i < nCol; i++ {
				beta.Set(0, i, rateSet[j]*macroAupr[i])
			}
			tmpScore = bScore
			bRate = rateSet[j]
			thresData = tmpThresData
		}
	}
	//while loop for optimal score
	notConverged := true
	maxItr := 100
	itr := 0
	lScore := 0.0 //larger score
	sScore := 0.0 //smaller score
	lRate := 0.0  //larger score rate
	sRate := 0.0  //smaller score rate
	if aScore > bScore {
		lRate = aRate
		sRate = bRate
		lScore = aScore
		sScore = bScore
	} else {
		lRate = bRate
		sRate = aRate
		lScore = bScore
		sScore = aScore
	}
	for notConverged || itr <= maxItr {
		//test rate, larger /smaller score rates
		tRate := (lRate + sRate) / 2.0
		//score for test rate
		itr += 1
		tmpThresData := thresData
		for i := 0; i < nCol; i++ {
			_, _, _, optThres := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i), tRate*macroAupr[i])
			tmpThresData.Set(0, i, optThres)
		}
		//tmpAccuracy := AccuracyWithThres(tsYdata, tsYhat, thresData)
		tmpTsYhat, _ := SoftThresScale(tsYhat, tmpThresData)
		//totLoss := CrossEntropy(tsYdata, tmpTsYhat)
		//totLoss := CostSensitiveLoss(tsYdata, tmpTsYhat, scaleThresData)
		//tmpMicroAupr := MicroAuprWithThres(tsYdata, tmpTsYhat, scaleThresData)
		//tmpMicroAupr := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
		//tmpMicroF1 := MicroF1WithThres(tsYdata, tmpTsYhat, scaleThresData, rankCut)
		//tScore := math.Sqrt(tmpMicroAupr * tmpMicroF1)
		//tScore := 0 - totLoss
		//tScore := math.Sqrt((0 - totLoss) * tmpMicroF1)
		tScore := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
		if tScore-sScore >= 0.001 {
			if tScore > lScore {
				sScore = lScore
				sRate = lRate
				lScore = tScore
				lRate = tRate

			} else {
				sScore = tScore
				sRate = tRate
			}
			//update beta with current lRate
			for i := 0; i < nCol; i++ {
				beta.Set(0, i, lRate*macroAupr[i])
			}
		} else {
			notConverged = false
		}
	}
	//thres and score using best beta
	for i := 0; i < nCol; i++ {
		_, _, _, optThres := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i), beta.At(0, i))
		thresData.Set(0, i, optThres)
	}
	tmpTsYhat, _ = SoftThresScale(tsYhat, thresData)
	//totLoss = CrossEntropy(tsYdata, tmpTsYhat)
	//totLoss = CostSensitiveLoss(tsYdata, tmpTsYhat, scaleThresData)
	//tmpMicroAupr := MicroAuprWithThres(tsYdata, tmpTsYhat, scaleThresData)
	//tmpMicroAupr := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
	//tmpMicroF1 := MicroF1WithThres(tsYdata, tmpTsYhat, scaleThresData, rankCut)
	//score := math.Sqrt(tmpMicroAupr * tmpMicroF1)
	//score := 0 - totLoss
	//score := math.Sqrt((0 - totLoss) * tmpMicroF1)
	score := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
	optOrder := RatioPosPerLabel(tmpTsYhat, tsYdata)
	//per label converge
	itr = 0
	maxItr = 10
	notConverged = true
	betaSet := []float64{2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.01}
	//betaSet := make([]float64, 0)
	//for i := 1.0; i <= 100.0; i += 1.0 {
	//	betaSet = append(betaSet, i)
	//}
	for notConverged && itr <= maxItr {
		itr += 1
		notConverged = false
		for i := 0; i < nCol; i++ {
			idx := optOrder[i]
			tmpThresData := thresData
			for j := 0; j < len(betaSet); j++ {
				_, _, _, optThres := ComputeAupr(tsYdata.ColView(idx), tsYhat.ColView(idx), betaSet[j])
				tmpThresData.Set(0, idx, optThres)
				//tmpScore
				tmpTsYhat, _ := SoftThresScale(tsYhat, tmpThresData)
				//totLoss := CrossEntropy(tsYdata, tmpTsYhat)
				//totLoss := CostSensitiveLoss(tsYdata, tmpTsYhat, scaleThresData)
				//tmpMicroAupr := MicroAuprWithThres(tsYdata, tmpTsYhat, scaleThresData)
				//tmpMicroAupr := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
				//tmpMicroF1 := MicroF1WithThres(tsYdata, tmpTsYhat, scaleThresData, rankCut)
				//tmpScore := math.Sqrt(tmpMicroAupr * tmpMicroF1)
				tmpScore := CostSensitiveMicroAupr(tsYdata, tmpTsYhat)
				//tmpScore := math.Sqrt((0 - totLoss) * tmpMicroF1)
				diff := tmpScore - score
				//if diff > 0 || (diff > -0.001 && betaSet[j] > beta.At(0, idx)) {
				if diff > 0 {
					score = tmpScore
					beta.Set(0, idx, betaSet[j])
					thresData = tmpThresData
					notConverged = true
				}
			}
		}
		//refresh optOrder
		tmpTsYhat, _ := SoftThresScale(tsYhat, thresData)
		optOrder = RatioPosPerLabel(tmpTsYhat, tsYdata)
	}
	return beta
}

func CostSensitiveMicroAupr(tsY *mat64.Dense, tsYhat *mat64.Dense) (microAupr float64) {
	nRow, nCol := tsY.Caps()
	tsYcost := mat64.NewDense(nRow, nCol, nil)
	colSum := make([]float64, nCol)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if tsY.At(i, j) == 1.0 {
				colSum[j] += 1.0
			}
		}
	}
	//equal pos/neg cost
	for j := 0; j < nCol; j++ {
		colSum[j] = (float64(nRow) - colSum[j]) / colSum[j]
	}
	//all 1 pos/neg cost
	//for j := 0; j < nCol; j++ {
	//	colSum[j] = 1.0
	//}

	//Cost matrix
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if tsY.At(i, j) == 1.0 {
				tsYcost.Set(i, j, colSum[j])
			} else {
				tsYcost.Set(i, j, 1.0)
			}
		}
	}
	//flats
	tsYdataVec := Flat(tsY)
	tsYhatVec := Flat(tsYhat)
	tsYcostVec := Flat(tsYcost)
	microAupr = CostSeneitiveAupr(tsYdataVec, tsYhatVec, tsYcostVec)
	return microAupr
}

func CostSensitiveLoss(tsY *mat64.Dense, tsYHat *mat64.Dense, thres *mat64.Dense) (totLoss float64) {
	nRow, nCol := tsY.Caps()
	totLoss = 0.0
	colSum := make([]float64, nCol)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if tsY.At(i, j) == 1.0 {
				colSum[j] += 1.0
			}
		}
	}
	//equal pos/neg cost
	for j := 0; j < nCol; j++ {
		colSum[j] = (float64(nRow) - colSum[j]) / colSum[j]
	}

	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			t := tsY.At(i, j)
			q := tsYHat.At(i, j)
			//false negative
			if t == 1.0 && q < thres.At(0, j) {
				totLoss += colSum[j]
			}
			//false positive
			if t == 0.0 && q > thres.At(0, j) {
				totLoss += 1.0
			}
		}
	}
	return totLoss
}
func CrossEntropy(tsY *mat64.Dense, tsYHat *mat64.Dense) (totLoss float64) {
	totLoss = 0.0
	nRow, nCol := tsY.Caps()
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			t := tsY.At(i, j)
			q := tsYHat.At(i, j)
			if q >= 0.99999999 {
				q = 0.99999999
			} else if q <= 0.00000001 {
				q = 0.00000001
			}
			loss := t*math.Log(q) + (1-t)*math.Log(1-q)
			totLoss += loss
		}
	}
	return totLoss
}

func AccuracyWithThres(tsYdata *mat64.Dense, tsYhat *mat64.Dense, thresData *mat64.Dense) (Accuracy float64) {
	_, nCol := tsYhat.Caps()
	rankCut := nCol - 1
	tsYhat = MaskZeroByThres(tsYhat, thresData)
	tsYhatAccuracy, _ := BinPredByAlpha(tsYhat, rankCut, true)
	Accuracy = ComputeAccuracy(tsYdata, tsYhatAccuracy, false)
	return Accuracy
}

func MicroAuprWithThres(tsYdata *mat64.Dense, tsYhat *mat64.Dense, thresData *mat64.Dense) (microAupr float64) {
	tsYdataVec := Flat(tsYdata)
	tsYhatVec := Flat(tsYhat)
	microAupr, _, _, _ = ComputeAupr(tsYdataVec, tsYhatVec, 1.0)
	return microAupr
}
func MicroF1WithThres(tsYdata *mat64.Dense, tsYhat *mat64.Dense, thresData *mat64.Dense, rankCut int) (microF1 float64) {
	sumTp := 0
	sumFp := 0
	sumFn := 0
	sumTn := 0
	_, nLabel := tsYhat.Caps()
	//rankCut := nLabel - 1
	tsYhatMicro, _ := BinPredByAlpha(tsYhat, rankCut, true)
	for i := 0; i < nLabel; i++ {
		_, tp, fp, fn, tn := ComputeF1_3(tsYdata.ColView(i), tsYhatMicro.ColView(i), 0.99)
		sumTp += tp
		sumFp += fp
		sumFn += fn
		sumTn += tn
	}
	p := float64(sumTp) / (float64(sumTp) + float64(sumFp))
	r := float64(sumTp) / (float64(sumTp) + float64(sumFn))
	invP := float64(sumTn) / (float64(sumTn) + float64(sumFn))
	invR := float64(sumTn) / (float64(sumTn) + float64(sumFp))
	microF1 = (1.0 + 1.0*1.0) * p * r / (p + r)
	invMicroF1 := (1.0 + 1.0*1.0) * invP * invR / (invP + invR)
	microF1 = math.Sqrt(microF1 * invMicroF1)
	return microF1
}
func FscoreThres(tsYdata *mat64.Dense, tsYhat *mat64.Dense, beta *mat64.Dense) (thres *mat64.Dense) {
	_, nCol := tsYdata.Caps()
	thres = mat64.NewDense(1, nCol, nil)
	for i := 0; i < nCol; i++ {
		_, _, _, optThres := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i), beta.At(0, i))
		thres.Set(0, i, optThres)
		//log tmp
		//ppAupr := fmt.Sprintf("%.3f", aupr)
		//ppThres := fmt.Sprintf("%.3f", optThres)
		//log.Print("label: ", i, ", ", ppAupr, ", ", ppThres)
	}
	return thres
}
func DefaultThres(tsYdata *mat64.Dense, tsYhat *mat64.Dense) (thres *mat64.Dense) {
	_, nCol := tsYdata.Caps()
	thres = mat64.NewDense(1, nCol, nil)
	//tsYhat2 := Zscore(tsYhat)
	for i := 0; i < nCol; i++ {
		thres.Set(0, i, 0.5)
	}
	return thres
}

func AccumThres(c int, colSum *mat64.Vector, thresSet *mat64.Dense, thres *mat64.Dense) {
	tC := 0
	_, nCol := thres.Caps()
	for j := 0; j < nCol; j++ {
		if colSum.At(j, 0) == 1.0 {
			thresSet.Set(c, j, thres.At(0, tC)+thresSet.At(c, j))
			tC += 1
		}
	}
}

func AveThres(cBest int, thresSet *mat64.Dense, plattCountSet *mat64.Dense) (aveThres *mat64.Dense) {
	_, nCol := thresSet.Caps()
	aveThres = mat64.NewDense(1, nCol, nil)
	for j := 0; j < nCol; j++ {
		if plattCountSet.At(cBest, j) > 0 {
		} else {
			plattCountSet.Set(cBest, j, 1.0)
		}
		aveThres.Set(0, j, thresSet.At(cBest, j)/plattCountSet.At(cBest, j))
	}
	return aveThres
}

func RefillIndCol(data *mat64.Dense, ind []int) (data2 *mat64.Dense) {
	isNeedRefill := false
	nAddCol := 0
	for i := 0; i < len(ind); i++ {
		if ind[i] <= 1 {
			isNeedRefill = true
			nAddCol += 1
		}
	}

	if isNeedRefill {
		nRow, nCol := data.Caps()
		nCol += nAddCol
		data2 = mat64.NewDense(nRow, nCol, nil)
		tickCol := 0
		for j := 0; j < nCol; j++ {
			if ind[j] > 1 {
				for i := 0; i < nRow; i++ {
					data2.Set(i, j, data.At(i, tickCol))
				}
				tickCol += 1
			} else {
				// do nothing
			}
		}
		return data2
	} else {
		return data
	}
}

func AccumTsYdata(iFold int, c int, colSum *mat64.Vector, tsYh *mat64.Dense, rawTsYh *mat64.Dense, tsY *mat64.Dense, tsX *mat64.Dense, indAccum []int, YhRawSet map[int]*mat64.Dense, YhPlattSet map[int]*mat64.Dense, YhPlattSetCalibrated map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense, iFoldMarker map[int]*mat64.Dense, yPredSet map[int]*mat64.Dense, xSet map[int]*mat64.Dense, rawThres *mat64.Dense) (map[int]*mat64.Dense, map[int]*mat64.Dense, map[int]*mat64.Dense, map[int]*mat64.Dense, map[int]*mat64.Dense, map[int]*mat64.Dense, map[int]*mat64.Dense) {
	nCol := colSum.Len()
	nRow, _ := tsYh.Caps()
	tsYh2 := mat64.NewDense(nRow, nCol, nil)
	rawTsYh2 := mat64.NewDense(nRow, nCol, nil)
	//tsX can be modified here as AccumTsYdata is after EcocRun
	////tsX = RefillIndCol(tsX, indAccum)
	//empty as it will be filled in YhPlattSetUpdate
	tsYhCalib2 := mat64.NewDense(nRow, nCol, nil)
	tsY2 := mat64.NewDense(nRow, nCol, nil)
	predY2 := mat64.NewDense(nRow, nCol, nil)
	//fold marker
	iFoldmat2 := mat64.NewDense(nRow, 1, nil)
	for i := 0; i < nRow; i++ {
		iFoldmat2.Set(i, 0, float64(iFold))
	}
	//tsYh, rawTsYh, tsY, and predY
	tC := 0
	for j := 0; j < nCol; j++ {
		if colSum.At(j, 0) == 1.0 {
			for i := 0; i < nRow; i++ {
				tsYh2.Set(i, j, tsYh.At(i, tC))
				rawTsYh2.Set(i, j, rawTsYh.At(i, tC))
				tsY2.Set(i, j, tsY.At(i, tC))
				if tsYh2.At(i, j) > rawThres.At(0, j) {
					predY2.Set(i, j, 1.0)
				}
			}
			tC += 1
		} else {
			for i := 0; i < nRow; i++ {
				tsYh2.Set(i, j, -1.0)
				rawTsYh2.Set(i, j, -1.0)
				tsY2.Set(i, j, -1.0)
				predY2.Set(i, j, -1.0)
			}
		}
	}

	//is the matrix defined previously?
	Yh, isYh := YhPlattSet[c]
	rawYh, _ := YhRawSet[c]
	YhCalib, _ := YhPlattSetCalibrated[c]
	Y, _ := yPlattSet[c]
	X, _ := xSet[c]
	iFoldmat := iFoldMarker[c]
	predY := yPredSet[c]
	if !isYh {
		YhPlattSet[c] = tsYh2
		YhRawSet[c] = rawTsYh2
		YhPlattSetCalibrated[c] = tsYhCalib2
		yPlattSet[c] = tsY2
		xSet[c] = tsX
		iFoldMarker[c] = iFoldmat2
		yPredSet[c] = predY2
	} else {
		newYh := mat64.NewDense(0, 0, nil)
		newRawYh := mat64.NewDense(0, 0, nil)
		newYhCalib := mat64.NewDense(0, 0, nil)
		newY := mat64.NewDense(0, 0, nil)
		newX := mat64.NewDense(0, 0, nil)
		newIfoldmat := mat64.NewDense(0, 0, nil)
		newPredY := mat64.NewDense(0, 0, nil)

		newYh.Stack(Yh, tsYh2)
		newRawYh.Stack(rawYh, rawTsYh2)
		newYhCalib.Stack(YhCalib, tsYhCalib2)
		newY.Stack(Y, tsY2)
		a, b := X.Caps()
		c, d := tsX.Caps()
		log.Print("stack X, dims: ", a, b, c, d)
		newX.Stack(X, tsX)
		newIfoldmat.Stack(iFoldmat, iFoldmat2)
		newPredY.Stack(predY, predY2)

		YhPlattSet[c] = newYh
		YhRawSet[c] = newRawYh
		YhPlattSetCalibrated[c] = newYhCalib
		yPlattSet[c] = newY
		xSet[c] = newX
		e, f := xSet[c].Caps()
		log.Print("stacked X, dims: ", e, f)
		iFoldMarker[c] = newIfoldmat
		yPredSet[c] = newPredY
	}
	return YhRawSet, YhPlattSet, YhPlattSetCalibrated, yPlattSet, iFoldMarker, yPredSet, xSet
}

func BestHyperParameterSetByMeasure(trainMeasure *mat64.Dense, index int, nLabel int, isPerLabel bool) (cBest []int, value []float64, microAupr []float64, microF1 []float64) {
	nRow, _ := trainMeasure.Caps()
	cBest = make([]int, 0)
	value = make([]float64, 0)
	microAupr = make([]float64, 0)
	microF1 = make([]float64, 0)
	if !isPerLabel {
		var sortMap []kv
		for i := 0; i < nRow; i++ {
			if math.IsNaN(trainMeasure.At(i, index)) {
				sortMap = append(sortMap, kv{i, 0.0})
			} else {
				sortMap = append(sortMap, kv{i, trainMeasure.At(i, index)})
			}
		}
		sort.Slice(sortMap, func(i, j int) bool {
			return sortMap[i].Value > sortMap[j].Value
		})
		cBest = append(cBest, sortMap[0].Key)
		value = append(value, sortMap[0].Value)
		microAupr = append(microAupr, trainMeasure.At(cBest[0], index-4))
		microF1 = append(microF1, trainMeasure.At(cBest[0], index-6))
		//log training info
		log.Print("choose default object score for all labels as object function in tuning.")
		ppAupr := fmt.Sprintf("%.3f", trainMeasure.At(cBest[0], index)/trainMeasure.At(cBest[0], 2))
		ppDim := fmt.Sprintf("%1.0f", trainMeasure.At(cBest[0], 0))
		ppLamda := fmt.Sprintf("%.3f", trainMeasure.At(cBest[0], 1))
		log.Print("best default object score, dimention and lamda: ", ppAupr, ", ", ppDim, " and ", ppLamda, ".")
	} else {
		//log training info
		log.Print("choose auprs for labels as object functions in tuning.")
		for j := 0; j < nLabel; j++ {
			var sortMap []kv
			//matrix 0-based
			indexPerLabel := j*2 + index + 2
			for i := 0; i < nRow; i++ {
				if math.IsNaN(trainMeasure.At(i, indexPerLabel)) {
					sortMap = append(sortMap, kv{i, 0.0})
				} else {
					sortMap = append(sortMap, kv{i, trainMeasure.At(i, indexPerLabel)})
				}
			}
			sort.Slice(sortMap, func(i, j int) bool {
				return sortMap[i].Value > sortMap[j].Value
			})
			cBest = append(cBest, sortMap[0].Key)
			value = append(value, sortMap[0].Value)
			microAupr = append(microAupr, trainMeasure.At(cBest[j], index-4))
			microF1 = append(microF1, trainMeasure.At(cBest[j], index-6))
			//log training info
			ppAupr := fmt.Sprintf("%.3f", trainMeasure.At(cBest[j], indexPerLabel)/trainMeasure.At(cBest[0], 2))
			ppDim := fmt.Sprintf("%1.0f", trainMeasure.At(cBest[j], 0))
			ppLamda := fmt.Sprintf("%.3f", trainMeasure.At(cBest[j], 1))
			log.Print("best training aupr, dimention and lamda for label ", j, ": ", ppAupr, ", ", ppDim, " and ", ppLamda, ".")
		}
	}
	return cBest, value, microAupr, microF1
}
func computeAuprSkipTr(Y *mat64.Vector, Yh *mat64.Vector, Ys *mat64.Vector) (aupr float64) {
	type kv struct {
		Key   int
		Value float64
	}
	n := Y.Len()
	mapY := make(map[int]int)
	skipY := make(map[int]int)
	var sortYh []kv
	for i := 0; i < n; i++ {
		if Y.At(i, 0) == 1.0 {
			mapY[i] = 1
		}
		if Ys.At(i, 0) == 1.0 {
			skipY[i] = 1
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
		_, ok2 := skipY[kv.Key]
		if ok2 {
			continue
		}
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

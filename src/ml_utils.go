package src

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"gonum.org/v1/gonum/stat"
	"math"
	"math/rand"
	"os"
	"sort"
)

//for cross validation
type CvFold struct {
	X *mat64.Dense
	Y *mat64.Dense
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
func (f *CvFold) SetXYinNestedTraining(idxArr []int, matX *mat64.Dense, matY *mat64.Dense) {
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

func ColStackMatrix(X *mat64.Dense, addX *mat64.Dense) (X2 *mat64.Dense) {
	X = mat64.DenseCopyOf(X.T())
	X2 = mat64.NewDense(0, 0, nil)
	//a, b := X.Caps()
	//c, d := addX.Caps()
	//fmt.Println(a, b, d, c)
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

func PosSelect(tsYdata *mat64.Dense, colSum *mat64.Vector) (tsYdataFilter *mat64.Dense) {
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
func ComputeAupr(Y *mat64.Vector, Yh *mat64.Vector) (aupr float64, maxFscore float64, optThres float64) {
	type kv struct {
		Key   int
		Value float64
	}
	n := Y.Len()
	mapY := make(map[int]int)
	//skipY := make(map[int]int)
	var sortYh []kv
	for i := 0; i < n; i++ {
		if Y.At(i, 0) == 1.0 {
			mapY[i] = 1
		}
		//if Ys.At(i, 0) == 1.0 {
		//	skipY[i] = 1
		//}
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
	maxFscore = 0.0
	optThres = 0.0
	maxThres := sortYh[0].Value
	total := float64(len(mapY))
	prData := make([]float64, 0)
	for _, kv := range sortYh {
		//fmt.Println(kv.Key, kv.Value)
		//_, ok2 := skipY[kv.Key]
		//if ok2 {
		//continue
		//}
		all += 1.0
		p += 1.0
		_, ok := mapY[kv.Key]
		if ok {
			tp += 1.0
			pr := tp / p
			re := tp / total
			//update Fscore and thres
			fscore := 2 * pr * re / (pr + re)
			//fscore := 1.25 * tp / (1.25*tp + 0.25*(total-tp) + p - tp)
			//fscore := 5.0 * tp / (p + 4.0*total)
			//fscore := 10.0 * tp / (p + 9.0*total)
			//fscore := 10.0 * tp / (p + 9.0*total)
			//fscore := 4.0 * tp / (p + 3.0*total)
			//fscore := 2.4 * tp / (p + 1.4*total)
			//fscore := 2 * tp / (p + 1*total)
			if fscore > maxFscore {
				maxFscore = fscore
				//optThres = kv.Value
				optThres = fscore / 2
				//optThres = (3*tp+fn+1)/(tp+fn+fp+1)
				//optThres = (2*tp + total+1) / (total + p - tp + 1)
				//optThres = (3*tp*(total-tp) + tp) / (total + tp + p - tp + 1)
				//optThres = (3*tp*(total-tp) + tp) / (total + tp + p - tp + float64(n))
				//optThres = (3*tp*(total-tp) + tp) / (total + tp + p - tp + 1)
				//optThres = (3*tp*(total-tp) + 4*tp) / (total + 4*tp + p - tp + float64(n))
			}
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
	if maxFscore < float64(len(mapY))/float64(n) {
		return aupr, maxFscore, maxThres
	}
	return aupr, maxFscore, optThres
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
func BinPredByAlpha(Yh *mat64.Dense, rankCut int, outBin bool) (binYh *mat64.Dense) {
	type kv struct {
		Key   int
		Value float64
	}
	nRow, nCol := Yh.Caps()
	binYh = mat64.NewDense(nRow, nCol, nil)
	for r := 0; r < nRow; r++ {
		var sortYh []kv
		for c := 0; c < nCol; c++ {
			//Yh
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
		for c := 0; c < nCol; c++ {
			ele := Yh.At(r, c)
			if math.IsNaN(ele) {
				binYh.Set(r, c, 0.0)
			} else if Yh.At(r, c) > thres {
				if outBin {
					binYh.Set(r, c, 1.0)
				} else {
					binYh.Set(r, c, (Yh.At(r, c)))
				}
			} else {
				binYh.Set(r, c, 0.0)
			}
		}
	}
	return binYh
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

func SoftThresScale(tsYhat *mat64.Dense, thresData *mat64.Dense) (tsYhat2 *mat64.Dense, thresData2 *mat64.Dense) {
	nRow, nCol := tsYhat.Caps()
	tsYhat2 = mat64.NewDense(nRow, nCol, nil)
	thresData2 = mat64.NewDense(1, nCol, nil)
	for j := 0; j < nCol; j++ {
		max := 0.0
		min := 0.0
		for i := 0; i < nRow; i++ {
			ele := (math.Exp(tsYhat.At(i, j)) - math.Exp(thresData.At(0, j))) / math.Exp(tsYhat.At(i, j))
			//ele := (Sigmoid(tsYhat.At(i, j)) - Sigmoid(thresData.At(0, j))) / Sigmoid(tsYhat.At(i, j))
			//ele := (math.Exp(tsYhat.At(i, j)) - math.Exp(thresData.At(0, j))) / math.Exp(thresData.At(0, j))
			//sum += ele
			if ele > max {
				max = ele
			}
			if ele < min {
				min = ele
			}
			tsYhat2.Set(i, j, ele)
		}
		//ave := sum / float64(nRow)
		for i := 0; i < nRow; i++ {
			raw := tsYhat2.At(i, j)
			ele := (raw - min) / (max - min)
			tsYhat2.Set(i, j, ele)
		}
		thresData2.Set(0, j, (0.0-min)/(max-min))
	}
	return tsYhat2, thresData2
	//return tsYhat2

}

func Sigmoid(x float64) (y float64) {
	y = (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x))
	//y = x / (1 + math.Abs(x))
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
		if y > 0 && yh >= thres {
			tp += 1
		} else if y <= 0 && yh >= thres {
			fp += 1
		} else if y > 0 && yh < thres {
			fn += 1
		} else if y <= 0 && yh < thres {
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
func ComputeAccuracy(tsYdata *mat64.Dense, Yhat *mat64.Dense) (accuracy float64) {
	nRow, nCol := tsYdata.Caps()
	count := 0.0
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if Yhat.At(i, j) == 1.0 && tsYdata.At(i, j) == 1.0 {
				count += 1.0
			}
		}
	}
	accuracy = count / float64(nRow)
	return accuracy
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
		aupr, _, _ := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
		sumAupr += aupr
	}
	//y-flat and microAupr
	tsYdataVec := Flat(tsYdata)
	tsYhatVec := Flat(tsYhat)
	microAupr, _, _ = ComputeAupr(tsYdataVec, tsYhatVec)

	//bin with rankCut
	tsYhat = BinPredByAlpha(tsYhat, rankCut, true)
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

//func Report(tsYdata *mat64.Dense, tsYhat *mat64.Dense, rebaData *mat64.Dense, rankCut int, isVerbose bool) (microF1 float64, accuracy float64, macroAupr float64, microAupr float64) {
func Report(tsYdata *mat64.Dense, tsYhat *mat64.Dense, thresData *mat64.Dense, rankCut int, isVerbose bool) (microF1 float64, accuracy float64, macroAupr float64, microAupr float64) {
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
	macroAuprSet := make([]float64, 0)
	microF1Set := make([]float64, 0)
	//tsYhat = Zscore(tsYhat)
	//tsYhat = ColScale(tsYhat, rebaData)
	//tsYhat = RescaleData(tsYhat, thresData)
	//macroAupr
	for i := 0; i < nLabel; i++ {
		aupr, _, _ := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
		macroAuprSet = append(macroAuprSet, aupr)
		sumAupr += aupr
	}
	macroAupr = sumAupr / float64(nLabel)
	//tmp, one thing at a time
	tsYhat, thresData = SoftThresScale(tsYhat, thresData)
	//y-flat and microAupr
	tsYdataVec := Flat(tsYdata)
	tsYhatVec := Flat(tsYhat)
	microAupr, _, _ = ComputeAupr(tsYdataVec, tsYhatVec)
	//accuracy
	tsYhatAccuracy := BinPredByAlpha(tsYhat, 1, true)
	accuracy = ComputeAccuracy(tsYdata, tsYhatAccuracy)
	//microF1
	//tsYhat = MaskZeroByThres(tsYhat, thresData)
	tsYhatMicro := BinPredByAlpha(tsYhat, rankCut, false)
	for i := 0; i < nLabel; i++ {
		//f1, tp, fp, fn, tn := ComputeF1_3(tsYdata.ColView(i), tsYhatMicro.ColView(i), thresData.At(0, i))
		f1, tp, fp, fn, tn := ComputeF1_3(tsYdata.ColView(i), tsYhatMicro.ColView(i), thresData.At(0, i))
		fmt.Println(thresData.At(0, i))
		//f1, tp, fp, fn, tn := ComputeF1_3(tsYdata.ColView(i), tsYhat.ColView(i), 0.5)
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
	return accuracy, microF1, microAupr, macroAupr
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
		for i := 0; i < nRow; i++ {
			colData[i] = data.At(i, j)
		}
		mean, sd := stat.MeanStdDev(colData, nil)
		for i := 0; i < nRow; i++ {
			scaleData.Set(i, j, stat.StdScore(data.At(i, j), mean, sd)/5+0.5)
		}
	}
	return scaleData
}

func FscoreThres(tsYdata *mat64.Dense, tsYhat *mat64.Dense) (thres *mat64.Dense) {
	_, nCol := tsYdata.Caps()
	thres = mat64.NewDense(1, nCol, nil)
	//tsYhat2 := Zscore(tsYhat)
	for i := 0; i < nCol; i++ {
		_, _, optThres := ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
		thres.Set(0, i, optThres)
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

func AccumTsYdata(c int, colSum *mat64.Vector, tsYh *mat64.Dense, tsY *mat64.Dense, YhPlattSet map[int]*mat64.Dense, yPlattSet map[int]*mat64.Dense) {
	nCol := colSum.Len()
	nRow, _ := tsYh.Caps()
	tsYh2 := mat64.NewDense(nRow, nCol, nil)
	tsY2 := mat64.NewDense(nRow, nCol, nil)

	tC := 0
	for j := 0; j < nCol; j++ {
		if colSum.At(j, 0) == 1.0 {
			for i := 0; i < nRow; i++ {
				tsYh2.Set(i, j, tsYh.At(i, tC))
				tsY2.Set(i, j, tsY.At(i, tC))
			}
			tC += 1
		} else {
			for i := 0; i < nRow; i++ {
				tsYh2.Set(i, j, -1.0)
				tsY2.Set(i, j, -1.0)
			}
		}
	}
	//whether the matrix is defined previously
	Yh, isYh := YhPlattSet[c]
	Y, _ := yPlattSet[c]
	if !isYh {
		YhPlattSet[c] = tsYh2
		yPlattSet[c] = tsY2
	} else {
		newYh := mat64.NewDense(0, 0, nil)
		newY := mat64.NewDense(0, 0, nil)
		newYh.Stack(Yh, tsYh2)
		newY.Stack(Y, tsY2)
		YhPlattSet[c] = newYh
		yPlattSet[c] = newY
	}
}

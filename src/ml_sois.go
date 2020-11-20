package src

import (
	"github.com/gonum/matrix/mat64"
	"log"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

type combineList struct {
	Key      string
	Sum      float64
	Combines [][]int
	Count    int
	Rows     []int
}

type combine struct {
	Key     string
	Combine []int
}

func SOIS(trY *mat64.Dense, nFold int, ratio int, isOutInfo bool) (folds map[int][]int) {
	nRow, nLabel := trY.Caps()
	rowUsed := make(map[int]bool)
	allCombine := make(map[string]float64)
	perRowCombine := make(map[int][][]int)
	sampleWithCombineMap := make(map[string]combineList)
	allNegRow := make([]int, 0)
	lRatio, _ := labelRatio(trY) //label scores for ranking most minority label (pairs) first
	//init folds
	folds = make(map[int][]int)
	for i := 0; i < nFold; i++ {
		tmp := make([]int, 0)
		folds[i] = tmp
	}
	//pseudo rand Ints for avoiding a timesteamp rand related anomaly in some runs
	//instances were non-randomly distributed way away from random
	rand.Seed(1)
	randInts := make([]int, 0)
	for i := 0; i < nRow; i++ {
		n := rand.Int()
		randInts = append(randInts, n)
	}

	for rowIdx := 0; rowIdx < nRow; rowIdx++ {
		key, sum, combines := genCombines(trY.RawRowView(rowIdx), lRatio)
		rowUsed[rowIdx] = false
		if sum == 0.0 {
			allNegRow = append(allNegRow, rowIdx)
			continue
		}
		perRowCombine[rowIdx] = combines
		//all combines
		for i := 0; i < len(combines); i++ {
			ele := strconv.Itoa(combines[i][0])
			if len(combines[i]) == 2 {
				ele = ele + "," + strconv.Itoa(combines[i][1])
			}
			_, exist := allCombine[ele]
			if exist {
				allCombine[ele] += 1.0
			} else {
				allCombine[ele] = 1.0
			}
		}
		//sample with combine
		_, exist := sampleWithCombineMap[key]
		if !exist {
			var tmp combineList
			tmp.Combines = combines
			tmp.Count = 1
			tmp.Sum = sum
			tmp.Key = key
			tmp.Rows = append(tmp.Rows, rowIdx)
			sampleWithCombineMap[key] = tmp
		} else {
			tmp := sampleWithCombineMap[key]
			tmp.Count += 1
			tmp.Rows = append(tmp.Rows, rowIdx)
			sampleWithCombineMap[key] = tmp
		}
	}
	//perCombinePerFold map[indexForFold]map[combineString..such..as"1"..and.."1,3"]counts
	//perFold           map[indexForFold]totalInstanceCount
	perCombinePerFold := make(map[int]map[string]float64)
	perLabelPerFold := make(map[int]map[string]float64)
	perFold := make(map[int]float64)
	for i := 0; i < nFold; i++ {
		for com, count := range allCombine {
			tmp := make(map[string]float64)
			tmp[com] = count / float64(nFold)
			perCombinePerFold[i] = tmp
		}
		perFold[i] = float64(nRow) / float64(nFold)
		for j := 0; j < len(lRatio); j++ {
			tmp := make(map[string]float64)
			ele := strconv.Itoa(j)
			tmp[ele] = 0.0
			perLabelPerFold[i] = tmp
		}
	}

	//folds
	key := mostDemandCombine(sampleWithCombineMap)
	for key != "" {
		//log.Print("Choose key: ", key, " with Sum ", sampleWithCombineMap[key].Sum, " and Count ", sampleWithCombineMap[key].Count)
		for j := 0; j < len(sampleWithCombineMap[key].Combines); j++ {
			//ele is one combine in the loop for key
			ele := strconv.Itoa(sampleWithCombineMap[key].Combines[j][0])
			if len(sampleWithCombineMap[key].Combines[j]) == 2 {
				ele = ele + "," + strconv.Itoa(sampleWithCombineMap[key].Combines[j][1])
			}
			//rows in this ele/combine, find the most demand fold for each row
			for i := 0; i < len(sampleWithCombineMap[key].Rows); i++ {
				rowIdx := sampleWithCombineMap[key].Rows[i]
				if rowUsed[rowIdx] {
					continue
				}
				//append rowIdx to folds and mark as used
				iFold := mostDemandFold(perCombinePerFold, perLabelPerFold, lRatio, perFold, ele, rowIdx, randInts[rowIdx], nFold)
				folds[iFold] = append(folds[iFold], rowIdx)
				//}
				rowUsed[rowIdx] = true
				//remove this rowIdx in all sampleWithCombineMap
				//sampleWithCombineMap count -1 as well
				for k, cl := range sampleWithCombineMap {
					for l := 0; l < len(cl.Rows); l++ {
						if cl.Rows[l] == rowIdx {
							cl.Rows = remove(cl.Rows, l)
							cl.Count -= 1
							sampleWithCombineMap[k] = cl
							break
						}
					}
				}
				//change counts in perCombinePerFold for all combine ele touched by rowIdx
				//record all label elements in this row
				allLabelInRow := make(map[string]string)
				for k := 0; k < len(perRowCombine[rowIdx]); k++ {
					ele2 := strconv.Itoa(perRowCombine[rowIdx][k][0])
					allLabelInRow[ele2] = ""
					if len(perRowCombine[rowIdx][k]) == 2 {
						ele3 := strconv.Itoa(perRowCombine[rowIdx][k][1])
						ele2 = ele2 + "," + ele3
						allLabelInRow[ele3] = ""
					}
					perCombinePerFold[iFold][ele2] -= 1.0
				}
				//change counts in perLabelPerFold for all label/ele touched by rowIdx
				for label, _ := range allLabelInRow {
					perLabelPerFold[iFold][label] += 1.0
				}
				//change counts in perFold
				perFold[iFold] -= 1.0
			}
		}
		key = mostDemandCombine(sampleWithCombineMap)
	}
	//balance negative /positive ratio
	nNeg := len(allNegRow)
	nPos := nRow - nNeg
	if nNeg/nPos > ratio {
		if isOutInfo {
			log.Print("negative instance ", nNeg, " is too many for ", nPos, " postives.")
			log.Print("down sampling negatives to ", ratio, " * ", nPos)
		}
		nNeg = nPos * ratio
	}

	nNegPerFold := make([]int, nFold)
	for j := 0; j < nNeg; j++ {
		iFold := j % nFold
		nNegPerFold[iFold] += 1
		folds[iFold] = append(folds[iFold], allNegRow[j])
		perFold[iFold] -= 1.0
	}
	if isOutInfo {
		log.Print("SOIS folds generated.\n")
		log.SetFlags(0)
		log.Print("\tnegatives per fold.")
		str := ""
		for j := 0; j < nFold; j++ {
			str = str + "\t" + strconv.Itoa(nNegPerFold[j])
		}
		log.Print(str)
		log.Print("\n\tpositive per fold(row) per label(column).")
		for i := 0; i < nFold; i++ {
			nPos := make([]int, nLabel)
			for j := 0; j < nLabel; j++ {
				for k := 0; k < len(folds[i]); k++ {
					if trY.At(folds[i][k], j) == 1 {
						nPos[j] += 1
					}
				}
			}
			str = ""
			for j := 0; j < nLabel; j++ {
				str = str + "\t" + strconv.Itoa(nPos[j])
			}
			log.Print(str)
		}
	}
	log.SetFlags(log.LstdFlags)
	return folds
}
func remove(slice []int, i int) []int {
	copy(slice[i:], slice[i+1:])
	return slice[:len(slice)-1]
}

//perCombinePerFold map[indexForFold]map[combineString..such..as"1"..and.."1,3"]counts
//perLabelPerFold map[indexForFold]map[labelString..such..as"0"]counts
//perFold           map[indexForFold]totalInstanceCount
//combine           combineString..such..as"1"..and.."1,3"
//rowIdx            rowIndex
//nFold             number of folds to distribute instances
//iFold             choosen fold index
func mostDemandFold(perCombinePerFold map[int]map[string]float64, perLabelPerFold map[int]map[string]float64, labelRatio map[int]float64, perFold map[int]float64, combine string, rowIdx int, randInt int, nFold int) (iFold int) {
	//is the label pair should be filled to minority labels?
	minorLabelFold, isMinorLabel := mostUnderRepFold(combine, perLabelPerFold, labelRatio, nFold, 0.8)
	if isMinorLabel {
		return minorLabelFold
	}
	var sortMap []kv
	for i := 0; i < nFold; i++ {
		sortMap = append(sortMap, kv{i, perCombinePerFold[i][combine]})
	}
	sort.Slice(sortMap, func(i, j int) bool {
		return sortMap[i].Value > sortMap[j].Value
	})
	//is more than one fold with the same demand value?
	demandValue := sortMap[0].Value
	var sortMap2 []kv
	for i := 0; i < nFold; i++ {
		if demandValue == perCombinePerFold[i][combine] {
			sortMap2 = append(sortMap2, kv{i, perFold[i]})
		}
	}
	//if so, roll the dice
	if len(sortMap2) == 1 {
		return sortMap2[0].Key
	} else {
		sort.Slice(sortMap2, func(i, j int) bool {
			return sortMap2[i].Value > sortMap2[j].Value
		})
		demandValue2 := sortMap2[0].Value
		foldsIdx := make([]int, 0)
		for j := 0; j < nFold; j++ {
			if demandValue2 == perFold[j] {
				foldsIdx = append(foldsIdx, j)
			}
		}
		if len(foldsIdx) == 1 {
			return foldsIdx[0]
		} else {
			n := randInt % len(foldsIdx)
			return foldsIdx[n]
		}

	}

}

func mostDemandCombine(sampleWithCombineMap map[string]combineList) (key string) {
	//map to slice
	sampleWithCombine := make([]combineList, 0)
	for _, cl := range sampleWithCombineMap {
		sampleWithCombine = append(sampleWithCombine, cl)
	}
	//sort sampleWithCombine by Sum  and then by Count
	//most sum and least counts
	sort.Slice(sampleWithCombine, func(i, j int) bool {
		if sampleWithCombine[i].Sum != sampleWithCombine[j].Sum {
			return sampleWithCombine[i].Sum > sampleWithCombine[j].Sum
		}
		return sampleWithCombine[i].Count < sampleWithCombine[j].Count
	})
	for i := 0; i < len(sampleWithCombine); i++ {
		if sampleWithCombine[i].Count == 0 {
			continue
		}
		return sampleWithCombine[i].Key
	}
	//return empty string if all zeroz
	return ""
}

//sets ordered so that the largest label scores are ranked top
//so that it is firstly considered in subsetting for this key type
func genCombines(rowVec []float64, labelRatio map[int]float64) (key string, sum float64, roSets [][]int) {
	sets := make([][]int, 0)
	roSets = make([][]int, 0) //reodered sets
	setScore := make([]float64, 0)
	key = ""
	sum = 0
	for i := 0; i < len(rowVec); i++ {
		key = key + strconv.Itoa(int(rowVec[i]))
		if rowVec[i] == 1.0 {
			sum += labelRatio[i]
			ele := []int{i}
			sets = append(sets, ele)
			setScore = append(setScore, labelRatio[i])
			for j := i + 1; j < len(rowVec); j++ {
				if rowVec[j] == 1.0 {
					ele := []int{i, j}
					sets = append(sets, ele)
					setScore = append(setScore, labelRatio[i]+labelRatio[j])
				}
			}
		}
	}
	//sort the setScores
	var sortMap []kv
	for i := 0; i < len(setScore); i++ {
		sortMap = append(sortMap, kv{i, setScore[i]})
	}
	sort.Slice(sortMap, func(i, j int) bool {
		return sortMap[i].Value > sortMap[j].Value
	})
	for i := 0; i < len(setScore); i++ {
		roSets = append(roSets, sets[sortMap[i].Key])
	}

	return key, sum, roSets
}

func labelRatio(trY *mat64.Dense) (labelScore map[int]float64, combineScore map[string]float64) {
	labelScore = make(map[int]float64, 0)
	combineScore = make(map[string]float64, 0)
	nRow, nCol := trY.Caps()
	colSum := mat64.NewDense(1, nCol, nil)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if trY.At(i, j) == 1.0 {
				colSum.Set(0, j, colSum.At(0, j)+1)
			}
		}
	}
	//max in colSums
	max := 0.0
	for j := 0; j < nCol; j++ {
		if colSum.At(0, j) > max {
			max = colSum.At(0, j)
		}
	}
	//label Score
	for j := 0; j < nCol; j++ {
		labelScore[j] = max / colSum.At(0, j)
		ele := strconv.Itoa(j)
		combineScore[ele] = labelScore[j]
	}
	//combine score
	for i := 0; i < nCol; i++ {
		for j := i + 1; j < nCol; j++ {
			ele := strconv.Itoa(i)
			ele = ele + "," + strconv.Itoa(j)
			combineScore[ele] = labelScore[i] + labelScore[j]
		}
	}
	return labelScore, combineScore
}

func mostUnderRepFold(combine string, perLabelPerFold map[int]map[string]float64, labelRatio map[int]float64, nFold int, thresRatio float64) (minorLabelFold int, isMinorLabel bool) {
	//init
	minorLabelFold = -1
	isMinorLabel = false
	ele := ""
	//single label
	if !strings.Contains(combine, ",") {
		ele = combine
		//label pair, choose larger ratio label
	} else {
		idx := strings.Split(combine, ",")
		a, _ := strconv.Atoi(idx[0])
		b, _ := strconv.Atoi(idx[1])
		if labelRatio[a] > labelRatio[b] {
			ele = idx[0]
		} else {
			ele = idx[1]
		}
	}
	//find empty subset
	for i := 0; i < nFold; i++ {
		//mark empty subset directly and break
		if perLabelPerFold[i][ele] == 0.0 {
			minorLabelFold = i
			isMinorLabel = true
			break
		}
	}
	//if no empty subset
	max := 0.0
	if !isMinorLabel {
		for i := 0; i < nFold; i++ {
			if perLabelPerFold[i][ele] < max {
				max = perLabelPerFold[i][ele]
			}
		}
		minRatio := 1.0
		minIdx := -1
		for i := 0; i < nFold; i++ {
			ratio := perLabelPerFold[i][ele] / max
			if ratio < minRatio {
				minRatio = ratio
				minIdx = i
			}
		}
		if minRatio <= thresRatio {
			minorLabelFold = minIdx
			isMinorLabel = true
		}
	}

	return minorLabelFold, isMinorLabel
}

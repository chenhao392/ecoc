package src

import (
	"github.com/gonum/matrix/mat64"
	"log"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"sync"
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

func single_SOIS(subFolds map[int]map[int][]int, idxFold int, trYdata *mat64.Dense, nFold int, ratio int, minPosPerFold int, isOutInfo bool, wg *sync.WaitGroup, mutex *sync.Mutex) {
	defer wg.Done()
	tmpFold := SOIS(trYdata, nFold, ratio, minPosPerFold, isOutInfo)
	mutex.Lock()
	subFolds[idxFold] = tmpFold
	mutex.Unlock()
}

func SOIS(trY *mat64.Dense, nFold int, ratio int, minPosPerFold int, isOutInfo bool) (folds map[int][]int) {
	nRow, nLabel := trY.Caps()
	rowUsed := make(map[int]bool)
	allCombine := make(map[string]float64)
	perRowCombine := make(map[int][][]int)
	sampleWithCombineMap := make(map[string]combineList)
	allNegRow := make([]int, 0)
	lRatio, colSum, _ := labelRatio(trY) //label scores for ranking most minority label (pairs) first
	folds, rowUsed = overSampleMinorLabel(trY, minPosPerFold, nFold, colSum)
	//pseudo rand Ints for avoiding a timesteamp rand related anomaly in some runs
	//instances were non-randomly distributed way away from random
	rand.Seed(1)
	randInts := make([]int, 0)
	for i := 0; i < nRow; i++ {
		n := rand.Int()
		randInts = append(randInts, n)
	}
	//init stats
	for rowIdx := 0; rowIdx < nRow; rowIdx++ {
		key, sum, combines := genCombines(trY.RawRowView(rowIdx), lRatio)
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

	//update stats to match the over sampled rows
	sampleWithCombineMap, perCombinePerFold, perLabelPerFold, perFold = updateStatAfterOverSampling(nFold, rowUsed, sampleWithCombineMap, perRowCombine, perCombinePerFold, perLabelPerFold, perFold)
	//subseting to folds
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

func labelRatio(trY *mat64.Dense) (labelScore map[int]float64, colSum *mat64.Dense, combineScore map[string]float64) {
	labelScore = make(map[int]float64, 0)
	combineScore = make(map[string]float64, 0)
	nRow, nCol := trY.Caps()
	colSum = mat64.NewDense(1, nCol, nil)
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
	return labelScore, colSum, combineScore
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

func overSampleMinorLabel(trY *mat64.Dense, minPosPerFold int, nFold int, colSum *mat64.Dense) (folds map[int][]int, rowUsed map[int]bool) {
	nRow, nLabel := trY.Caps()
	minPos := float64(minPosPerFold * nFold)
	perLabelRequired := make(map[int]float64)
	rowUsed = make(map[int]bool)
	//init folds
	folds = make(map[int][]int)
	for i := 0; i < nFold; i++ {
		tmp := make([]int, 0)
		folds[i] = tmp
	}
	for i := 0; i < nRow; i++ {
		rowUsed[i] = false
	}
	//minimum pos instance per fold
	for i := 0; i < nLabel; i++ {
		if colSum.At(0, i) < minPos {
			perLabelRequired[i] = 1.0 + (minPos-colSum.At(0, i))/float64(nFold)
		} else {
			perLabelRequired[i] = 0.0
		}
	}
	//sort the demanding label
	var sortMap []kv
	for i := 0; i < nLabel; i++ {
		sortMap = append(sortMap, kv{i, perLabelRequired[i]})
	}
	sort.Slice(sortMap, func(i, j int) bool {
		return sortMap[i].Value > sortMap[j].Value
	})
	//start with the most demanding label
	for i := 0; i < nLabel; i++ {
		idx := sortMap[i].Key
		//break when label not demanding over sampling
		if perLabelRequired[idx] <= 0.0 {
			break
		} else {
			for p := 0; p < nRow; p++ {
				//row with the demanding minor label
				if trY.At(p, idx) == 1.0 && !rowUsed[p] {
					rowUsed[p] = true
					//row pushed to all folds
					for j := 0; j < nFold; j++ {
						folds[j] = append(folds[j], p)
					}
					//updating perLabelRequired counts for all minor label
					for q := 0; q < nLabel; q++ {
						if trY.At(p, q) == 1.0 {
							perLabelRequired[q] -= 1.0
						}
					}
				}
			}
		}
	}
	//
	return folds, rowUsed
}

func updateStatAfterOverSampling(nFold int, rowUsed map[int]bool, sampleWithCombineMap map[string]combineList, perRowCombine map[int][][]int, perCombinePerFold map[int]map[string]float64, perLabelPerFold map[int]map[string]float64, perFold map[int]float64) (sampleWithCombineMap2 map[string]combineList, perCombinePerFold2 map[int]map[string]float64, perLabelPerFold2 map[int]map[string]float64, perFold2 map[int]float64) {
	//remove this rowIdx in all sampleWithCombineMap
	//sampleWithCombineMap count -1 as well
	for rowIdx, isUsed := range rowUsed {
		if isUsed {
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
				for i := 0; i < nFold; i++ {
					perCombinePerFold[i][ele2] -= 1.0
				}
			}
			//change counts in perLabelPerFold for all label/ele touched by rowIdx
			for label, _ := range allLabelInRow {
				for i := 0; i < nFold; i++ {
					//each label in mostUnderRepFold should be counted the same, thus 1.0 is good
					perLabelPerFold[i][label] += 1.0
				}
			}
			//change counts in perFold
			for i := 0; i < nFold; i++ {
				//used in mostDemandFold when it's a tie in demand Value
				//to find the fold with least instances
				//averaging to nFold here won't hurt and keep number positive
				perFold[i] -= 1.0 / float64(nFold)
			}
		}
	}
	return sampleWithCombineMap, perCombinePerFold, perLabelPerFold, perFold
}

func MLSMOTE(trXdata *mat64.Dense, trYdata *mat64.Dense, nKnn int, mlsRatio float64, randValues []float64) (newTrXdata *mat64.Dense, newTrYdata *mat64.Dense) {
	nRow, nColX := trXdata.Caps()
	_, nColY := trYdata.Caps()
	perLabelRequired := make(map[int]float64)
	rowUsed := make(map[int]bool)
	colSum := make(map[int]float64, nColY)
	trXsyn := make(map[int][]float64)
	trYsyn := make(map[int][]float64)
	synIdx := 0
	meanLabel := 0.0
	for i := 0; i < nRow; i++ {
		for j := 0; j < nColY; j++ {
			if trYdata.At(i, j) == 1.0 {
				meanLabel += 1.0
				colSum[j] += 1.0
			}
		}
	}
	//minimum pos instance per label
	meanLabel = mlsRatio * meanLabel / float64(nColY)
	for i := 0; i < nColY; i++ {
		if colSum[i] < meanLabel {
			perLabelRequired[i] = 1.0 + (meanLabel - colSum[i])
		} else {
			perLabelRequired[i] = 0.0
		}
	}
	//sort the demanding label
	var sortMap []kv
	for i := 0; i < nColY; i++ {
		sortMap = append(sortMap, kv{i, perLabelRequired[i]})
	}
	sort.Slice(sortMap, func(i, j int) bool {
		return sortMap[i].Value > sortMap[j].Value
	})
	//start with the most demanding label
	for i := 0; i < nColY; i++ {
		idx := sortMap[i].Key
		//break when label not demanding over sampling
		if perLabelRequired[idx] <= 0.0 {
			break
		} else {
			for p := 0; p < nRow; p++ {
				//break if the idx label reached the required number
				if perLabelRequired[idx] <= 0 {
					break
				}
				//row with the demanding minor label
				if trYdata.At(p, idx) == 1.0 && !rowUsed[p] {
					rowUsed[p] = true
					//row synthetic from k nearest neighbors in feature set
					//note 1st instance in DistanceTopK is itself
					//system rand from range distuv.Uniform{Min: -0.00000001, Max: 0.00000001}
					//rand values array length is nTr
					nnIdx := DistanceTopK(nKnn+1, p, trXdata, trXdata)
					for q := 1; q < len(nnIdx); q++ {
						tmp := make([]float64, 0)
						trXsyn[synIdx] = tmp
						for m := 0; m < nColX; m++ {
							xEle := trXdata.At(nnIdx[q], m) + 50000000.0*(randValues[p]+0.00000001)*(trXdata.At(nnIdx[0], m)-trXdata.At(nnIdx[q], m))
							trXsyn[synIdx] = append(trXsyn[synIdx], xEle)
						}
						tmp2 := make([]float64, 0)
						trYsyn[synIdx] = tmp2
						for m := 0; m < nColY; m++ {
							tmpL := 0.0
							if trYdata.At(nnIdx[0], m) == 1.0 {
								tmpL = 1.0
							}
							if trYdata.At(nnIdx[q], m) == 1.0 {
								tmpL = 1.0
							}
							trYsyn[synIdx] = append(trYsyn[synIdx], tmpL)
						}
						//updating perLabelRequired counts for all minor label
						for q := 0; q < nColY; q++ {
							if trYdata.At(p, q) == 1.0 {
								perLabelRequired[q] -= 1.0
							}
						}
						synIdx += 1
						if perLabelRequired[idx] <= 0 {
							break
						}
					}
				}
			}
		}
	}
	//check if mlsmote or not
	if synIdx > 0 {
		synTrXdata := mat64.NewDense(synIdx, nColX, nil)
		synTrYdata := mat64.NewDense(synIdx, nColY, nil)
		nSynPos := make([]int, nColY)
		for i := 0; i < synIdx; i++ {
			for j := 0; j < nColX; j++ {
				synTrXdata.Set(i, j, trXsyn[i][j])
			}
			for j := 0; j < nColY; j++ {
				synTrYdata.Set(i, j, trYsyn[i][j])
				if trYsyn[i][j] == 1.0 {
					nSynPos[j] += 1
				}
			}
		}
		//log number of syn positive instances per label
		log.Print("\tsynthetic pos label with knn thres ", nKnn, ".")
		str := ""
		for j := 0; j < nColY; j++ {
			str = str + "\t" + strconv.Itoa(nSynPos[j])
		}
		log.Print(str)
		newTrXdata = mat64.NewDense(0, 0, nil)
		newTrYdata = mat64.NewDense(0, 0, nil)
		newTrXdata.Stack(trXdata, synTrXdata)
		newTrYdata.Stack(trYdata, synTrYdata)
	} else {
		log.Print("\tsynthetic pos label not generated, no label below mlsRatio(", mlsRatio, ") * meanLabel(", meanLabel, ").")
		newTrXdata = trXdata
		newTrYdata = trYdata
	}
	return newTrXdata, newTrYdata
}

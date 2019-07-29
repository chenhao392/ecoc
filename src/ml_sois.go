package src

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"sort"
	"strconv"
)

type combineList struct {
	Key      string
	Sum      int
	Combines [][]int
	Count    int
	Rows     []int
}

type combine struct {
	Key     string
	Combine []int
}

func SOIS(trY *mat64.Dense, nFold int) (folds map[int][]int) {
	nRow, _ := trY.Caps()
	rowUsed := make(map[int]bool)
	allCombine := make(map[string]float64)
	perRowCombine := make(map[int][][]int)
	sampleWithCombineMap := make(map[string]combineList)
	allNegRow := make([]int, 0)
	//init folds
	folds = make(map[int][]int)
	for i := 0; i < nFold; i++ {
		tmp := make([]int, 0)
		folds[i] = tmp
	}
	for rowIdx := 0; rowIdx < nRow; rowIdx++ {
		key, sum, combines := genCombines(trY.RawRowView(rowIdx))
		rowUsed[rowIdx] = false
		if sum == 0 {
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
	//per combination per fold and per fold
	perCombinePerFold := make(map[int]map[string]float64)
	perFold := make(map[int]float64)
	for i := 0; i < nFold; i++ {
		for com, count := range allCombine {
			//_, exist := perCombinePerFold[i]
			//if !exist {
			tmp := make(map[string]float64)
			tmp[com] = count / float64(nFold)
			perCombinePerFold[i] = tmp
			//} else {
			//	_, exist2 := perCombinePerFold[i][com]
			//	if !exist2 {
			//		tmp := make(map[string]float64)
			//		tmp[com] = count / float64(nFold)
			//		perCombinePerFold[i][com] += count / float64(nFold)
			//	}
			//}
		}
		perFold[i] = float64(nRow) / float64(nFold)
	}

	//folds
	key := mostDemandCombine(sampleWithCombineMap)
	for key != "" {
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
				iFold := mostDemandFold(perCombinePerFold, perFold, ele, rowIdx, nFold)
				//_, exist := folds[iFold]
				//if !exist {
				//	tmp := make([]int, 0)
				//	tmp = append(tmp, rowIdx)
				//	folds[iFold] = tmp
				//} else {
				//fmt.Println(iFold)
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
				for k := 0; k < len(perRowCombine[rowIdx]); k++ {
					ele2 := strconv.Itoa(perRowCombine[rowIdx][k][0])
					if len(perRowCombine[rowIdx][k]) == 2 {
						ele2 = ele2 + "," + strconv.Itoa(perRowCombine[rowIdx][k][1])
					}
					perCombinePerFold[iFold][ele] -= 1.0
				}
				//change counts in perFold
				perFold[iFold] -= 1.0
			}
		}
		key = mostDemandCombine(sampleWithCombineMap)
	}
	//all negative Rows
	nNeg := make([]int, nFold)
	for j := 0; j < len(allNegRow); j++ {
		iFold := j % nFold
		nNeg[iFold] += 1
		//fmt.Println(iFold)
		folds[iFold] = append(folds[iFold], allNegRow[j])
		perFold[iFold] -= 1.0
	}
	for j := 0; j < nFold; j++ {
		fmt.Printf("\t%d", nNeg[j])
	}
	fmt.Printf("\n")

	return folds
}
func remove(slice []int, i int) []int {
	copy(slice[i:], slice[i+1:])
	return slice[:len(slice)-1]
}

func mostDemandFold(perCombinePerFold map[int]map[string]float64, perFold map[int]float64, combine string, rowIdx int, nFold int) (iFold int) {
	var sortMap []kv
	for i := 0; i < nFold; i++ {
		sortMap = append(sortMap, kv{i, perCombinePerFold[i][combine]})
	}
	sort.Slice(sortMap, func(i, j int) bool {
		return sortMap[i].Value > sortMap[j].Value
	})
	//more than one fold with the same fold??
	demandValue := sortMap[0].Value
	var sortMap2 []kv
	for i := 0; i < nFold; i++ {
		if demandValue == perCombinePerFold[i][combine] {
			sortMap2 = append(sortMap2, kv{i, perFold[i]})
		}
	}
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
			n := rand.Int() % len(foldsIdx)
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

func genCombines(rowVec []float64) (key string, sum int, sets [][]int) {
	sets = make([][]int, 0)
	key = ""
	sum = 0
	for i := 0; i < len(rowVec); i++ {
		key = key + strconv.Itoa(i)
		if rowVec[i] == 1.0 {
			sum += 1
			ele := []int{i}
			sets = append(sets, ele)
			for j := i + 1; j < len(rowVec); j++ {
				if rowVec[j] == 1.0 {
					ele := []int{i, j}
					sets = append(sets, ele)
				}
			}
		}
	}
	return key, sum, sets
}

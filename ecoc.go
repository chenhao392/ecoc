package main

//#include <stdlib.h>
//import "C"
import (
	//"bufio"
	"flag"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	//"strconv"
	"strings"
	"sync"
	//"unsafe"
)

var wg sync.WaitGroup
var mutex sync.Mutex

type kv struct {
	Key   int
	Value float64
}

//type matSet struct {
//	Key int
//	Mat *mat64.Dense
//}

func main() {
	//input argv
	var tsX *string = flag.String("tsX", "", "test FeatureSet")
	var tsY *string = flag.String("tsY", "data/human.bp.level1.set1.tsMatrix.txt", "test LabelSet")
	//var tsY *string = flag.String("tsY", "data/yeast.level1.set1.tsMatrix.txt", "test LabelSet")
	var trX *string = flag.String("trX", "", "train FeatureSet")
	var trY *string = flag.String("trY", "data/human.bp.level1.set1.trMatrix.txt", "train LabelSet")
	//var trY *string = flag.String("trY", "data/yeast.level1.set1.trMatrix.txt", "train LabelSet")

	//var tsX *string = flag.String("tsX", "data/tsX.emo.txt", "test FeatureSet")
	//var tsY *string = flag.String("tsY", "data/tsY.emo.txt", "test LabelSet")
	//var trX *string = flag.String("trX", "data/trX.emo.txt", "train FeatureSet")
	//var trY *string = flag.String("trY", "data/trY.emo.txt", "train LabelSet")
	//var inNetworkFiles *string = flag.String("n", "data/hs_coe_net.txt,data/hs_db_net.txt,data/hs_exp_net.txt,data/hs_fus_net.txt,data/hs_nej_net.txt,data/hs_pp_net.txt", "network file")
	//var inNetworkFiles *string = flag.String("n", "data/sc_coe_net.txt,data/sc_db_net.txt,data/sc_exp_net.txt,data/sc_fus_net.txt,data/sc_nej_net.txt,data/sc_pp_net.txt", "network file")
	//var inNetworkFiles *string = flag.String("n", "data/sc_db_net.txt,data/sc_exp_net.txt,data/sc_fus_net.txt,data/sc_nej_net.txt,data/sc_pp_net.txt", "network file")
	//var inNetworkFiles *string = flag.String("n", "data/sc_db_net.txt,data/sc_exp_net.txt", "network file")
	var inNetworkFiles *string = flag.String("n", "data/hs_exp_net.txt", "network file")
	//var inNetworkFiles *string = flag.String("n", "data/hs_fus_net.txt,data/hs_pp_net.txt", "network file")
	//var inNetworkFiles *string = flag.String("n", "", "network file")
	var priorMatrixFiles *string = flag.String("p", "data/human.bp.level1.set1.trMatrix.txt", "prior/known gene file")
	//var priorMatrixFiles *string = flag.String("p", "data/yeast.level1.set1.trMatrix.txt", "prior/known gene file")
	//var priorMatrixFiles *string = flag.String("p", "", "prior/known gene file")
	var resFolder *string = flag.String("res", "resultEmo", "resultFolder")
	var inThreads *int = flag.Int("t", 48, "number of threads")
	var rankCut *int = flag.Int("c", 3, "rank cut (alpha) for F1 calculation")
	var reg *bool = flag.Bool("r", false, "regularize CCA, default false")
	flag.Parse()
	if *priorMatrixFiles != "" && *inNetworkFiles != "" {
		fmt.Println("prior file and network file argv defined. Generating ECOC codeword on the fly.")
	} else if *tsX != "" && *trX != "" {
		fmt.Println("Defined tsX and trX argv found. Using these ECOC codeword.")
	} else {
		flag.PrintDefaults()
		os.Exit(1)
	}
	//kSet := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	kSet := []int{1}
	//0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 and rev
	sigmaFctsSet := []float64{0.0001, 0.0025, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1, 1.23, 1.56, 2.04, 2.78, 4.0, 6.25, 11.11, 25.0, 100.0, 400.0, 10000.0}
	nFold := 5
	rand.Seed(1)
	runtime.GOMAXPROCS(*inThreads)
	//read data
	tsYdata, tsRowName, _, _ := readFile(*tsY, true, true)
	trYdata, trRowName, _, _ := readFile(*trY, true, true)
	tsXdata := mat64.NewDense(0, 0, nil)
	trXdata := mat64.NewDense(0, 0, nil)
	if *priorMatrixFiles == "" && *inNetworkFiles == "" {
		tsXdata, _, _, _ = readFile(*tsX, true, true)
		trXdata, _, _, _ = readFile(*trX, true, true)
	} else if *tsX == "" && *trX == "" {
		inNetworkFile := strings.Split(*inNetworkFiles, ",")
		priorMatrixFile := strings.Split(*priorMatrixFiles, ",")
		// for filtering prior genes, only those in training set are used for propagation
		trGeneMap := make(map[string]int)
		for i := 0; i < len(trRowName); i++ {
			trGeneMap[trRowName[i]] = i
		}
		for i := 0; i < len(inNetworkFile); i++ {
			for j := 0; j < len(priorMatrixFile); j++ {
				//idIdx as gene -> idx in net
				network, idIdx, _ := readNetwork(inNetworkFile[i])
				//network, idIdx, idxToId := readNetwork(*inNetworkFile)
				//idArr  gene index as in prior file
				priorData, idArr, _, _ := readFile(priorMatrixFile[j], true, true)
				sPriorData := propagateSet(network, priorData, idIdx, idArr, trGeneMap)
				_, nLabel := sPriorData.Caps()
				tmpTsXdata := mat64.NewDense(len(tsRowName), nLabel, nil)
				tmpTrXdata := mat64.NewDense(len(trRowName), nLabel, nil)
				//tsX
				for k := 0; k < len(tsRowName); k++ {
					for l := 0; l < nLabel; l++ {
						_, exist := idIdx[tsRowName[k]]
						if exist {
							tmpTsXdata.Set(k, l, sPriorData.At(idIdx[tsRowName[k]], l))
						}
					}
				}
				nRow, _ := tsXdata.Caps()
				if nRow == 0 {
					tsXdata = tmpTsXdata
				} else {
					tsXdata = colStackMatrix(tsXdata, tmpTsXdata)
				}
				//trX
				for k := 0; k < len(trRowName); k++ {
					for l := 0; l < nLabel; l++ {
						_, exist := idIdx[trRowName[k]]
						if exist {
							tmpTrXdata.Set(k, l, sPriorData.At(idIdx[trRowName[k]], l))
						}
					}
				}
				nRow, _ = trXdata.Caps()
				if nRow == 0 {
					trXdata = tmpTrXdata
				} else {
					trXdata = colStackMatrix(trXdata, tmpTrXdata)
				}
			}
		}
	}

	_, nFea := trXdata.Caps()
	//nTs, _ := tsXdata.Caps()
	nTr, nLabel := trYdata.Caps()
	//nRowTsY, _ := tsYdata.Caps()
	if nFea < nLabel {
		fmt.Println("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
		os.Exit(0)
	}
	//split training data for nested cv
	idxPerm := rand.Perm(nTr)
	trainFold := make([]cvFold, nFold)
	testFold := make([]cvFold, nFold)

	for i := 0; i < nFold; i++ {
		cvTrain := make([]int, 0)
		cvTest := make([]int, 0)
		cvTestMap := map[int]int{}
		for j := i * nTr / nFold; j < (i+1)*nTr/nFold-1; j++ {
			cvTest = append(cvTest, idxPerm[j])
			cvTestMap[idxPerm[j]] = idxPerm[j]
		}
		//the rest is for training
		for j := 0; j < nTr; j++ {
			_, exist := cvTestMap[j]
			if !exist {
				cvTrain = append(cvTrain, j)
			}
		}
		//generating ECOC
		if *tsX == "" && *trX == "" {
			inNetworkFile := strings.Split(*inNetworkFiles, ",")
			priorMatrixFile := strings.Split(*priorMatrixFiles, ",")
			//trXdataCV should use genes in trYdata for training only
			trGeneMapCV := make(map[string]int)
			for j := 0; j < len(cvTrain); j++ {
				trGeneMapCV[trRowName[cvTrain[j]]] = cvTrain[j]
			}
			trXdataCV := mat64.NewDense(0, 0, nil)
			//codes
			for i := 0; i < len(inNetworkFile); i++ {
				for j := 0; j < len(priorMatrixFile); j++ {
					//idIdx as gene -> idx in net
					network, idIdx, _ := readNetwork(inNetworkFile[i])
					//network, idIdx, idxToId := readNetwork(*inNetworkFile)
					//idArr  gene index as in prior file
					priorData, idArr, _, _ := readFile(priorMatrixFile[j], true, true)
					sPriorData := propagateSet(network, priorData, idIdx, idArr, trGeneMapCV)
					_, nLabel := sPriorData.Caps()
					tmpTrXdata := mat64.NewDense(len(trRowName), nLabel, nil)
					//trX
					for k := 0; k < len(trRowName); k++ {
						for l := 0; l < nLabel; l++ {
							_, exist := idIdx[trRowName[k]]
							if exist {
								tmpTrXdata.Set(k, l, sPriorData.At(idIdx[trRowName[k]], l))
							}
						}
					}
					nRow, _ := trXdataCV.Caps()
					if nRow == 0 {
						trXdataCV = tmpTrXdata
					} else {
						trXdataCV = colStackMatrix(trXdataCV, tmpTrXdata)
					}
				}
			}

			trainFold[i].setXYinNestedTraining(cvTrain, trXdataCV, trYdata)
			testFold[i].setXYinNestedTraining(cvTest, trXdataCV, trYdata)
		} else {
			trainFold[i].setXYinNestedTraining(cvTrain, trXdata, trYdata)
			testFold[i].setXYinNestedTraining(cvTest, trXdata, trYdata)
		}
	}

	//min dims
	//potential bug when cv set's minDims is smaller
	minDims := int(math.Min(float64(nFea), float64(nLabel)))
	//nK
	nK := 0
	for k := 0; k < len(kSet); k++ {
		if kSet[k] < minDims {
			nK += 1
		}
	}
	//measures
	nL := nK * len(sigmaFctsSet)
	trainF1 := mat64.NewDense(nL, 4, nil)
	trainAccuracy := mat64.NewDense(nL, 4, nil)
	trainMicroAupr := mat64.NewDense(nL, 4, nil)
	trainMacroAupr := mat64.NewDense(nL, 4, nil)

	testF1 := mat64.NewDense(1, 4, nil)
	testAccuracy := mat64.NewDense(1, 4, nil)
	testMicroAupr := mat64.NewDense(1, 4, nil)
	testMacroAupr := mat64.NewDense(1, 4, nil)

	//out dir
	err := os.MkdirAll("./"+*resFolder, 0755)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("pass ecoc")
	for i := 0; i < nFold; i++ {
		YhSet := make(map[int]*mat64.Dense)
		ecocRun(testFold[i].X, testFold[i].Y, trainFold[i].X, trainFold[i].Y, rankCut, reg, kSet, sigmaFctsSet, nFold, nK, YhSet)
		//update all meassures
		c := 0
		for m := 0; m < nK; m++ {
			for n := 0; n < len(sigmaFctsSet); n++ {
				//fmt.Println(i, j, c)
				//fmt.Println(YhSet[c].At(0, 0))
				microF1, accuracy, macroAupr, microAupr := single_compute(testFold[i].Y, YhSet[c], trainFold[i].Y, *rankCut)
				trainF1.Set(c, 0, float64(kSet[m]))
				trainF1.Set(c, 1, sigmaFctsSet[n])
				trainF1.Set(c, 2, trainF1.At(c, 2)+1.0)
				trainF1.Set(c, 3, trainF1.At(c, 3)+microF1)
				trainAccuracy.Set(c, 0, float64(kSet[m]))
				trainAccuracy.Set(c, 1, sigmaFctsSet[n])
				trainAccuracy.Set(c, 2, trainAccuracy.At(c, 2)+1.0)
				trainAccuracy.Set(c, 3, trainAccuracy.At(c, 3)+accuracy)
				trainMicroAupr.Set(c, 0, float64(kSet[m]))
				trainMicroAupr.Set(c, 1, sigmaFctsSet[n])
				trainMicroAupr.Set(c, 2, trainMicroAupr.At(c, 2)+1.0)
				trainMicroAupr.Set(c, 3, trainMicroAupr.At(c, 3)+microAupr)
				trainMacroAupr.Set(c, 0, float64(kSet[m]))
				trainMacroAupr.Set(c, 1, sigmaFctsSet[n])
				trainMacroAupr.Set(c, 2, trainMacroAupr.At(c, 2)+1.0)
				trainMacroAupr.Set(c, 3, trainMacroAupr.At(c, 3)+macroAupr)
				c += 1
			}
		}
	}
	//micro Aupr
	//n, _ := microAupr.Caps()
	//for i := 0; i < n; i++ {
	//	microAupr.Set(i, 4, microAupr.At(i, 0)/(microAupr.At(i, 0)+microAupr.At(i, 1)+microAupr.At(i, 2)+microAupr.At(i, 3)))
	//}

	//sort by meanAupr
	var sortMap []kv
	n, _ := trainMicroAupr.Caps()
	for i := 0; i < n; i++ {
		sortMap = append(sortMap, kv{i, trainMicroAupr.At(i, 3)})
		//sortMap = append(sortMap, kv{i, macroF1.At(i, 3) / macroF1.At(i, 2)})
	}
	sort.Slice(sortMap, func(i, j int) bool {
		return sortMap[i].Value > sortMap[j].Value
	})
	//best training aupr
	cBest := sortMap[0].Key
	kSet = []int{int(trainMicroAupr.At(cBest, 0))}
	sigmaFctsSet = []float64{trainMicroAupr.At(cBest, 1)}
	YhSet := make(map[int]*mat64.Dense)
	ecocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, sigmaFctsSet, nFold, 1, YhSet)
	//corresponding testing measures
	c := 0
	//for i := 0; i < nK; i++ {
	i := 0
	for j := 0; j < len(sigmaFctsSet); j++ {
		microF1, accuracy, macroAupr, microAupr := single_compute(tsYdata, YhSet[c], trYdata, *rankCut)
		testF1.Set(c, 0, float64(kSet[i]))
		testF1.Set(c, 1, sigmaFctsSet[j])
		testF1.Set(c, 2, testF1.At(c, 2)+1.0)
		testF1.Set(c, 3, testF1.At(c, 3)+microF1)
		testAccuracy.Set(c, 0, float64(kSet[i]))
		testAccuracy.Set(c, 1, sigmaFctsSet[j])
		testAccuracy.Set(c, 2, testAccuracy.At(c, 2)+1.0)
		testAccuracy.Set(c, 3, testAccuracy.At(c, 3)+accuracy)
		testMicroAupr.Set(c, 0, float64(kSet[i]))
		testMicroAupr.Set(c, 1, sigmaFctsSet[j])
		testMicroAupr.Set(c, 2, testMicroAupr.At(c, 2)+1.0)
		testMicroAupr.Set(c, 3, testMicroAupr.At(c, 3)+microAupr)
		testMacroAupr.Set(c, 0, float64(kSet[i]))
		testMacroAupr.Set(c, 1, sigmaFctsSet[j])
		testMacroAupr.Set(c, 2, testMacroAupr.At(c, 2)+1.0)
		testMacroAupr.Set(c, 3, testMacroAupr.At(c, 3)+macroAupr)
		c += 1
	}
	//}

	//result file. note: per gene result is written in single_IOC_MFADecoding_and_result func, controlled by isGridSearch bool

	oFile := "./" + *resFolder + "/cvTraining.microF1.txt"
	writeFile(oFile, trainF1)
	oFile = "./" + *resFolder + "/cvTraining.accuracy.txt"
	writeFile(oFile, trainAccuracy)
	oFile = "./" + *resFolder + "/cvTraining.macroAupr.txt"
	writeFile(oFile, trainMacroAupr)
	oFile = "./" + *resFolder + "/cvTraining.microAupr.txt"
	writeFile(oFile, trainMicroAupr)
	oFile = "./" + *resFolder + "/cvTesting.microF1.txt"
	writeFile(oFile, testF1)
	oFile = "./" + *resFolder + "/cvTesting.accuracy.txt"
	writeFile(oFile, testAccuracy)
	oFile = "./" + *resFolder + "/cvTesting.macroAupr.txt"
	writeFile(oFile, testMacroAupr)
	oFile = "./" + *resFolder + "/cvTesting.microAupr.txt"
	writeFile(oFile, testMicroAupr)
	oFile = "./" + *resFolder + "/test.probMatrix.txt"
	writeFile(oFile, YhSet[0])
	os.Exit(0)
}

//func single_IOC_MFADecoding_and_result(outPerLabel bool, isGridSearch bool, nTs int, k int, c int, tsY_Prob *mat64.Dense, tsY_C *mat64.Dense, sigma *mat64.Dense, Bsub *mat64.Dense, sigmaFcts float64, nLabel int, sumResF1 *mat64.Dense, macroF1 *mat64.Dense, sumResAupr *mat64.Dense, sumResContingency *mat64.Dense, microAupr *mat64.Dense, meanAupr *mat64.Dense, tsYdata *mat64.Dense, rankCut int, minDims int, resFolder string) {
func single_IOC_MFADecoding_and_result(nTs int, k int, c int, tsY_Prob *mat64.Dense, tsY_C *mat64.Dense, sigma *mat64.Dense, Bsub *mat64.Dense, sigmaFcts float64, nLabel int, tsYdata *mat64.Dense, rankCut int, minDims int, YhSet map[int]*mat64.Dense) {
	defer wg.Done()
	if k >= minDims {
		return
	}
	tsYhat := mat64.NewDense(nTs, nLabel, nil)
	for i := 0; i < nTs; i++ {
		//the doc seems to be old, (0,x] seems to be correct
		//dim checked to be correct
		tsY_Prob_slice := tsY_Prob.Slice(i, i+1, 0, nLabel)
		tsY_C_slice := tsY_C.Slice(i, i+1, 0, k)
		arr := IOC_MFADecoding(nTs, mat64.DenseCopyOf(tsY_Prob_slice), mat64.DenseCopyOf(tsY_C_slice), sigma, Bsub, k, sigmaFcts, nLabel)
		tsYhat.SetRow(i, arr)
	}
	mutex.Lock()
	//fmt.Println("set:", c, tsYhat.At(0, 0))
	YhSet[c] = tsYhat
	//return tsYhat
	mutex.Unlock()
	//if !isGridSearch {
	//	sFctStr := strconv.FormatFloat(sigmaFcts, 'f', 3, 64)
	//	kStr := strconv.FormatInt(int64(k), 16)
	//	oFile := "./" + resFolder + "/k" + kStr + "sFct" + sFctStr + ".txt"
	//	writeFile(oFile, tsYhat)
	//}
	//score
	//mutex.Lock()
	//sumF1 := 0.0
	//sumAupr := 0.0
	//for i := 0; i < nLabel; i++ {
	//	f1, tp, fp, fn, tn := computeF1_3(tsYdata.ColView(i), tsYhat.ColView(i), rankCut)
	//	aupr := computeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
	//	if outPerLabel {
	//		sumResF1.Set(c, i, f1)
	//		sumResAupr.Set(c, i, aupr)
	//	}
	//	sumResContingency.Set(0, 0, sumResContingency.At(0, 0)+float64(tp))
	//	sumResContingency.Set(0, 1, sumResContingency.At(0, 1)+float64(fp))
	//	sumResContingency.Set(0, 2, sumResContingency.At(0, 2)+float64(fn))
	//	sumResContingency.Set(0, 3, sumResContingency.At(0, 3)+float64(tn))
	//	sumF1 += f1
	//	sumAupr += aupr
	//}
	//if isGridSearch {
	//	macroF1.Set(c, 0, float64(k))
	//	macroF1.Set(c, 1, sigmaFcts)
	//	macroF1.Set(c, 2, macroF1.At(c, 2)+1.0)
	//	macroF1.Set(c, 3, sumF1/float64(nLabel)+macroF1.At(c, 3))
	//
	//	microAupr.Set(c, 0, sumResContingency.At(0, 0)+microAupr.At(c, 0))
	//	microAupr.Set(c, 1, sumResContingency.At(0, 1)+microAupr.At(c, 1))
	//	microAupr.Set(c, 2, sumResContingency.At(0, 2)+microAupr.At(c, 2))
	//	microAupr.Set(c, 3, sumResContingency.At(0, 3)+microAupr.At(c, 3))
	//
	//	meanAupr.Set(c, 0, float64(k))
	//	meanAupr.Set(c, 1, sigmaFcts)
	//	meanAupr.Set(c, 2, meanAupr.At(c, 2)+1.0)
	//	meanAupr.Set(c, 3, sumAupr/float64(nLabel)+meanAupr.At(c, 3))
	//}
	//mutex.Unlock()
}
func single_adaptiveTrainRLS_Regress_CG(i int, trXdataB *mat64.Dense, nFold int, nFea int, nTr int, tsXdataB *mat64.Dense, sigma *mat64.Dense, trY_Cdata *mat64.Dense, nTs int, tsY_C *mat64.Dense, randValues []float64, idxPerm []int) {
	defer wg.Done()
	beta, _, optMSE := adaptiveTrainRLS_Regress_CG(trXdataB, trY_Cdata.ColView(i), nFold, nFea, nTr, randValues, idxPerm)
	mutex.Lock()
	sigma.Set(0, i, math.Sqrt(optMSE))
	fmt.Println("at", i)
	//bias term for tsXdata added before
	element := mat64.NewDense(0, 0, nil)
	element.Mul(tsXdataB, beta)
	for j := 0; j < nTs; j++ {
		tsY_C.Set(j, i, element.At(j, 0))
	}
	//fmt.Println(i, lamda, sigma)
	mutex.Unlock()
}

//func ecocRun(tsXdata *mat64.Dense, tsYdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, rankCut *int, reg *bool, kSet []int, sigmaFctsSet []float64, sumResF1 *mat64.Dense, sumResAupr *mat64.Dense, sumResContingency *mat64.Dense, microAupr *mat64.Dense, macroF1 *mat64.Dense, meanAupr *mat64.Dense, nFold int, nK int, resFolder string, outPerLabel bool, isGridSearch bool) (err error) {
func ecocRun(tsXdata *mat64.Dense, tsYdata *mat64.Dense, trXdata *mat64.Dense, trYdata *mat64.Dense, rankCut *int, reg *bool, kSet []int, sigmaFctsSet []float64, nFold int, nK int, YhSet map[int]*mat64.Dense) (err error) {
	colSum, trYdata := posFilter(trYdata)
	tsYdata = posSelect(tsYdata, colSum)
	//vars
	nTr, nFea := trXdata.Caps()
	nTs, _ := tsXdata.Caps()
	_, nLabel := trYdata.Caps()
	nRowTsY, _ := tsYdata.Caps()
	//min dims
	minDims := int(math.Min(float64(nFea), float64(nLabel)))
	if nFea < nLabel {
		fmt.Println("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
		return nil
	}
	//tsY_prob
	tsY_Prob := mat64.NewDense(nRowTsY, nLabel, nil)
	//adding bias term for tsXData
	ones := make([]float64, nTs)
	for i := range ones {
		ones[i] = 1
	}
	tsXdataB := colStack(tsXdata, ones)
	//adding bias term for trXdata
	ones = make([]float64, nTr)
	for i := range ones {
		ones[i] = 1
	}
	trXdataB := colStack(trXdata, ones)
	//step 1
	for i := 0; i < nLabel; i++ {
		wMat, _, _ := adaptiveTrainLGR_Liblin(trXdata, trYdata.ColView(i), nFold, nFea)
		element := mat64.NewDense(0, 0, nil)
		element.Mul(tsXdataB, wMat)
		for j := 0; j < nTs; j++ {
			//the -1*element.At() is changed to 1*element.At() in this implementation
			//the sign was flipped in this golang library, as manually checked
			value := 1.0 / (1 + math.Exp(1*element.At(j, 0)))
			tsY_Prob.Set(j, i, value)
		}
	}
	fmt.Println("pass step 1 coding\n")

	//cca
	B := mat64.NewDense(0, 0, nil)
	if !*reg {
		var cca stat.CC
		err := cca.CanonicalCorrelations(trXdataB, trYdata, nil)
		if err != nil {
			log.Fatal(err)
		}
		B = cca.Right(nil, false)
	} else {
		//B is not the same with matlab code
		//_, B = ccaProjectTwoMatrix(trXdataB, trYdata)
		B = ccaProject(trXdataB, trYdata)
	}
	fmt.Println("pass step 2 cca coding\n")

	//CCA code
	trY_Cdata := mat64.NewDense(0, 0, nil)
	trY_Cdata.Mul(trYdata, B)
	//decoding with regression
	tsY_C := mat64.NewDense(nRowTsY, nLabel, nil)
	sigma := mat64.NewDense(1, nLabel, nil)
	//for workers
	randValues := RandListFromUniDist(nTr)
	idxPerm := rand.Perm(nTr)
	_, nCol := trY_Cdata.Caps()
	fmt.Println("nCol:", nCol, "nLabel:", nLabel)
	wg.Add(nLabel)
	for i := 0; i < nLabel; i++ {
		go single_adaptiveTrainRLS_Regress_CG(i, trXdataB, nFold, nFea, nTr, tsXdataB, sigma, trY_Cdata, nTs, tsY_C, randValues, idxPerm)
	}
	wg.Wait()
	fmt.Println("pass step 3 cg decoding\n")
	//decoding and step 4
	c := 0
	//if isGridSearch {
	wg.Add(nK * len(sigmaFctsSet))
	for k := 0; k < nK; k++ {
		Bsub := mat64.DenseCopyOf(B.Slice(0, nLabel, 0, kSet[k]))
		for s := 0; s < len(sigmaFctsSet); s++ {
			//fmt.Println(k, s, c)
			go single_IOC_MFADecoding_and_result(nTs, kSet[k], c, tsY_Prob, tsY_C, sigma, Bsub, sigmaFctsSet[s], nLabel, tsYdata, *rankCut, minDims, YhSet)
			c += 1
		}
	}
	//} else {
	//wg.Add(1)
	//Bsub := mat64.DenseCopyOf(B.Slice(0, nLabel, 0, kSet[0]))
	//go single_IOC_MFADecoding_and_result(outPerLabel, isGridSearch, nTs, kSet[0], c, tsY_Prob, tsY_C, sigma, Bsub, sigmaFctsSet[0], nLabel, sumResF1, macroF1, sumResAupr, sumResContingency, microAupr, meanAupr, tsYdata, *rankCut, minDims, resFolder)

	//}
	wg.Wait()
	//fmt.Println(len(YhSet))
	return nil
}

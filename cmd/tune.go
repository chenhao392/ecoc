// Copyright © 2019 Hao Chen <chenhao.mymail@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package cmd

//#include <stdlib.h>
//import "C"
import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	//"sort"
	"strings"
	//"sync"
	//"unsafe"
	"github.com/chenhao392/ecoc/src"
	"github.com/spf13/cobra"
	//"time"
)

// tuneCmd represents the tune command
var tuneCmd = &cobra.Command{
	Use:   "tune",
	Short: "hyperparameter tuning and benchmarking",
	Long: `Hyperparameter tuning and benchmarking for the following parameters.
 1) number of CCA dimensions for explaining the label dependency.
 2) the trade-off between the gaussion and binomial model in decoding.

The label data in tab delimited matrix form is required as train/testing matrices.
The tune command automatically use the feature data (ECOC matrix) if provided with
-tsX and -trX flags. Otherwise, it computes the corresponding matrices using label 
propagation on a network or networks.

 1) In label data, each row is one instance/gene and each column is one 
    label, such as GO term or pathway ID. the first column should be unique 
	instance/gene IDs. 
 2) In feature data, each row is also one instance/gene, using the same exact 
    order as defined in label data matrices. And each column is one feature in 0-1 
	scale. The column order can be randomly shuffled.

If at least one network file is provided and no feature data found, the program will 
compute the matrix.

1) The network file is a tab delimited file with three columns. The first two
    columns define gene-gene interactions using the instance/gene names IDs used 
	in training and test data. The third column is the confidence score.
 2) Multiple additional priors can be added into the label propagation process if provided.

Sample usages:
  with feature data:
  ecoc tune -trY training_label -tsY test_label -trX training_feature -tsX testing_feature 
  with network data:
  ecoc tune -trY training_label -tsY test_label -n network_file -nFold 5 -t 48
  with network data and addtional prior:
  ecoc tune -trY training_label -tsY test_label -n net_file1,net_file2 -p prior_file1,prior_file2`,

	Run: func(cmd *cobra.Command, args []string) {
		tsY, _ := cmd.Flags().GetString("tsY")
		trY, _ := cmd.Flags().GetString("trY")
		inNetworkFiles, _ := cmd.Flags().GetString("n")
		priorMatrixFiles, _ := cmd.Flags().GetString("p")
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		rankCut, _ := cmd.Flags().GetInt("c")
		nKnn, _ := cmd.Flags().GetInt("k")
		reg, _ := cmd.Flags().GetBool("r")
		nFold, _ := cmd.Flags().GetInt("nFold")
		isDada, _ := cmd.Flags().GetBool("ec")
		alpha, _ := cmd.Flags().GetFloat64("alpha")
		isAddPrior, _ := cmd.Flags().GetBool("addPrior")

		fBetaThres := 1.0
		//out dir and logging
		err := os.MkdirAll("./"+resFolder, 0755)
		if err != nil {
			fmt.Println(err)
			return
		}
		logFile, err := os.OpenFile("./"+resFolder+"/log.txt", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			log.Fatal(err)
		}
		defer logFile.Close()
		log.SetOutput(logFile)
		log.Print("Program started.")
		//
		rand.Seed(1)
		runtime.GOMAXPROCS(threads)
		debug.SetGCPercent(50)
		//read data
		tsYdata, tsRowName, _, _ := src.ReadFile(tsY, true, true)
		trYdata, trRowName, _, _ := src.ReadFile(trY, true, true)
		posLabelRls, negLabelRls := src.LabelRelationship(trYdata)
		tsXdata := mat64.NewDense(0, 0, nil)
		trXdata := mat64.NewDense(0, 0, nil)
		// for filtering prior genes, only those in training set are used for propagation
		trGeneMap := make(map[string]int)
		for i := 0; i < len(trRowName); i++ {
			trGeneMap[trRowName[i]] = i
		}
		//network
		inNetworkFile := strings.Split(inNetworkFiles, ",")
		priorMatrixFile := strings.Split(priorMatrixFiles, ",")
		network := mat64.NewDense(0, 0, nil)
		idIdx := make(map[string]int)
		idxToId := make(map[int]string)
		indAccum := make([]int, 0)
		for i := 0; i < len(inNetworkFile); i++ {
			//idIdx as gene -> idx in net
			log.Print("loading network file: ", inNetworkFile[i])
			network, idIdx, idxToId = src.ReadNetwork(inNetworkFile[i])
			//network, idIdx, idxToId := readNetwork(*inNetworkFile)
			if !isAddPrior {
				sPriorData, ind := src.PropagateSet(network, trYdata, idIdx, trRowName, trGeneMap, isDada, alpha, &wg, &mutex)
				tsXdata, trXdata = src.FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
				indAccum = append(indAccum, ind...)
			} else {
				for j := 0; j < len(priorMatrixFile); j++ {
					priorData, priorGeneID, priorIdxToId := src.ReadNetwork(priorMatrixFile[j])
					sPriorData, ind := src.PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdata, idIdx, idxToId, trRowName, trGeneMap, isDada, alpha, &wg, &mutex)
					tsXdata, trXdata = src.FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
					indAccum = append(indAccum, ind...)
				}
			}
		}

		_, nFea := trXdata.Caps()
		nTr, nLabel := trYdata.Caps()
		//fmt.Println(nFea)
		if nFea < nLabel {
			log.Print("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
			os.Exit(0)
		}
		kSet, sigmaFctsSet, _ := src.HyperParameterSet(nLabel, 0.1, 0.5, 4)
		_, sigmaFctsSet2, _ := src.HyperParameterSet(nLabel, 0.025, 0.1, 3)
		sigmaFctsSet = append(sigmaFctsSet2, sigmaFctsSet...)
		//split training data for nested cv
		folds := src.SOIS(trYdata, nFold, true)
		//os.Exit(0)
		//idxPerm := rand.Perm(nTr)
		trainFold := make([]src.CvFold, nFold)
		testFold := make([]src.CvFold, nFold)

		for f := 0; f < nFold; f++ {
			cvTrain := make([]int, 0)
			cvTest := make([]int, 0)
			cvTestMap := map[int]int{}
			for j := 0; j < len(folds[f]); j++ {
				cvTest = append(cvTest, folds[f][j])
				cvTestMap[folds[f][j]] = folds[f][j]
			}
			//the rest is for training
			for j := 0; j < nTr; j++ {
				_, exist := cvTestMap[j]
				if !exist {
					cvTrain = append(cvTrain, j)
				}
			}
			//generating ECOC
			//trXdataCV should use genes in trYdata for training only
			trGeneMapCV := make(map[string]int)
			for j := 0; j < len(cvTrain); j++ {
				trGeneMapCV[trRowName[cvTrain[j]]] = cvTrain[j]
			}
			trXdataCV := mat64.NewDense(0, 0, nil)
			_, nColY := trYdata.Caps()
			trYdataCV := mat64.NewDense(len(cvTrain), nColY, nil)
			trRowNameCV := make([]string, 0)
			for s := 0; s < len(cvTrain); s++ {
				trYdataCV.SetRow(s, trYdata.RawRowView(cvTrain[s]))
				trRowNameCV = append(trRowNameCV, trRowName[cvTrain[s]])
			}
			//codes
			indAccum := make([]int, 0)
			for i := 0; i < len(inNetworkFile); i++ {
				//idIdx as gene -> idx in net
				network, idIdx, idxToId = src.ReadNetwork(inNetworkFile[i])
				if !isAddPrior {
					sPriorData, ind := src.PropagateSet(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, isDada, alpha, &wg, &mutex)
					indAccum = append(indAccum, ind...)
					_, nTrLabel := trYdataCV.Caps()
					_, nLabel := sPriorData.Caps()
					tmpTrXdata := mat64.NewDense(len(trRowName), nLabel, nil)
					//trX
					cLabel := 0
					for l := 0; l < nTrLabel; l++ {
						if ind[l] > 1 {
							for k := 0; k < len(trRowName); k++ {
								_, exist := idIdx[trRowName[k]]
								if exist {
									tmpTrXdata.Set(k, cLabel, sPriorData.At(idIdx[trRowName[k]], cLabel)/float64(ind[l]))
									//adding trY label as max value to trX
									if trYdata.At(k, l) == 1.0 {
										_, exist2 := cvTestMap[k]
										if !exist2 {
											tmpTrXdata.Set(k, cLabel, 1.0/float64(ind[l]))
										}
									}
								}
							}
							cLabel += 1
						}
					}
					nRow, _ := trXdataCV.Caps()
					if nRow == 0 {
						trXdataCV = tmpTrXdata
					} else {
						trXdataCV = src.ColStackMatrix(trXdataCV, tmpTrXdata)
					}
				} else {
					for j := 0; j < len(priorMatrixFile); j++ {
						priorData, priorGeneID, priorIdxToId := src.ReadNetwork(priorMatrixFile[j])
						sPriorData, ind := src.PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdataCV, idIdx, idxToId, trRowNameCV, trGeneMapCV, isDada, alpha, &wg, &mutex)
						indAccum = append(indAccum, ind...)
						_, nTrLabel := trYdataCV.Caps()
						_, nLabel := sPriorData.Caps()
						tmpTrXdata := mat64.NewDense(len(trRowName), nLabel, nil)
						//trX
						cLabel := 0
						for l := 0; l < nTrLabel; l++ {
							if ind[l] > 1 {
								for k := 0; k < len(trRowName); k++ {
									_, exist := idIdx[trRowName[k]]
									if exist {
										tmpTrXdata.Set(k, cLabel, sPriorData.At(idIdx[trRowName[k]], cLabel)/float64(ind[l]))
										if trYdata.At(k, l) == 1.0 {
											_, exist2 := cvTestMap[k]
											if !exist2 {
												tmpTrXdata.Set(k, cLabel, 1.0/float64(ind[l]))
											}
										}
									}
								}
								cLabel += 1
							}
						}
						nRow, _ := trXdataCV.Caps()
						if nRow == 0 {
							trXdataCV = tmpTrXdata
						} else {
							trXdataCV = src.ColStackMatrix(trXdataCV, tmpTrXdata)
						}
					}
				}
			}

			trainFold[f].SetXYinNestedTraining(cvTrain, trXdataCV, trYdata, []int{})
			testFold[f].SetXYinNestedTraining(cvTest, trXdataCV, trYdata, indAccum)
		}

		log.Print("testing and nested training ecoc matrix after propagation generated.")
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
		trainMeasure := mat64.NewDense(nL, 15, nil)
		testMeasure := mat64.NewDense(1, 7, nil)

		//traing data per hyperparameter
		YhPlattSet := make(map[int]*mat64.Dense)
		YhPlattSetCalibrated := make(map[int]*mat64.Dense)
		yPlattSet := make(map[int]*mat64.Dense)
		yPredSet := make(map[int]*mat64.Dense)
		iFoldMarker := make(map[int]*mat64.Dense)
		xSet := make(map[int]*mat64.Dense)
		//thresSet := mat64.NewDense(nL, nLabel, nil)

		for i := 0; i < nFold; i++ {
			YhSet, colSum := src.EcocRun(testFold[i].X, testFold[i].Y, trainFold[i].X, trainFold[i].Y, rankCut, reg, kSet, sigmaFctsSet, nFold, nK, &wg, &mutex)
			tsYfold := src.PosSelect(testFold[i].Y, colSum)

			c := 0
			//accum calculated training data
			for m := 0; m < nK; m++ {
				for n := 0; n < len(sigmaFctsSet); n++ {
					tsYhat := src.Zscore(YhSet[c])
					tsYhat, _ = src.Platt(tsYhat, tsYfold, tsYhat)
					//raw thres added
					rawThres := src.FscoreThres(tsYfold, tsYhat, fBetaThres)
					//accum to add information for KNN calibaration
					src.AccumTsYdata(i, c, colSum, YhSet[c], tsYfold, testFold[i].X, testFold[i].IndAccum, YhPlattSet, YhPlattSetCalibrated, yPlattSet, iFoldMarker, yPredSet, xSet, rawThres)
					c += 1
				}
			}
		}
		//update all meassures before or after KNN calibration
		for i := 0; i < nFold; i++ {
			c := 0
			for m := 0; m < nK; m++ {
				for n := 0; n < len(sigmaFctsSet); n++ {
					trainMeasure.Set(c, 0, float64(kSet[m]))
					trainMeasure.Set(c, 1, sigmaFctsSet[n])
					trainMeasure.Set(c, 2, trainMeasure.At(c, 2)+1.0)
					yPlattTrain, yPredTrain, xTrain, xTest, tsYhat, tsYfold := src.SubSetTrain(i, yPlattSet[c], YhPlattSet[c], yPredSet[c], xSet[c], iFoldMarker[c])
					//calculate platt scaled tsYhat again for measures
					tsYhat = src.Zscore(tsYhat)
					tsYhat, _ = src.Platt(tsYhat, tsYfold, tsYhat)
					thres := src.FscoreThres(tsYfold, tsYhat, fBetaThres)
					_ = src.PlattChopScale(tsYfold, tsYhat)
					accuracy, microF1, microAupr, macroAupr, kPrec, firstAupr := src.Report(tsYfold, tsYhat, thres, rankCut, false)
					trainMeasure.Set(c, 3, trainMeasure.At(c, 3)+accuracy)
					trainMeasure.Set(c, 5, trainMeasure.At(c, 5)+microF1)
					trainMeasure.Set(c, 7, trainMeasure.At(c, 7)+microAupr)
					trainMeasure.Set(c, 9, trainMeasure.At(c, 9)+macroAupr)
					trainMeasure.Set(c, 11, trainMeasure.At(c, 11)+kPrec)
					trainMeasure.Set(c, 13, trainMeasure.At(c, 13)+firstAupr)
					//probability to be recalibrated for label dependency, subset train by fold
					tsYhat = src.MultiLabelRecalibrate(nKnn, tsYhat, xTest, yPlattTrain, yPredTrain, xTrain, posLabelRls, negLabelRls, &wg, &mutex)
					thres = src.FscoreThres(tsYfold, tsYhat, fBetaThres)
					//update MultiLabelRecalibrate tsYhat to YhPlattSet
					src.YhPlattSetUpdate(i, c, YhPlattSetCalibrated, tsYhat, iFoldMarker[c])
					accuracy, microF1, microAupr, macroAupr, kPrec, firstAupr = src.Report(tsYfold, tsYhat, thres, rankCut, false)
					trainMeasure.Set(c, 4, trainMeasure.At(c, 4)+accuracy)
					trainMeasure.Set(c, 6, trainMeasure.At(c, 6)+microF1)
					trainMeasure.Set(c, 8, trainMeasure.At(c, 8)+microAupr)
					trainMeasure.Set(c, 10, trainMeasure.At(c, 10)+macroAupr)
					trainMeasure.Set(c, 12, trainMeasure.At(c, 12)+kPrec)
					trainMeasure.Set(c, 14, trainMeasure.At(c, 14)+firstAupr)
					c += 1

				}
			}
		}
		log.Print("pass training")

		//choosing all hyper parameters, nDim in CCA, lamda and kNN calibration
		cBestRaw, vBestRaw := src.BestHyperParameterSetByMeasure(trainMeasure, 13)
		cBestKnn, vBestKnn := src.BestHyperParameterSetByMeasure(trainMeasure, 14)
		cBest := 0
		isKnn := false
		if vBestKnn > vBestRaw {
			isKnn = true
			cBest = cBestKnn
		} else {
			cBest = cBestRaw
		}
		//best training parameters, data and max Pos scaling factor
		kSet = []int{int(trainMeasure.At(cBest, 0))}
		sigmaFctsSet = []float64{trainMeasure.At(cBest, 1)}
		//maxArr := []float64{}
		thres := mat64.NewDense(0, 0, nil)
		plattAB := mat64.NewDense(0, 0, nil)
		//platt scale for testing
		YhPlattScale := src.Zscore(YhPlattSet[cBest])
		YhPlattScale, plattAB = src.Platt(YhPlattScale, yPlattSet[cBest], YhPlattScale)
		maxArr := src.PlattChopScale(yPlattSet[cBest], YhPlattScale)
		if isKnn {
			thres = src.FscoreThres(yPlattSet[cBest], YhPlattSetCalibrated[cBest], fBetaThres)
		} else {
			thres = src.FscoreThres(yPlattSet[cBest], YhPlattScale, fBetaThres)
		}
		//testing run with cBest hyperparameter
		YhSet, _ := src.EcocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, sigmaFctsSet, nFold, 1, &wg, &mutex)
		tsYhat := src.Zscore(YhSet[0])
		tsYhat = src.PlattScaleSet(tsYhat, plattAB)
		src.TestDataPlattChopScale(tsYhat, maxArr)
		if isKnn {
			tsXdata = src.RefillIndCol(tsXdata, indAccum)
			tsYhat = src.MultiLabelRecalibrate(nKnn, tsYhat, tsXdata, yPlattSet[cBest], yPredSet[cBest], xSet[cBest], posLabelRls, negLabelRls, &wg, &mutex)
		}
		//corresponding testing measures
		c := 0
		i := 0
		for j := 0; j < len(sigmaFctsSet); j++ {
			accuracy, microF1, microAupr, macroAupr, _, _ := src.Report(tsYdata, tsYhat, thres, rankCut, false)
			testMeasure.Set(c, 0, float64(kSet[i]))
			testMeasure.Set(c, 1, sigmaFctsSet[j])
			testMeasure.Set(c, 2, testMeasure.At(c, 2)+1.0)
			testMeasure.Set(c, 3, testMeasure.At(c, 3)+accuracy)
			testMeasure.Set(c, 4, testMeasure.At(c, 4)+microF1)
			testMeasure.Set(c, 5, testMeasure.At(c, 5)+microAupr)
			testMeasure.Set(c, 6, testMeasure.At(c, 6)+macroAupr)
			c += 1
		}

		//result file.
		oFile := "./" + resFolder + "/cvTraining.measure.txt"
		src.WriteFile(oFile, trainMeasure)
		oFile = "./" + resFolder + "/cvTesting.measure.txt"
		src.WriteFile(oFile, testMeasure)
		oFile = "./" + resFolder + "/posLabelRls.txt"
		src.WriteFile(oFile, posLabelRls)
		oFile = "./" + resFolder + "/negLabelRls.txt"
		src.WriteFile(oFile, negLabelRls)
		oFile = "./" + resFolder + "/test.probMatrix.txt"
		src.WriteFile(oFile, tsYhat)
		oFile = "./" + resFolder + "/thres.txt"
		src.WriteFile(oFile, thres)
		oFile = "./" + resFolder + "/train.probMatrix.txt"
		src.WriteFile(oFile, YhPlattScale)
		oFile = "./" + resFolder + "/trainCalibrated.probMatrix.txt"
		src.WriteFile(oFile, YhPlattSetCalibrated[cBest])
		oFile = "./" + resFolder + "/reorder.trMatrix.txt"
		src.WriteFile(oFile, yPlattSet[cBest])

		//mem profile
		memprofile := resFolder + "/mem.prof"
		f, err2 := os.Create(memprofile)
		if err2 != nil {
			log.Fatal("could not create memory profile: ", err2)
		}
		defer f.Close()
		runtime.GC() // get up-to-date statistics
		if err2 := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("could not write memory profile: ", err2)
		}
		defer f.Close()

		log.Print("Program finished.")
		os.Exit(0)

	},
}

func init() {
	rootCmd.AddCommand(tuneCmd)

	tuneCmd.PersistentFlags().String("tsY", "data/human.bp.level1.set1.tsMatrix.txt", "test LabelSet")
	tuneCmd.PersistentFlags().String("trY", "data/human.bp.level1.set1.trMatrix.txt", "train LabelSet")
	tuneCmd.PersistentFlags().String("res", "resultEcoc", "resultFolder")

	tuneCmd.PersistentFlags().String("n", "data/hs_db_net.txt,data/hs_fus_net.txt", "network file")
	tuneCmd.PersistentFlags().String("p", "", "addtional prior file, use together with addPrior flag")
	tuneCmd.PersistentFlags().Int("t", 48, "number of threads")
	tuneCmd.PersistentFlags().Int("c", 3, "rank cut (alpha) for F1 calculation")
	tuneCmd.PersistentFlags().Int("k", 10, "number of nearest neighbors for multiabel probability calibration")
	tuneCmd.PersistentFlags().Int("nFold", 5, "number of folds for cross validation")
	tuneCmd.PersistentFlags().Bool("addPrior", false, "adding additional priors, default false")
	tuneCmd.PersistentFlags().Bool("r", false, "regularize CCA, default false")
	tuneCmd.Flags().Float64("alpha", 0.2, "alpha for propgation, default 0.6")
	tuneCmd.Flags().Bool("ec", false, "ec method for propgation, default false")
}

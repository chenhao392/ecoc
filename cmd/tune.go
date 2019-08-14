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
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	//"sync"
	//"unsafe"
	"github.com/chenhao392/ecoc/src"
	"github.com/spf13/cobra"
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
		src.PrintMemUsage()
		tsY, _ := cmd.Flags().GetString("tsY")
		trY, _ := cmd.Flags().GetString("trY")
		inNetworkFiles, _ := cmd.Flags().GetString("n")
		priorMatrixFiles, _ := cmd.Flags().GetString("p")
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		rankCut, _ := cmd.Flags().GetInt("c")
		reg, _ := cmd.Flags().GetBool("r")
		nFold, _ := cmd.Flags().GetInt("nFold")
		isDada, _ := cmd.Flags().GetBool("ec")
		alpha, _ := cmd.Flags().GetFloat64("alpha")
		isAddPrior, _ := cmd.Flags().GetBool("addPrior")

		rand.Seed(1)
		runtime.GOMAXPROCS(threads)
		//read data
		tsYdata, tsRowName, _, _ := src.ReadFile(tsY, true, true)
		trYdata, trRowName, _, _ := src.ReadFile(trY, true, true)
		tsXdata := mat64.NewDense(0, 0, nil)
		//trXdata := mat64.NewDense(0, 0, nil)
		vlXdata := mat64.NewDense(0, 0, nil)
		nTr, nLabel := trYdata.Caps()
		//split training data for nested cv
		folds := src.SOIS(trYdata, nFold)
		//for i := 0; i < 5; i++ {
		//	nPos := make([]int, nLabel)
		//	for j := 0; j < nLabel; j++ {
		//		for k := 0; k < len(folds[i]); k++ {
		//			if trYdata.At(folds[i][k], j) == 1 {
		//				nPos[j] += 1
		//			}
		//		}
		//	}
		//	for j := 0; j < nLabel; j++ {
		//		fmt.Printf("\t%d", nPos[j])
		//	}
		//	fmt.Printf("\n")
		//}
		trainFold := make([]src.CvFold, nFold-1)
		testFold := make([]src.CvFold, nFold-1)
		//validFold := make([]src.CvFold, 1)

		cvValid := make([]int, 0)
		cvValidMap := map[int]int{}
		//validation set for Platt scaling
		for j := 0; j < len(folds[0]); j++ {
			cvValid = append(cvValid, folds[0][j])
			cvValidMap[folds[0][j]] = folds[0][j]
		}
		cvTotalTrain := make([]int, 0)
		//the rest is for total training
		for j := 0; j < nTr; j++ {
			_, exist := cvValidMap[j]
			if !exist {
				cvTotalTrain = append(cvTotalTrain, j)
			}
		}
		// for filtering prior genes, only those in total training set are used for propagation
		trGeneMapTotalTrain := make(map[string]int)
		for i := 0; i < len(cvTotalTrain); i++ {
			trGeneMapTotalTrain[trRowName[cvTotalTrain[i]]] = cvTotalTrain[i]
		}

		//trX and trY for total training data
		trXdataTotalTrain := mat64.NewDense(0, 0, nil)
		_, nColY := trYdata.Caps()
		trYdataTotalTrain := mat64.NewDense(len(cvTotalTrain), nColY, nil)
		//row name and label data for total training gene set
		trRowNameTotalTrain := make([]string, 0)
		for s := 0; s < len(cvTotalTrain); s++ {
			trYdataTotalTrain.SetRow(s, trYdata.RawRowView(cvTotalTrain[s]))
			trRowNameTotalTrain = append(trRowNameTotalTrain, trRowName[cvTotalTrain[s]])
		}
		//row names and label data for validation set
		vlRowName := make([]string, 0)
		vlYdata := mat64.NewDense(len(cvValid), nColY, nil)
		for s := 0; s < len(cvValid); s++ {
			vlYdata.SetRow(s, trYdata.RawRowView(cvValid[s]))
			vlRowName = append(vlRowName, trRowName[cvValid[s]])
		}
		//network loading and label propagation
		inNetworkFile := strings.Split(inNetworkFiles, ",")
		priorMatrixFile := strings.Split(priorMatrixFiles, ",")
		for i := 0; i < len(inNetworkFile); i++ {
			//idIdx as gene -> idx in net
			fmt.Println(inNetworkFile[i])
			network, idIdx, idxToId := src.ReadNetwork(inNetworkFile[i])
			//network, idIdx, idxToId := readNetwork(*inNetworkFile)
			if !isAddPrior {
				sPriorData, ind := src.PropagateSet(network, trYdataTotalTrain, idIdx, trRowNameTotalTrain, trGeneMapTotalTrain, isDada, alpha, &wg, &mutex)
				tsXdata, trXdataTotalTrain = src.FeatureDataStack(sPriorData, tsRowName, trRowNameTotalTrain, idIdx, tsXdata, trXdataTotalTrain, trYdataTotalTrain, ind, true)
				vlXdata, _ = src.FeatureDataStack(sPriorData, vlRowName, trRowNameTotalTrain, idIdx, vlXdata, trXdataTotalTrain, trYdataTotalTrain, ind, false)
			} else {
				for j := 0; j < len(priorMatrixFile); j++ {
					priorData, priorGeneID, priorIdxToId := src.ReadNetwork(priorMatrixFile[j])
					sPriorData, ind := src.PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdataTotalTrain, idIdx, idxToId, trRowNameTotalTrain, trGeneMapTotalTrain, isDada, alpha, &wg, &mutex)
					tsXdata, trXdataTotalTrain = src.FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdataTotalTrain, trYdataTotalTrain, ind, true)
					vlXdata, _ = src.FeatureDataStack(sPriorData, vlRowName, trRowName, idIdx, vlXdata, trXdataTotalTrain, trYdataTotalTrain, ind, false)
				}
			}
		}

		_, nFea := trXdataTotalTrain.Caps()
		fmt.Println(nFea)
		if nFea < nLabel {
			fmt.Println("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
			os.Exit(0)
		}
		sigmaFctsSet := []float64{0.0001, 0.0025, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1, 1.23, 1.56, 2.04, 2.78, 4.0, 6.25, 11.11, 25.0, 100.0, 400.0, 10000.0, 40000.0, 1000000.0}
		//kSet := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		kSet := make([]int, 0)
		for i := 5; i <= 95; i += 10 {
			k := nLabel * i / 100
			if k > 0 {
				kSet = append(kSet, k)
			}
		}

		for f := 1; f < nFold; f++ {
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
				_, exist2 := cvValidMap[j]
				if !exist && !exist2 {
					cvTrain = append(cvTrain, j)
				}
			}
			//fmt.Println("cvTrain largest:", cvTrain[len(cvTrain)-1])
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
			for i := 0; i < len(inNetworkFile); i++ {
				//idIdx as gene -> idx in net
				network, idIdx, idxToId := src.ReadNetwork(inNetworkFile[i])
				if !isAddPrior {
					sPriorData, ind := src.PropagateSet(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, isDada, alpha, &wg, &mutex)
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

			trainFold[f-1].SetXYinNestedTraining(cvTrain, trXdataCV, trYdata)
			testFold[f-1].SetXYinNestedTraining(cvTest, trXdataCV, trYdata)
		}

		fmt.Println("pass ecoc matrix")
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
		err := os.MkdirAll("./"+resFolder, 0755)
		if err != nil {
			fmt.Println(err)
			return
		}
		//map
		//plattAset := mat64.NewDense(nL, nLabel, nil)
		//plattBset := mat64.NewDense(nL, nLabel, nil)
		//plattCountSet := mat64.NewDense(nL, nLabel, nil)
		//YhPlattSet := make(map[int]*mat64.Dense)
		//yPlattSet := make(map[int]*mat64.Dense)
		//thresSet := mat64.NewDense(nL, nLabel, nil)

		fmt.Println("pass ecoc")
		for i := 0; i < nFold-1; i++ {
			YhSet, colSum := src.EcocRun(testFold[i].X, testFold[i].Y, trainFold[i].X, trainFold[i].Y, rankCut, reg, kSet, sigmaFctsSet, nFold, nK, &wg, &mutex)
			//trYfold := src.PosSelect(trainFold[i].Y, colSum)
			tsYfold := src.PosSelect(testFold[i].Y, colSum)

			//update all meassures
			c := 0
			for m := 0; m < nK; m++ {
				for n := 0; n < len(sigmaFctsSet); n++ {
					tsYhat, _ := src.Platt(YhSet[c], tsYfold, YhSet[c])
					thres := src.FscoreThres(tsYfold, tsYhat)
					//src.AccumPlatt(c, colSum, plattAB, plattAset, plattBset, plattCountSet)
					//src.AccumThres(c, colSum, thresSet, thres)
					//src.AccumTsYdata(c, colSum, YhSet[c], tsYfold, YhPlattSet, yPlattSet)
					accuracy, microF1, microAupr, macroAupr := src.Report(tsYfold, tsYhat, thres, rankCut, false)
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
		fmt.Println("pass training")

		//sort by microAupr
		var sortMap []kv
		n, _ := trainMicroAupr.Caps()
		for i := 0; i < n; i++ {
			sortMap = append(sortMap, kv{i, trainMicroAupr.At(i, 3)})
		}
		sort.Slice(sortMap, func(i, j int) bool {
			return sortMap[i].Value > sortMap[j].Value
		})

		//best training aupr
		cBest := sortMap[0].Key
		//Platt
		//plattAB := src.SelectPlattAB(cBest, plattAset, plattBset, plattCountSet)
		//YhPlattScale, plattAB := src.Platt(YhPlattSet[cBest], yPlattSet[cBest], YhPlattSet[cBest])
		//thres := src.FscoreThres(yPlattSet[cBest], YhPlattScale)
		//k and sigma
		kSet = []int{int(trainMicroAupr.At(cBest, 0))}
		sigmaFctsSet = []float64{trainMicroAupr.At(cBest, 1)}
		vlYhSet, _ := src.EcocRun(vlXdata, vlYdata, trXdataTotalTrain, trYdataTotalTrain, rankCut, reg, kSet, sigmaFctsSet, nFold, 1, &wg, &mutex)
		YhSet, _ := src.EcocRun(tsXdata, tsYdata, trXdataTotalTrain, trYdataTotalTrain, rankCut, reg, kSet, sigmaFctsSet, nFold, 1, &wg, &mutex)
		//trYdata = src.PosSelect(trYdata, colSum)
		vlYhh, plattAB := src.Platt(vlYhSet[0], vlYdata, vlYhSet[0])
		thres := src.FscoreThres(vlYdata, vlYhh)
		tsYhat := src.PlattScaleSet(YhSet[0], plattAB)

		//corresponding testing measures
		c := 0
		i := 0
		for j := 0; j < len(sigmaFctsSet); j++ {
			accuracy, microF1, microAupr, macroAupr := src.Report(tsYdata, tsYhat, thres, rankCut, false)
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

		//result file.
		oFile := "./" + resFolder + "/cvTraining.microF1.txt"
		src.WriteFile(oFile, trainF1)
		oFile = "./" + resFolder + "/cvTraining.accuracy.txt"
		src.WriteFile(oFile, trainAccuracy)
		oFile = "./" + resFolder + "/cvTraining.macroAupr.txt"
		src.WriteFile(oFile, trainMacroAupr)
		oFile = "./" + resFolder + "/cvTraining.microAupr.txt"
		src.WriteFile(oFile, trainMicroAupr)
		oFile = "./" + resFolder + "/cvTesting.microF1.txt"
		src.WriteFile(oFile, testF1)
		oFile = "./" + resFolder + "/cvTesting.accuracy.txt"
		src.WriteFile(oFile, testAccuracy)
		oFile = "./" + resFolder + "/cvTesting.macroAupr.txt"
		src.WriteFile(oFile, testMacroAupr)
		oFile = "./" + resFolder + "/cvTesting.microAupr.txt"
		src.WriteFile(oFile, testMicroAupr)
		oFile = "./" + resFolder + "/test.probMatrix.txt"
		src.WriteFile(oFile, tsYhat)
		oFile = "./" + resFolder + "/thres.txt"
		src.WriteFile(oFile, thres)

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
	tuneCmd.PersistentFlags().Int("nFold", 5, "number of folds for cross validation")
	tuneCmd.PersistentFlags().Bool("addPrior", false, "adding additional priors, default false")
	tuneCmd.PersistentFlags().Bool("r", false, "regularize CCA, default false")
	tuneCmd.Flags().Float64("alpha", 0.2, "alpha for propgation, default 0.6")
	tuneCmd.Flags().Bool("ec", false, "ec method for propgation, default false")
}

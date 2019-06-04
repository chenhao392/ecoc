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
	//"github.com/gonum/stat"
	//"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	//"strconv"
	"strings"
	"sync"
	//"unsafe"
	"github.com/chenhao392/ecoc/src"
	"github.com/spf13/cobra"
)

type kv struct {
	Key   int
	Value float64
}

var wg sync.WaitGroup
var mutex sync.Mutex

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
		//tsX, _ := cmd.Flags().GetString("tsX")
		tsY, _ := cmd.Flags().GetString("tsY")
		//trX, _ := cmd.Flags().GetString("trX")
		trY, _ := cmd.Flags().GetString("trY")
		inNetworkFiles, _ := cmd.Flags().GetString("n")
		priorMatrixFiles, _ := cmd.Flags().GetString("p")
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		rankCut, _ := cmd.Flags().GetInt("c")
		reg, _ := cmd.Flags().GetBool("r")
		nFold, _ := cmd.Flags().GetInt("nFold")

		kSet := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		sigmaFctsSet := []float64{0.0001, 0.0025, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1, 1.23, 1.56, 2.04, 2.78, 4.0, 6.25, 11.11, 25.0, 100.0, 400.0, 10000.0}
		//sigmaFctsSet := []float64{0.01, 1, 100.0}
		rand.Seed(1)
		runtime.GOMAXPROCS(threads)
		//read data
		tsYdata, tsRowName, _, _ := src.ReadFile(tsY, true, true)
		trYdata, trRowName, _, _ := src.ReadFile(trY, true, true)
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
		for i := 0; i < len(inNetworkFile); i++ {
			//idIdx as gene -> idx in net
			fmt.Println(inNetworkFile[i])
			network, idIdx, idxToId := src.ReadNetwork(inNetworkFile[i])
			//network, idIdx, idxToId := readNetwork(*inNetworkFile)
			if priorMatrixFiles == "" {
				sPriorData := src.PropagateSet(network, trYdata, idIdx, trRowName, trGeneMap, &wg, &mutex)
				tsXdata, trXdata = src.FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata)
			} else {
				for j := 0; j < len(priorMatrixFile); j++ {
					priorData, priorGeneID, priorIdxToId := src.ReadNetwork(priorMatrixFile[j])
					sPriorData := src.PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdata, idIdx, idxToId, trRowName, trGeneMap, &wg, &mutex)
					tsXdata, trXdata = src.FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata)
				}
			}
		}

		_, nFea := trXdata.Caps()
		nTr, nLabel := trYdata.Caps()
		if nFea < nLabel {
			fmt.Println("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
			os.Exit(0)
		}
		//split training data for nested cv
		idxPerm := rand.Perm(nTr)
		trainFold := make([]src.CvFold, nFold)
		testFold := make([]src.CvFold, nFold)

		for f := 0; f < nFold; f++ {
			fmt.Println("5, fold:", f)
			cvTrain := make([]int, 0)
			cvTest := make([]int, 0)
			cvTestMap := map[int]int{}
			for j := f * nTr / nFold; j < (f+1)*nTr/nFold-1; j++ {
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
				fmt.Println(inNetworkFile[i])
				network, idIdx, idxToId := src.ReadNetwork(inNetworkFile[i])
				sPriorData := mat64.NewDense(0, 0, nil)
				//network, idIdx, idxToId := readNetwork(*inNetworkFile)
				if priorMatrixFiles == "" {
					sPriorData = src.PropagateSet(network, trYdataCV, idIdx, trRowNameCV, trGeneMapCV, &wg, &mutex)
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
						trXdataCV = src.ColStackMatrix(trXdataCV, tmpTrXdata)
					}
				} else {
					for j := 0; j < len(priorMatrixFile); j++ {
						priorData, priorGeneID, priorIdxToId := src.ReadNetwork(priorMatrixFile[j])
						sPriorData = src.PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdataCV, idIdx, idxToId, trRowNameCV, trGeneMapCV, &wg, &mutex)
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
							trXdataCV = src.ColStackMatrix(trXdataCV, tmpTrXdata)
						}
					}
				}
			}

			trainFold[f].SetXYinNestedTraining(cvTrain, trXdataCV, trYdata)
			testFold[f].SetXYinNestedTraining(cvTest, trXdataCV, trYdata)
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
		fmt.Println("pass ecoc")
		for i := 0; i < nFold; i++ {
			YhSet := src.EcocRun(testFold[i].X, testFold[i].Y, trainFold[i].X, trainFold[i].Y, rankCut, reg, kSet, sigmaFctsSet, nFold, nK, &wg, &mutex)
			//update all meassures
			c := 0
			for m := 0; m < nK; m++ {
				for n := 0; n < len(sigmaFctsSet); n++ {
					microF1, accuracy, macroAupr, microAupr := src.Single_compute(testFold[i].Y, YhSet[c], rankCut)
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
		YhSet := src.EcocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, sigmaFctsSet, nFold, 1, &wg, &mutex)
		//corresponding testing measures
		c := 0
		i := 0
		for j := 0; j < len(sigmaFctsSet); j++ {
			microF1, accuracy, macroAupr, microAupr := src.Single_compute(tsYdata, YhSet[c], rankCut)
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
		src.WriteFile(oFile, YhSet[0])
		os.Exit(0)

	},
}

func init() {
	rootCmd.AddCommand(tuneCmd)

	//tuneCmd.PersistentFlags().String("tsX", "", "test FeatureSet")
	tuneCmd.PersistentFlags().String("tsY", "data/human.bp.level1.set1.tsMatrix.txt", "test LabelSet")
	//tuneCmd.PersistentFlags().String("trX", "", "train FeatureSet")
	tuneCmd.PersistentFlags().String("trY", "data/human.bp.level1.set1.trMatrix.txt", "train LabelSet")
	tuneCmd.PersistentFlags().String("res", "resultEcoc", "resultFolder")

	tuneCmd.PersistentFlags().String("n", "data/hs_exp_net.txt", "network file")
	tuneCmd.PersistentFlags().String("p", "", "prior/known gene file")
	tuneCmd.PersistentFlags().Int("t", 48, "number of threads")
	tuneCmd.PersistentFlags().Int("c", 3, "rank cut (alpha) for F1 calculation")
	tuneCmd.PersistentFlags().Int("nFold", 5, "number of folds for cross validation")
	tuneCmd.PersistentFlags().Bool("r", false, "regularize CCA, default false")
}

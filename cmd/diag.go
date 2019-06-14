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

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"os"
	"runtime"
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

// diagCmd represents the diag command
var diagCmd = &cobra.Command{
	Use:   "diag",
	Short: "diag run of ecoc",
	Long:  `diag run of ecoc`,
	Run: func(cmd *cobra.Command, args []string) {
		tsY, _ := cmd.Flags().GetString("tsY")
		trY, _ := cmd.Flags().GetString("trY")
		inNetworkFiles, _ := cmd.Flags().GetString("n")
		priorMatrixFiles, _ := cmd.Flags().GetString("p")
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		rankCut, _ := cmd.Flags().GetInt("c")
		reg, _ := cmd.Flags().GetBool("r")
		nFold, _ := cmd.Flags().GetInt("nFold")

		kSet := []int{1}
		sigmaFctsSet := []float64{1.0}
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
			if priorMatrixFiles == "" {
				sPriorData, ind := src.PropagateSet(network, trYdata, idIdx, trRowName, trGeneMap, &wg, &mutex)
				tsXdata, trXdata = src.FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
			} else {
				for j := 0; j < len(priorMatrixFile); j++ {
					priorData, priorGeneID, priorIdxToId := src.ReadNetwork(priorMatrixFile[j])
					sPriorData, ind := src.PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdata, idIdx, idxToId, trRowName, trGeneMap, &wg, &mutex)
					tsXdata, trXdata = src.FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
				}
			}
		}
		_, nFea := trXdata.Caps()
		_, nLabel := trYdata.Caps()
		if nFea < nLabel {
			fmt.Println("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
			os.Exit(0)
		}
		//run
		YhSet := src.EcocRun(tsXdata, tsYdata, trXdata, trYdata, rankCut, reg, kSet, sigmaFctsSet, nFold, 1, &wg, &mutex)
		rebaData := src.RebalanceData(trYdata)
		//measures
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
		//corresponding testing measures
		c := 0
		i := 0
		for j := 0; j < len(sigmaFctsSet); j++ {
			microF1, accuracy, macroAupr, microAupr := src.Report(tsYdata, YhSet[c], rebaData, rankCut, false)

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
		oFile := "./" + resFolder + "/cvTesting.microF1.txt"
		src.WriteFile(oFile, testF1)
		oFile = "./" + resFolder + "/cvTesting.accuracy.txt"
		src.WriteFile(oFile, testAccuracy)
		oFile = "./" + resFolder + "/cvTesting.macroAupr.txt"
		src.WriteFile(oFile, testMacroAupr)
		oFile = "./" + resFolder + "/cvTesting.microAupr.txt"
		src.WriteFile(oFile, testMicroAupr)
		oFile = "./" + resFolder + "/test.probMatrix.txt"
		src.WriteFile(oFile, YhSet[0])
		oFile = "./" + resFolder + "/rebalance.scale.txt"
		src.WriteFile(oFile, rebaData)
		os.Exit(0)
	},
}

func init() {
	rootCmd.AddCommand(diagCmd)
	//diagCmd.PersistentFlags().String("tsX", "", "test FeatureSet")
	diagCmd.PersistentFlags().String("tsY", "data/human.bp.level1.set1.tsMatrix.txt", "test LabelSet")
	//diagCmd.PersistentFlags().String("trX", "", "train FeatureSet")
	diagCmd.PersistentFlags().String("trY", "data/human.bp.level1.set1.trMatrix.txt", "train LabelSet")
	diagCmd.PersistentFlags().String("res", "resultDiag", "resultFolder")

	diagCmd.PersistentFlags().String("n", "data/hs_exp_net.txt", "network file")
	diagCmd.PersistentFlags().String("p", "", "prior/known gene file")
	diagCmd.PersistentFlags().Int("t", 48, "number of threads")
	diagCmd.PersistentFlags().Int("c", 3, "rank cut (alpha) for F1 calculation")
	diagCmd.PersistentFlags().Int("nFold", 5, "number of folds for cross validation")
	diagCmd.PersistentFlags().Bool("r", false, "regularize CCA, default false")
}

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
	//"strings"
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
		tsX, _ := cmd.Flags().GetString("tsX")
		trX, _ := cmd.Flags().GetString("trX")
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		rankCut, _ := cmd.Flags().GetInt("c")
		reg, _ := cmd.Flags().GetBool("r")
		nFold, _ := cmd.Flags().GetInt("nFold")
		k, _ := cmd.Flags().GetInt("k")
		s, _ := cmd.Flags().GetFloat64("s")

		kSet := make([]int, 0)
		sigmaFctsSet := make([]float64, 0)
		kSet = append(kSet, k)
		sigmaFctsSet = append(sigmaFctsSet, s)
		rand.Seed(1)
		runtime.GOMAXPROCS(threads)
		//read data
		tsYdata, _, _, _ := src.ReadFile(tsY, true, true)
		trYdata, _, _, _ := src.ReadFile(trY, true, true)
		tsXdata, _, _, _ := src.ReadFile(tsX, false, false)
		trXdata, _, _, _ := src.ReadFile(trX, false, false)

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
		oFile := "./" + resFolder + ".cvTesting.microF1.txt"
		src.WriteFile(oFile, testF1)
		oFile = "./" + resFolder + ".cvTesting.accuracy.txt"
		src.WriteFile(oFile, testAccuracy)
		oFile = "./" + resFolder + ".cvTesting.macroAupr.txt"
		src.WriteFile(oFile, testMacroAupr)
		oFile = "./" + resFolder + ".cvTesting.microAupr.txt"
		src.WriteFile(oFile, testMicroAupr)
		oFile = "./" + resFolder + ".test.probMatrix.txt"
		src.WriteFile(oFile, YhSet[0])
		oFile = "./" + resFolder + ".rebalance.scale.txt"
		src.WriteFile(oFile, rebaData)
		os.Exit(0)
	},
}

func init() {
	rootCmd.AddCommand(diagCmd)
	diagCmd.PersistentFlags().String("tsX", "", "test FeatureSet")
	diagCmd.PersistentFlags().String("tsY", "data/human.bp.level1.set1.tsMatrix.txt", "test LabelSet")
	diagCmd.PersistentFlags().String("trX", "", "train FeatureSet")
	diagCmd.PersistentFlags().String("trY", "data/human.bp.level1.set1.trMatrix.txt", "train LabelSet")
	diagCmd.PersistentFlags().String("res", "resultDiag", "resultFolder")

	diagCmd.PersistentFlags().Int("t", 48, "number of threads")
	diagCmd.PersistentFlags().Int("c", 3, "rank cut (alpha) for F1 calculation")
	diagCmd.PersistentFlags().Int("k", 9, "number of CCA dims")
	diagCmd.PersistentFlags().Float64("s", 10000.0, "1/lamda^2")
	diagCmd.PersistentFlags().Int("nFold", 5, "number of folds for cross validation")
	diagCmd.PersistentFlags().Bool("r", false, "regularize CCA, default false")
}

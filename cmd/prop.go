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
	"github.com/chenhao392/ecoc/src"
	"github.com/gonum/matrix/mat64"
	"github.com/spf13/cobra"
	"math/rand"
	"os"
	"runtime"
	"strings"
)

// propCmd represents the prop command
var propCmd = &cobra.Command{
	Use:   "prop",
	Short: "label propagation",
	Long:  `generating ecoc matrix using label propagation`,
	Run: func(cmd *cobra.Command, args []string) {
		tsY, _ := cmd.Flags().GetString("tsY")
		trY, _ := cmd.Flags().GetString("trY")
		inNetworkFiles, _ := cmd.Flags().GetString("n")
		priorMatrixFiles, _ := cmd.Flags().GetString("p")
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		isDada, _ := cmd.Flags().GetBool("ec")
		alpha, _ := cmd.Flags().GetFloat64("alpha")
		isAddPrior, _ := cmd.Flags().GetBool("addPrior")
		rand.Seed(1)
		runtime.GOMAXPROCS(threads)
		//read data
		_, tsRowName, _, _ := src.ReadFile(tsY, true, true)
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
			if !isAddPrior {
				sPriorData, ind := src.PropagateSet(network, trYdata, idIdx, trRowName, trGeneMap, isDada, alpha, &wg, &mutex)
				tsXdata, trXdata = src.FeatureDataStack(sPriorData, tsRowName, trRowName, idIdx, tsXdata, trXdata, trYdata, ind)
			} else {
				for j := 0; j < len(priorMatrixFile); j++ {
					priorData, priorGeneID, priorIdxToId := src.ReadNetwork(priorMatrixFile[j])
					sPriorData, ind := src.PropagateSetWithPrior(priorData, priorGeneID, priorIdxToId, network, trYdata, idIdx, idxToId, trRowName, trGeneMap, isDada, alpha, &wg, &mutex)
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
		//result file.
		oFile := "./" + resFolder + ".trX.txt"
		src.WriteFile(oFile, trXdata)
		oFile = "./" + resFolder + ".tsX.txt"
		src.WriteFile(oFile, tsXdata)

	},
}

func init() {
	rootCmd.AddCommand(propCmd)
	propCmd.Flags().String("tsY", "data/human.bp.level1.set1.tsMatrix.txt", "test LabelSet")
	propCmd.Flags().String("trY", "data/human.bp.level1.set1.trMatrix.txt", "train LabelSet")
	propCmd.Flags().String("res", "ecoc", "resultBase")

	propCmd.Flags().String("n", "data/hs_exp_net.txt", "network file")
	propCmd.Flags().String("p", "", "addtional prior file, use together with addPrior flag")
	propCmd.Flags().Int("t", 48, "number of threads")
	propCmd.Flags().Int("c", 3, "rank cut (alpha) for F1 calculation")
	propCmd.Flags().Int("nFold", 5, "number of folds for cross validation")
	propCmd.Flags().Bool("ec", false, "ec method for propgation, default false")
	propCmd.Flags().Bool("addPrior", false, "adding additional priors, default false")
	propCmd.Flags().Float64("alpha", 0.2, "alpha for propgation, default 0.6")
}

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
	"github.com/chenhao392/ecoc/src"
	"github.com/spf13/cobra"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
)

// predCmd represents the pred command
var predCmd = &cobra.Command{
	Use:   "pred",
	Short: "prediction with specific hyperparameters",
	Long: `

  ______ _____ ____   _____   _____  _____  ______ _____  
 |  ____/ ____/ __ \ / ____| |  __ \|  __ \|  ____|  __ \ 
 | |__ | |   | |  | | |      | |__) | |__) | |__  | |  | |
 |  __|| |   | |  | | |      |  ___/|  _  /|  __| | |  | |
 | |___| |___| |__| | |____  | |    | | \ \| |____| |__| |
 |______\_____\____/ \_____| |_|    |_|  \_\______|_____/ 
		                                                             
		                                                             

	Hyperparameter tuning and benchmarking for the following parameters.
 1) number of CCA dimensions for explaining the label dependency.
 2) the trade-off between the gaussion and binomial model in decoding.

 The inputs are (1) gene-gene network or a set of network 
 and (2) multi-label gene by label matrices for training and
 testing, where "1" mark a gene annotated by a label.  
 
 1) The network file is a tab-delimited file with three columns. 
    The first two columns define gene-gene interactions using 
    the gene IDs. The third column is the confidence score. Multiple 
    network files are also supported, with the file names concatenated
    together with comma(s). 

 2) The multi-label matrix is a tab-delimited file with each gene 
    for one row and each label for one column. If a gene is annotated
    with a label, the corresponding cell is filled with 1, otherwise 0. 

 Sample usages:
   ecoc pred --trY trMatrix.txt --tsY tsMatrix.txt \
             --n net1.txt,net2.txt --nFold 2 -t 4`,
	Run: func(cmd *cobra.Command, args []string) {
		if !cmd.Flags().Changed("trY") {
			cmd.Help()
			os.Exit(0)
		}
		tsY, _ := cmd.Flags().GetString("tsY")
		trY, _ := cmd.Flags().GetString("trY")
		inNetworkFiles, _ := cmd.Flags().GetString("n")
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		objFuncIndex, _ := cmd.Flags().GetInt("o")
		rankCut, _ := cmd.Flags().GetInt("c")
		nKnn, _ := cmd.Flags().GetInt("k")
		nDim, _ := cmd.Flags().GetInt("d")
		lamda, _ := cmd.Flags().GetFloat64("l")
		isKnn, _ := cmd.Flags().GetBool("isCali")
		isPerLabel, _ := cmd.Flags().GetBool("isPerLabel")
		reg, _ := cmd.Flags().GetBool("r")
		nFold, _ := cmd.Flags().GetInt("nFold")
		isDada, _ := cmd.Flags().GetBool("ec")
		//alpha, _ := cmd.Flags().GetFloat64("alpha")
		isVerbose, _ := cmd.Flags().GetBool("v")
		fBetaThres := 1.0
		isAutoBeta := false

		//result dir and logging
		logFile := src.Init(resFolder)
		defer logFile.Close()
		log.SetOutput(logFile)

		//program start
		log.Print("Program started.")
		rand.Seed(1)
		runtime.GOMAXPROCS(threads)
		debug.SetGCPercent(50)

		//read data
		tsYdata, tsRowName, _, _ := src.ReadFile(tsY, true, true)
		trYdata, trRowName, _, _ := src.ReadFile(trY, true, true)
		posLabelRls, negLabelRls, transLabels := src.LabelRelationship(trYdata)
		inNetworkFile := strings.Split(inNetworkFiles, ",")
		networkSet, idIdxSet := src.ReadAndNormNetworks(inNetworkFile, 1, &wg, &mutex)
		//folds
		folds := src.SOIS(trYdata, nFold, 10, 2, true)
		//alphaEstimate
		alphaSet := src.NpAlphaEstimate(folds, trRowName, tsRowName, trYdata, networkSet, idIdxSet, transLabels, isDada, threads, &wg, &mutex)
		tsXdata, trXdata, indAccum := src.PropagateNetworks(trRowName, tsRowName, trYdata, networkSet, idIdxSet, transLabels, isDada, alphaSet, threads, &wg, &mutex)
		nTr, nFea := trXdata.Caps()
		_, nLabel := trYdata.Caps()
		if nFea < nLabel {
			log.Print("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
			os.Exit(0)
		}

		//min dims, potential bug when cv set's minDims is smaller
		minDims := int(math.Min(float64(nFea), float64(nLabel)))
		if nDim >= minDims {
			nDim = minDims - 1
			log.Print("number of dimensions larger than number of labels, reduced to ", nDim, ".")
		}
		//prepare hyperparameter grid
		kSet, lamdaSet := src.HyperParameterSet(nLabel, nDim, nDim, lamda, lamda, 1, 1)
		nK := len(kSet)

		//rands
		rand.Seed(1)
		randValues := src.RandListFromUniDist(nTr, nFea)

		//split training data for nested cv
		trainFold := make([]src.CvFold, nFold)
		testFold := make([]src.CvFold, nFold)

		//nested cv training data propagation on networks
		for f := 0; f < nFold; f++ {
			cvTrain, cvTest, trXdataCV, indAccum, _ := src.PropagateNetworksCV(f, folds, trRowName, tsRowName, trYdata, networkSet, idIdxSet, transLabels, isDada, alphaSet, threads, &wg, &mutex)
			trainFold[f].SetXYinNestedTraining(cvTrain, trXdataCV, trYdata, []int{})
			testFold[f].SetXYinNestedTraining(cvTest, trXdataCV, trYdata, indAccum)
		}

		log.Print("testing and nested training ecoc matrix after propagation generated.")
		//tune and predict
		trainMeasure, testMeasure, tsYhat, Yhat, YhatCalibrated, Ylabel := src.TuneAndPredict(objFuncIndex, nFold, folds, randValues, fBetaThres, isAutoBeta, nK, nKnn, isPerLabel, isKnn, kSet, lamdaSet, reg, rankCut, trainFold, testFold, indAccum, tsXdata, tsYdata, trXdata, trYdata, posLabelRls, negLabelRls, &wg, &mutex)
		//result file
		src.WriteOutputFiles(isVerbose, resFolder, trainMeasure, testMeasure, posLabelRls, negLabelRls, tsYhat, Yhat, YhatCalibrated, Ylabel)
		log.Print("Program finished.")
		os.Exit(0)
	},
}

func init() {
	rootCmd.AddCommand(predCmd)
	predCmd.Flags().Float64("alpha", 0.2, "alpha value for a single label propgation\n")
	predCmd.Flags().Float64("mlsRatio", 0.1, "multi-label SMOTE ratio\n")
	predCmd.Flags().Int("c", 3, "top c predictions for a gene to used\nin multi-label F1 calculation")
	predCmd.Flags().Int("d", 1, "number of dimensions")
	predCmd.Flags().Bool("ec", false, "experimental label propgation alternative\n(default false)")
	predCmd.Flags().Bool("isCali", false, "nearest neighbors calibration for the predictions\n(default false)")
	predCmd.Flags().Bool("isPerLabel", false, "training objection as the aupr of first label/column\n(default false)")
	predCmd.Flags().Int("k", 10, "number of nearest neighbors \nfor multiabel probability calibration\n")
	predCmd.Flags().Float64("l", 1.0, "lamda balancing bernoulli and gaussian potentials\n")
	predCmd.Flags().String("n", "data/net1.txt,data/net2.txt", "three columns network file(s)\n")
	predCmd.Flags().Int("nFold", 2, "number of folds for cross validation\n")
	predCmd.Flags().Bool("r", false, "experimental regularized CCA\n(default false)")
	predCmd.Flags().String("res", "result", "result folder")
	predCmd.Flags().Int("t", 4, "number of threads")
	predCmd.Flags().Int("o", 1, "object function choice")
	predCmd.Flags().String("trY", "data/trMatrix.txt", "train label matrix")
	predCmd.Flags().String("tsY", "data/tsMatrix.txt", "test label matrix")
	predCmd.Flags().Bool("v", false, "verbose outputs")
}

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

// tuneCmd represents the tune command
var tuneCmd = &cobra.Command{
	Use:   "tune",
	Short: "hyperparameter tuning and automatic prediction",
	Long: `
  ______ _____ ____   _____   _______ _    _ _   _ ______ 
 |  ____/ ____/ __ \ / ____| |__   __| |  | | \ | |  ____|
 | |__ | |   | |  | | |         | |  | |  | |  \| | |__   
 |  __|| |   | |  | | |         | |  | |  | | . \ |  __|  
 | |___| |___| |__| | |____     | |  | |__| | |\  | |____ 
 |______\_____\____/ \_____|    |_|   \____/|_| \_|______|
		                                                             
		                                                             
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
   ecoc tune --trY trMatrix.txt --tsY tsMatrix.txt \
             --n net1.txt,net2.txt --nFold 2 --t 4`,

	Run: func(cmd *cobra.Command, args []string) {
		if !cmd.Flags().Changed("trY") {
			cmd.Help()
			os.Exit(0)
		}
		tsY, _ := cmd.Flags().GetString("tsY")
		trY, _ := cmd.Flags().GetString("trY")
		inNetworkFiles, _ := cmd.Flags().GetString("n")
		//priorMatrixFiles, _ := cmd.Flags().GetString("p")
		priorMatrixFiles := ""
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		rankCut, _ := cmd.Flags().GetInt("c")
		nKnn, _ := cmd.Flags().GetInt("k")
		isKnn, _ := cmd.Flags().GetBool("isCali")
		isFirst, _ := cmd.Flags().GetBool("isFirstLabel")
		reg, _ := cmd.Flags().GetBool("r")
		nFold, _ := cmd.Flags().GetInt("nFold")
		isDada, _ := cmd.Flags().GetBool("ec")
		alpha, _ := cmd.Flags().GetFloat64("alpha")
		isVerbose, _ := cmd.Flags().GetBool("v")
		//isAddPrior, _ := cmd.Flags().GetBool("addPrior")
		isAddPrior := false
		fBetaThres := 1.0

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
		posLabelRls, negLabelRls := src.LabelRelationship(trYdata)
		inNetworkFile := strings.Split(inNetworkFiles, ",")
		priorMatrixFile := strings.Split(priorMatrixFiles, ",")
		tsXdata, trXdata, indAccum := src.ReadNetworkPropagate(trRowName, tsRowName, trYdata, inNetworkFile, priorMatrixFile, isAddPrior, isDada, alpha, &wg, &mutex)
		_, nFea := trXdata.Caps()
		_, nLabel := trYdata.Caps()
		if nFea < nLabel {
			log.Print("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
			os.Exit(0)
		}

		//prepare hyperparameter grid
		kSet, _, lamdaSet := src.HyperParameterSet(nLabel, 0.025, 0.225, 8)
		//_, sigmaFctsSet2, _ := src.HyperParameterSet(nLabel, 0.005, 0.025, 4)
		//sigmaFctsSet = append(sigmaFctsSet2, sigmaFctsSet...)

		//min dims, potential bug when cv set's minDims is smaller
		minDims := int(math.Min(float64(nFea), float64(nLabel)))
		nK := 0
		for k := 0; k < len(kSet); k++ {
			if kSet[k] < minDims {
				nK += 1
			}
		}

		//split training data for nested cv
		folds := src.SOIS(trYdata, nFold, 10, 0, true)
		trainFold := make([]src.CvFold, nFold)
		testFold := make([]src.CvFold, nFold)
		//nested cv training data propagation on networks
		for f := 0; f < nFold; f++ {
			cvTrain, cvTest, trXdataCV, indAccum := src.ReadNetworkPropagateCV(f, folds, trRowName, tsRowName, trYdata, inNetworkFile, priorMatrixFile, isAddPrior, isDada, alpha, &wg, &mutex)
			trainFold[f].SetXYinNestedTraining(cvTrain, trXdataCV, trYdata, []int{})
			testFold[f].SetXYinNestedTraining(cvTest, trXdataCV, trYdata, indAccum)
		}
		log.Print("testing and nested training ecoc matrix after propagation generated.")
		//tune and predict
		trainMeasure, testMeasure, tsYhat, thres, Yhat, YhatCalibrated, Ylabel := src.TuneAndPredict(nFold, folds, fBetaThres, nK, nKnn, isFirst, isKnn, kSet, lamdaSet, reg, rankCut, trainFold, testFold, indAccum, tsXdata, tsYdata, trXdata, trYdata, posLabelRls, negLabelRls, &wg, &mutex)
		//result file
		src.WriteOutputFiles(isVerbose, resFolder, trainMeasure, testMeasure, posLabelRls, negLabelRls, tsYhat, thres, Yhat, YhatCalibrated, Ylabel)
		log.Print("Program finished.")
		os.Exit(0)

	},
}

func init() {
	rootCmd.AddCommand(tuneCmd)

	tuneCmd.Flags().Float64("alpha", 0.2, "alpha value for a single label propgation\n")
	tuneCmd.Flags().Int("c", 3, "top c predictions for a gene to used\nin multi-label F1 calculation")
	tuneCmd.Flags().Bool("ec", false, "experimental label propgation alternative\n(default false)")
	tuneCmd.Flags().Bool("isCali", false, "nearest neighbors calibration for the predictions\n(default false)")
	tuneCmd.Flags().Bool("isFirstLabel", false, "training objection as the aupr of first label/column\n(default false)")
	tuneCmd.Flags().Int("k", 10, "number of nearest neighbors \nfor multiabel probability calibration\n")
	tuneCmd.Flags().String("n", "data/net1.txt,data/net2.txt", "three columns network file(s)\n")
	tuneCmd.Flags().Int("nFold", 2, "number of folds for cross validation\n")
	tuneCmd.Flags().Bool("r", false, "experimental regularized CCA\n(default false)")
	tuneCmd.Flags().String("res", "result", "result folder")
	tuneCmd.Flags().Int("t", 4, "number of threads")
	tuneCmd.Flags().String("trY", "data/trMatrix.txt", "train label matrix")
	tuneCmd.Flags().String("tsY", "data/tsMatrix.txt", "test label matrix")
	tuneCmd.Flags().Bool("v", false, "verbose outputs")

	//tuneCmd.Flags().String("p", "", "addtional prior file, use together with addPrior flag")
	//tuneCmd.PersistentFlags().Bool("addPrior", false, "adding additional priors, default false")
}

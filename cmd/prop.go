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
	"github.com/gonum/matrix/mat64"
	"github.com/spf13/cobra"
	"log"
	"math/rand"
	"os"
	"runtime"
	"strings"
)

// propCmd represents the prop command
var propCmd = &cobra.Command{
	Use:   "prop",
	Short: "multi-label propagation",
	Long: `

  ______ _____ ____   _____   _____  _____   ____  _____  
 |  ____/ ____/ __ \ / ____| |  __ \|  __ \ / __ \|  __ \ 
 | |__ | |   | |  | | |      | |__) | |__) | |  | | |__) |
 |  __|| |   | |  | | |      |  ___/|  _  /| |  | |  ___/ 
 | |___| |___| |__| | |____  | |    | | \ \| |__| | |     
 |______\_____\____/ \_____| |_|    |_|  \_\\____/|_|     


Propagating a set of labels on networks.
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
     ecoc prop --trY trMatrix.txt --tsGene tsMatrix.txt \
	 --n net1.txt,net2.txt --t 4
	`,
	Run: func(cmd *cobra.Command, args []string) {
		if !cmd.Flags().Changed("trY") {
			cmd.Help()
			os.Exit(0)
		}
		tsGene, _ := cmd.Flags().GetString("tsGene")
		trY, _ := cmd.Flags().GetString("trY")
		inNetworkFiles, _ := cmd.Flags().GetString("n")
		//priorMatrixFiles, _ := cmd.Flags().GetString("p")
		resFolder, _ := cmd.Flags().GetString("res")
		threads, _ := cmd.Flags().GetInt("t")
		isDada, _ := cmd.Flags().GetBool("ec")
		alpha, _ := cmd.Flags().GetFloat64("alpha")
		//isAddPrior, _ := cmd.Flags().GetBool("addPrior")
		priorMatrixFiles := ""
		isAddPrior := false

		rand.Seed(1)
		runtime.GOMAXPROCS(threads)
		//result dir and logging
		logFile := src.Init(resFolder)
		defer logFile.Close()
		log.SetOutput(logFile)

		//program start
		log.Print("Program started.")
		//read data
		tsRowName := src.ReadIDfile(tsGene)
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
			log.Print("loading network file: ", inNetworkFile[i])
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
			log.Print("number of features less than number of labels to classify.", nFea, nLabel, "\nexit...")
			os.Exit(0)
		}
		//result file.
		oFile := "./" + resFolder + ".trX.txt"
		src.WriteFile(oFile, trXdata, nil, false)
		oFile = "./" + resFolder + ".tsX.txt"
		src.WriteFile(oFile, tsXdata, nil, false)
		log.Print("Program finished.")
	},
}

func init() {
	rootCmd.AddCommand(propCmd)
	//propCmd.Flags().Bool("addPrior", false, "adding additional priors, default false")
	propCmd.Flags().Float64("alpha", 0.2, "alpha for propgation, default 0.6")
	propCmd.Flags().Bool("ec", false, "experimental label propgation alternative\n(default false)")
	propCmd.Flags().String("n", "data/net1.txt,data/net2.txt", "three columns network file(s)")
	//propCmd.Flags().String("p", "", "addtional prior file, use together with addPrior flag")
	propCmd.Flags().String("res", "result", "result folder")
	propCmd.Flags().Int("t", 4, "number of threads")
	propCmd.Flags().String("tsGene", "data/tsMatrix.txt", "additional genes to propagate values.")
	propCmd.Flags().String("trY", "data/trMatrix.txt", "train label matrix")
}

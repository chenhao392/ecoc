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
	"os"
)

// reportCmd represents the report command
var reportCmd = &cobra.Command{
	Use:   "report",
	Short: "calculate benchmark scores",
	Long: `

  ______ _____ ____   _____   _____  ______ _____   ____  _____ _______ 
 |  ____/ ____/ __ \ / ____| |  __ \|  ____|  __ \ / __ \|  __ \__   __|
 | |__ | |   | |  | | |      | |__) | |__  | |__) | |  | | |__) | | |   
 |  __|| |   | |  | | |      |  _  /|  __| |  ___/| |  | |  _  /  | |   
 | |___| |___| |__| | |____  | | \ \| |____| |    | |__| | | \ \  | |   
 |______\_____\____/ \_____| |_|  \_\______|_|     \____/|_|  \_\ |_|   
                                                                           
Calculate per label benchmark scores.
  Sample usages:
  ecoc report --tsY tsMatrix.txt --i pred.matrix.txt`,
	Run: func(cmd *cobra.Command, args []string) {
		if !cmd.Flags().Changed("tsY") {
			cmd.Help()
			os.Exit(0)
		}
		tsY, _ := cmd.Flags().GetString("tsY")
		tsYh, _ := cmd.Flags().GetString("i")
		rankCut, _ := cmd.Flags().GetInt("r")
		//thresFile, _ := cmd.Flags().GetString("s")

		tsYdata, _, _, _ := src.ReadFile(tsY, true, true)
		tsYhat, _, _, _ := src.ReadFile(tsYh, false, false)
		//thresData, _, _, _ := src.ReadFile(thresFile, false, false)
		_, nLabel := tsYhat.Caps()
		thresData := mat64.NewDense(1, nLabel, nil)
		for i := 0; i < nLabel; i++ {
			thresData.Set(0, i, 0.5)
		}
		detectNanInf := src.NanFilter(tsYhat)
		accuracy, microF1, microAupr, macroAupr, agMicroAupr, _, macroAuprSet := src.Report(1, tsYdata, tsYhat, thresData, rankCut, true)
		if detectNanInf {
			fmt.Println("NaN or Inf found.")
		}
		fmt.Printf("acc: %1.3f microF1: %1.3f microAupr: %1.3f macroAupr: %1.3f agMicroF1: %1.3f firstAupr: %1.3f\n", accuracy, microF1, microAupr, macroAupr, agMicroAupr, macroAuprSet[0])
	},
}

func init() {
	rootCmd.AddCommand(reportCmd)
	reportCmd.Flags().String("tsY", "data/tsMatrix.txt", "true testing data")
	reportCmd.Flags().String("i", "", "predictions")
	reportCmd.Flags().String("s", "", "thresholds")
	reportCmd.Flags().Int("r", 3, "predictions")
}

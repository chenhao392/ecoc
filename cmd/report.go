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
)

// reportCmd represents the report command
var reportCmd = &cobra.Command{
	Use:   "report",
	Short: "Calculate per label benchmarks",
	Long:  `Calculate per label benchmarks`,
	Run: func(cmd *cobra.Command, args []string) {
		tsY, _ := cmd.Flags().GetString("tsY")
		tsYh, _ := cmd.Flags().GetString("i")
		rankCut, _ := cmd.Flags().GetInt("r")

		tsYdata, _, _, _ := src.ReadFile(tsY, true, true)
		tsYhat, _, _, _ := src.ReadFile(tsYh, false, false)
		tsYhat = src.ColScale(tsYhat)
		microF1, accuracy, macroAupr, microAupr := report(tsYdata, tsYhat, rankCut)
		fmt.Println(microF1, accuracy, macroAupr, microAupr)

	},
}

func init() {
	rootCmd.AddCommand(reportCmd)
	reportCmd.PersistentFlags().String("tsY", "data/human.bp.level1.set1.tsMatrix.txt", "true testing data")
	reportCmd.PersistentFlags().String("i", "", "predictions")
	reportCmd.PersistentFlags().Int("r", 3, "predictions")
}

func report(tsYdata *mat64.Dense, tsYhat *mat64.Dense, rankCut int) (microF1 float64, accuracy float64, macroAupr float64, microAupr float64) {
	//F1 score
	_, nLabel := tsYdata.Caps()
	sumAupr := 0.0
	sumF1 := 0.0
	sumTp := 0
	sumFp := 0
	sumFn := 0
	sumTn := 0
	macroAuprSet := make([]float64, 0)
	accuracySet := make([]float64, 0)
	microF1Set := make([]float64, 0)
	for i := 0; i < nLabel; i++ {
		aupr := src.ComputeAupr(tsYdata.ColView(i), tsYhat.ColView(i))
		macroAuprSet = append(macroAuprSet, aupr)
		sumAupr += aupr
		//os.Exit(0)
	}
	macroAupr = sumAupr / float64(nLabel)
	tsYhat = src.BinPredByAlpha(tsYhat, rankCut)
	//y-flat
	tsYdataVec := src.Flat(tsYdata)
	tsYhatVec := src.Flat(tsYhat)
	microAupr = src.ComputeAupr(tsYdataVec, tsYhatVec)

	for i := 0; i < nLabel; i++ {
		f1, tp, fp, fn, tn := src.ComputeF1_3(tsYdata.ColView(i), tsYhat.ColView(i), rankCut)
		fmt.Println(i, tp, fp, fn, tn)
		microF1Set = append(microF1Set, f1)
		accuracySet = append(accuracySet, (float64(tp)+float64(tn))/(float64(tp)+float64(fp)+float64(fn)+float64(tn)))
		sumF1 += f1
		sumTp += tp
		sumFp += fp
		sumFn += fn
		sumTn += tn
	}
	p := float64(sumTp) / (float64(sumTp) + float64(sumFp))
	r := float64(sumTp) / (float64(sumTp) + float64(sumFn))
	microF1 = 2.0 * p * r / (p + r)
	accuracy = (float64(sumTp) + float64(sumTn)) / (float64(sumTp) + float64(sumFp) + float64(sumFn) + float64(sumTn))
	for i := 0; i < nLabel; i++ {
		fmt.Printf("%d\t%.3f\t%.3f\t%.3f\n", i, accuracySet[i], microF1Set[i], macroAuprSet[i])
	}

	return microF1, accuracy, macroAupr, microAupr
}

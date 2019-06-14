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
		rebaFile, _ := cmd.Flags().GetString("s")

		tsYdata, _, _, _ := src.ReadFile(tsY, true, true)
		tsYhat, _, _, _ := src.ReadFile(tsYh, false, false)
		rebaData, _, _, _ := src.ReadFile(rebaFile, false, false)
		microF1, accuracy, macroAupr, microAupr := src.Report(tsYdata, tsYhat, rebaData, rankCut, true)
		fmt.Println(microF1, accuracy, macroAupr, microAupr)

	},
}

func init() {
	rootCmd.AddCommand(reportCmd)
	reportCmd.PersistentFlags().String("tsY", "data/human.bp.level1.set1.tsMatrix.txt", "true testing data")
	reportCmd.PersistentFlags().String("i", "", "predictions")
	reportCmd.PersistentFlags().String("s", "", "rebalance")
	reportCmd.PersistentFlags().Int("r", 3, "predictions")
}

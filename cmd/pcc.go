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
	"os"
)

// pccCmd represents the pcc command
var pccCmd = &cobra.Command{
	Use:   "pcc",
	Short: "Fast Pearson correlation coefficient calculation",
	Long: `
  ______ _____ ____   _____   _____   _____ _____ 
 |  ____/ ____/ __ \ / ____| |  __ \ / ____/ ____|
 | |__ | |   | |  | | |      | |__) | |   | |     
 |  __|| |   | |  | | |      |  ___/| |   | |     
 | |___| |___| |__| | |____  | |    | |___| |____ 
 |______\_____\____/ \_____| |_|     \_____\_____|
		                                                    
This command calculate pairwise PCC between instances, such as 
genes, given a gene by feature matrix, where each row starts with
gene ID and each column is one feature, without column headers.

 Sample usages:
 ecoc pcc --i trMatrix.txt --o pcc.txt`,

	Run: func(cmd *cobra.Command, args []string) {
		if !cmd.Flags().Changed("i") {
			cmd.Help()
			os.Exit(0)
		}
		inFile, _ := cmd.Flags().GetString("i")
		outFile, _ := cmd.Flags().GetString("0")
		threads, _ := cmd.Flags().GetInt("t")
		isColumn, _ := cmd.Flags().GetBool("c")
		data := mat64.NewDense(0, 0, nil)
		name := make([]string, 0)
		if isColumn {
			data, _, name, _ = src.ReadFile(inFile, false, true)
			data = mat64.DenseCopyOf(data.T())
		} else {
			data, name, _, _ = src.ReadFile(inFile, true, false)
		}
		data2, _ := src.ParaCov(data, threads)
		//_, nCol := data2.Caps()
		src.WriteFile(outFile, data2, name, true)
		//for i := range name {
		//	fmt.Printf("%v", name[i])
		//	for j := 0; j < nCol; j++ {
		//		fmt.Printf("\t%1.6f", data2.At(i, j))
		//	}
		//	fmt.Printf("\n")
		//}
	},
}

func init() {
	rootCmd.AddCommand(pccCmd)
	pccCmd.PersistentFlags().String("i", "", "input tab delimited matrix")
	pccCmd.PersistentFlags().String("o", "", "output tab delimited matrix")
	pccCmd.PersistentFlags().Int("t", 1, "number of threads")
	pccCmd.PersistentFlags().Bool("c", false, "column pairs calculation (rowise as default)")
}

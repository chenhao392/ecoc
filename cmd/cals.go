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
	"math"
	"runtime"
	"sync"

	"github.com/chenhao392/ecoc/src"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/spf13/cobra"
)

type fm struct {
	mat64.Matrix
}

type coor struct {
	i [100]int
	j [100]int
}

func (f *coor) SetI(i [100]int) {
	f.i = i
}

func (f coor) I() [100]int {
	return f.i
}

func (f *coor) SetJ(j [100]int) {
	f.j = j
}

func (f coor) J() [100]int {
	return f.j
}

// calsCmd represents the cals command
var calsCmd = &cobra.Command{
	Use:   "cals",
	Short: "Fast Pearson correlation coefficient calculation",
	Long: `This command calculate pairwise PCC between genes, given
a gene by feature matrix.`,

	Run: func(cmd *cobra.Command, args []string) {
		inFile, _ := cmd.Flags().GetString("i")
		threads, _ := cmd.Flags().GetInt("t")
		data, rName, _, _ := src.ReadFile(inFile, true, false)
		data2, _ := paraCov(data, threads)
		_, nCol := data2.Caps()
		for i := range rName {
			fmt.Printf("%v", rName[i])
			for j := 0; j < nCol; j++ {
				fmt.Printf("\t%1.6f", data2.At(i, j))
			}
			fmt.Printf("\n")
		}
	},
}

func init() {
	rootCmd.AddCommand(calsCmd)
	calsCmd.PersistentFlags().String("i", "", "tab delimited matrix")
	calsCmd.PersistentFlags().Int("t", 1, "number of threads")

}

//Multiple threads PCC
func paraCov(data *mat64.Dense, goro int) (covmat *mat64.Dense, err error) {
	nSets, nData := data.Dims()
	if nSets == 0 {
	}
	runtime.GOMAXPROCS(goro)
	c := make([]coor, 1)

	element := coor{}
	var iArr [100]int
	var jArr [100]int
	k := 0

	for i := 0; i < nSets; i++ {
		for j := i; j < nSets; j++ {
			if k <= 99 {
				iArr[k] = i
				jArr[k] = j
			} else {
				element.SetI(iArr)
				element.SetJ(jArr)
				c = append(c, element)
				element = coor{}
				k = 0
				iArr[k] = i
				jArr[k] = j
			}
			k++
		}
	}
	//last coor
	element.SetI(iArr)
	element.SetJ(jArr)
	c = append(c, element)

	//pcc matrix, mean and var sqrt
	covmat = mat64.NewDense(nSets, nSets, nil)
	means := make([]float64, nSets)
	vs := make([]float64, nSets)

	for i := range means {
		means[i] = floats.Sum(data.RawRowView(i)) / float64(nData)
		var element float64
		for j, _ := range data.RawRowView(i) {
			data.Set(i, j, data.At(i, j)-means[i])
			element += data.At(i, j) * data.At(i, j)
		}
		vs[i] = math.Sqrt(element)
	}

	var wg sync.WaitGroup
	in := make(chan coor, goro*40)

	singlePCC := func() {
		for {
			select {
			case element := <-in:

				iArr := element.I()
				jArr := element.J()

				for m := 0; m < len(iArr); m++ {
					i := iArr[m]
					j := jArr[m]
					var cv float64
					for k, val := range data.RawRowView(i) {
						cv += data.At(j, k) * val
					}

					cv = cv / (vs[i] * vs[j])
					if (i == 0 && j == 0 && covmat.At(0, 0) == 0.0) || (i+j) > 0 {
						covmat.Set(i, j, cv)
						covmat.Set(j, i, cv)
					}
				}
				wg.Done()
			}
		}
	}

	wg.Add(len(c))
	for i := 0; i < goro; i++ {
		go singlePCC()
	}
	for i := 0; i < len(c); i++ {
		in <- c[i]
	}
	wg.Wait()

	return covmat, nil
}

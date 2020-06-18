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
	"os"
)

// netenCmd represents the neten command
var netenCmd = &cobra.Command{
	Use:   "neten",
	Short: "network enhancement",
	Long: `

  ______ _____ ____   _____   _   _ ______ _______ ______ _   _  
 |  ____/ ____/ __ \ / ____| | \ | |  ____|__   __|  ____| \ | |
 | |__ | |   | |  | | |      |  \| | |__     | |  | |__  |  \| |
 |  __|| |   | |  | | |      | . \ |  __|    | |  |  __| | . \ |
 | |___| |___| |__| | |____  | |\  | |____   | |  | |____| |\  |
 |______\_____\____/ \_____| |_| \_|______|  |_|  |______|_| \_|
                                                                   
network enhancement for a network.
  Sample usage:
  ecoc neten --i net1.txt --o net1enhanced.txt`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			cmd.Help()
			os.Exit(0)
		}
		rawNetworkFile, _ := cmd.Flags().GetString("i")
		enNetworkFile, _ := cmd.Flags().GetString("o")
		network, _, idxToId := src.ReadNetwork(rawNetworkFile)
		network = src.NetworkEnhance(network)
		src.WriteNetwork(enNetworkFile, network, idxToId)
	},
}

func init() {
	rootCmd.AddCommand(netenCmd)

	netenCmd.PersistentFlags().String("i", "", "raw network file ")
	netenCmd.PersistentFlags().String("o", "neten.txt", "enhanced network file")

}

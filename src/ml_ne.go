package src

import (
	"github.com/gonum/matrix/mat64"
	"log"
	"math"
	"math/cmplx"
	"sort"
)

func NetworkEnhance(network *mat64.Dense) (networkEnhanced *mat64.Dense) {
	nRow, nCol := network.Caps()
	k := int(math.Min(20, float64(nRow/10)))
	alpha := 0.9
	//order := 2
	eps := math.Nextafter(1.0, 2.0) - 1.0
	//empty diag
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if i == j {
				network.Set(i, j, 0.0)
			}
		}
	}
	//diag as abs colSums
	nRow, nCol = network.Caps()
	diagSum := mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < nCol; j++ {
		sum := 0.0
		for i := 0; i < nRow; i++ {
			sum += math.Abs(network.At(i, j))
		}
		diagSum.Set(j, j, sum)
	}
	network = aveDN(network, eps)
	network = symAve(network)
	//kNN of the net
	network = knnNet(network, k)
	//transition
	network = transitionFields(network, eps)
	var eigNet mat64.Eigen
	var uNet *mat64.Dense
	//init Eigen
	ok := eigNet.Factorize(network, false, true)
	if !ok {
		log.Fatal("left factorization for network failed!")
	}
	uNet = eigNet.Vectors()
	eigValues := eigNet.Values(nil)
	diagM := getDiagFromEigValues(eigValues, eps, alpha)
	term1 := mat64.NewDense(0, 0, nil)
	term1.Mul(uNet, diagM)
	network.Mul(term1, uNet.T())
	//update network
	nRow, nCol = network.Caps()
	d := mat64.NewDense(nRow, 1, nil)
	for i := 0; i < nCol; i++ {
		d.Set(i, 0, 1.0-network.At(i, i))
		network.Set(i, i, 0.0)
	}
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			network.Set(i, j, network.At(i, j)/d.At(i, 0))
		}
	}
	networkEnhanced = mat64.NewDense(0, 0, nil)
	networkEnhanced.Mul(diagSum, network)
	networkEnhanced = symAve(networkEnhanced)
	return networkEnhanced
}

func getDiagFromEigValues(eigValues []complex128, eps float64, alpha float64) (diagM *mat64.Dense) {
	n := len(eigValues)
	diagM = mat64.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		value := cmplx.Abs(eigValues[i]) - eps
		value = (1 - alpha) * value / (1 - alpha*value*value)
		diagM.Set(i, i, value)
	}
	return diagM
}

func transitionFields(network *mat64.Dense, eps float64) (network2 *mat64.Dense) {
	nRow, nCol := network.Caps()
	w := mat64.NewDense(1, nCol, nil)
	network2 = mat64.NewDense(0, 0, nil)
	//update diag by adding (rowSum+1)
	for i := 0; i < nRow; i++ {
		sum := 0.0
		for j := 0; j < nCol; j++ {
			sum += math.Abs(network.At(i, j))
		}
		network.Set(i, i, sum+1.0)
		//time length
		for j := 0; j < nCol; j++ {
			network.Set(i, j, network.At(i, j)*float64(nRow))
		}
	}
	network = aveDN(network, eps)
	//update w
	for j := 0; j < nCol; j++ {
		sum := 0.0
		for i := 0; i < nRow; i++ {
			sum += math.Abs(network.At(i, j))
		}
		w.Set(0, j, math.Sqrt(sum+eps))
	}
	//update network
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			network.Set(i, j, network.At(i, j)/w.At(0, j))
		}
	}
	network2.Mul(network, network.T())
	return network2
}

func knnNet(network *mat64.Dense, nKNN int) (network2 *mat64.Dense) {
	//abs ele
	nRow, nCol := network.Caps()
	network2 = mat64.NewDense(nRow, nCol, nil)
	for i := 0; i < nRow; i++ {
		var sortMap []kv
		for j := 0; j < nCol; j++ {
			sortMap = append(sortMap, kv{j, math.Abs(network.At(i, j))})
		}
		sort.Slice(sortMap, func(m, n int) bool {
			return sortMap[m].Value > sortMap[n].Value
		})
		thres := sortMap[nKNN].Value

		for j := 0; j < nCol; j++ {
			//so that abs value set, rather than time sign in return
			value := math.Abs(network.At(i, j))
			if value > thres {
				network2.Set(i, j, value)
			}
		}
	}
	network2 = symAve(network2)
	return network2
}

func aveDN(network *mat64.Dense, eps float64) (network2 *mat64.Dense) {
	nRow, nCol := network.Caps()
	d := mat64.NewDense(nRow, nCol, nil)
	for j := 0; j < nCol; j++ {
		sum := 0.0
		for i := 0; i < nRow; i++ {
			network.Set(i, j, network.At(i, j)*float64(nRow))
			sum += network.At(i, j)
		}
		d.Set(j, j, 1.0/(sum+eps))
	}
	network2 = mat64.NewDense(0, 0, nil)
	network2.Mul(d, network)
	return network2
}

func symAve(network *mat64.Dense) (network2 *mat64.Dense) {
	nRow, nCol := network.Caps()
	network2 = mat64.NewDense(nRow, nCol, nil)

	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			network2.Set(i, j, (network.At(i, j)+network.At(j, i))/2)
		}
	}
	return network2
}

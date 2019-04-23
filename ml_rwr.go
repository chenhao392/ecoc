package main

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

func colNorm(network *mat64.Dense) (normNet *mat64.Dense, n int) {
	n, _ = network.Caps()
	normNet = mat64.NewDense(n, n, nil)
	for j := 0; j < n; j++ {
		s := mat64.Sum(network.ColView(j))
		if s > 0.0 {
			for i := 0; i < n; i++ {
				normNet.Set(i, j, network.At(i, j)/s)
			}
		}
	}
	return normNet, n
}

func propagate(network *mat64.Dense, alpha float64, inPrior *mat64.Dense) (smoothPrior *mat64.Dense) {
	sum := mat64.Sum(inPrior)
	r, _ := inPrior.Caps()
	restart := mat64.NewDense(r, 1, nil)
	prior := mat64.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		restart.Set(i, 0, inPrior.At(i, 0)/sum)
		prior.Set(i, 0, inPrior.At(i, 0)/sum)
	}
	thres := 0.00001
	maxIter := 1000
	i := 0
	res := 1.0
	for res > thres && i < maxIter {
		prePrior := mat64.DenseCopyOf(prior)
		term1 := mat64.NewDense(0, 0, nil)
		term1.Mul(network, prior)
		for i := 0; i < r; i++ {
			prior.Set(i, 0, alpha*term1.At(i, 0)+(1-alpha)*restart.At(i, 0))
		}
		res = 0.0
		for i := 0; i < r; i++ {
			res += math.Abs(prior.At(i, 0) - prePrior.At(i, 0))
		}
		i += 1
	}
	return prior
}

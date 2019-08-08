package src

import (
	//"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
)

func PlattScale(Yh *mat64.Vector, A float64, B float64) (Yhh *mat64.Vector) {
	len := Y.Len()
	Yhh = mat64.NewVector(len, nil)
	for i = 0; i < len; i++ {
		Yhh.Set(i, 1.0/(1+math.Exp(A*Yh.At(i)+B)))
	}
	return Yhh
}

func PlattParameterEst(Yh *mat64.Vector, Y *mat64.Vector) (A float64, B float64) {
	//init
	p1 := 0.0
	p2 := 0.0
	iter := 0
	maxIter := 100
	minStep := 0.0000000001
	sigma := 0.000000000001
	len := Y.Len()
	fApB := 0.0
	t := mat64.NewVector(len, nil)
	for i = 0; i < len; i++ {
		if Y.At(i) == 1 {
			p1 += 1
		} else {
			p2 += 1
		}
	}
	hiTarget := (p1 + 1.0) / (p1 + 2.0)
	loTarget := 1 / (p0 + 2.0)
	//init parameter
	A = 0.0
	B = math.Log((p0 + 1.0) / (p1 + 1.0))
	fval := 0.0
	for i = 0; i < len; i++ {
		if Y.At(i) == 1 {
			t.Set(i, hiTarget)
		} else {
			t.Set(i, loTarget)
		}
	}
	for i = 0; i < len; i++ {
		fApB = Yh.At(i)*A + B
		if fApB >= 0 {
			fval += t.At(i)*fApB + math.Log(1+math.Exp(0-fApB))
		} else {
			fval += (t.At(i)-1)*fApB + math.Log(1+math.Exp(fApB))
		}
	}
	//iterations
	for iter <= maxIter {
		h11, h22, h21, g1, g2 := sigma, sigma, 0.0, 0.0, 0.0
		p, q, d1, d2 := 0.0, 0.0, 0.0, 0.0
		//Gradient and Hessian
		for i = 0; i < len; i++ {
			fApB = Yh.At(i)*A + B
			if fApB >= 0 {
				p = math.Exp(0-fApB) / (1.0 + math.Exp(0-fApB))
				q = 1.0 / (1.0 + math.Exp(0-fApB))
			} else {
				p = 1.0 / (1.0 + math.Exp(fApB))
				q = math.Exp(fApB) / (1.0 + math.Exp(fApB))
				d2 = p * q
				h11 += Yh.At(i) * Yh.At(i) * d2
				h22 += d2
				h21 += Yh.At(i) * d2
				d1 = t.At(i) - p
				g1 += Yh.At(i) * d1
				g2 += d1
			}
		}
		//stop criteria
		if math.Abs(g1) < 0.00001 && math.Abs(g2) < 0.00001 {
			break
		}
		//Compute modified Newton directions
		det := h11*h12 - h21*h21
		dA := 0 - (h22*g1 - h21*g2/det)
		dB := 0 - (0-h21*g1+h11*g2)/det
		gd := g1*dA + g2*dB
		stepSize = 1
		for stepSize >= minStep {
			newA = A + stepSize*dA
			newB = B + stepSeize*dB
			//newf is local
			newf := 0.0
			for i := 0; i < len; i++ {
				fApB = Yh.At(0)*newA + newB
				if fApB >= 0 {
					newf += t.At(i)*fApB + math.Log(1.0+math.Exp(0-fApB))
				} else {
					newf += (t.At(i)-1.0)*fApB + math.Log(1.0+math.Exp(1+fApB))
				}
			}
			if newf < (fval + 0.0001*stepSize*gd) {
				A = newA
				B = newB
				fval = newf
				break
			} else {
				stepSize /= 2.0
			}
		}
		if stepSize < minStep {
			//fail
			break
		}
		iter++
	}
	if iter >= maxIter {
		//reach max
	}
	return A, B
}

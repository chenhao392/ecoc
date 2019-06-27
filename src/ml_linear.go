package src

/*
#cgo LDFLAGS: -llinear
#include <linear.h>
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"
*/
import "C"
import "unsafe"
import (
	//"errors"
	//"fmt"
	"github.com/gonum/matrix/mat64"
)

// Model contains a pointer to C's struct model (i.e., `*C.struct_model`). It is
// returned after training and used for predicting.
type Model struct {
	// struct model
	// {
	// 	struct parameter param;
	// 	int nr_class;		/* number of classes */
	// 	int nr_feature;
	// 	double *w;
	// 	int *label;		/* label of each class */
	// 	double bias;
	// };
	cModel *C.struct_model
}

func (f *Model) W() []float64 {
	w := doubleToFloats(f.cModel.w, int(f.cModel.nr_feature)+1)
	return w
}
func (f *Model) Label() int {
	label := int(C.int(*f.cModel.label))
	return label
}

// Wrapper for the `train` function in liblinear.
//
// `model* train(const struct problem *prob, const struct parameter *param);`
//
// The explanation of parameters are:
//
// solverType:
//
//   for multi-class classification
//          0 -- L2-regularized logistic regression (primal)
//          1 -- L2-regularized L2-loss support vector classification (dual)
//          2 -- L2-regularized L2-loss support vector classification (primal)
//          3 -- L2-regularized L1-loss support vector classification (dual)
//          4 -- support vector classification by Crammer and Singer
//          5 -- L1-regularized L2-loss support vector classification
//          6 -- L1-regularized logistic regression
//          7 -- L2-regularized logistic regression (dual)
//   for regression
//         11 -- L2-regularized L2-loss support vector regression (primal)
//         12 -- L2-regularized L2-loss support vector regression (dual)
//         13 -- L2-regularized L1-loss support vector regression (dual)
//
// eps is the stopping criterion.
//
// C_ is the cost of constraints violation.
//
// p is the sensitiveness of loss of support vector regression.
//
// classWeights is a map from int to float64, with the key be the class and the
// value be the weight. For example, {1: 10, -1: 0.5} means giving weight=10 for
// class=1 while weight=0.5 for class=-1
//
// If you do not want to change penalty for any of the classes, just set
// classWeights to nil.
func Train(X, y *mat64.Dense, bias float64, solverType int, c_, p, eps float64, classWeights map[int]float64) *Model {
	var weightLabelPtr *C.int
	var weightPtr *C.double

	nRows, nCols := X.Dims()

	cX := mapCDouble(X.RawMatrix().Data)
	cY := mapCDouble(y.ColView(0).RawVector().Data)

	nrWeight := len(classWeights)
	weightLabel := []C.int{}
	weight := []C.double{}

	for key, val := range classWeights {
		weightLabel = append(weightLabel, (C.int)(key))
		weight = append(weight, (C.double)(val))
	}

	if nrWeight > 0 {
		weightLabelPtr = &weightLabel[0]
		weightPtr = &weight[0]
	} else {
		weightLabelPtr = nil
		weightPtr = nil
	}

	model := C.call_train(
		&cX[0], &cY[0],
		C.int(nRows), C.int(nCols), C.double(bias),
		C.int(solverType), C.double(c_), C.double(p), C.double(eps),
		C.int(nrWeight), weightLabelPtr, weightPtr)

	return &Model{
		cModel: model,
	}
}

// convert C double pointer to float64 slice ...
func doubleToFloats(in *C.double, size int) []float64 {
	outD := (*[1 << 30]C.double)(unsafe.Pointer(in))[:size:size]
	defer C.free(unsafe.Pointer(in))
	out := make([]float64, size, size)
	for i := 0; i < size; i++ {
		out[i] = float64(outD[i])
	}
	return out
}

func mapCDouble(in []float64) []C.double {
	out := make([]C.double, len(in), len(in))
	for i, val := range in {
		out[i] = C.double(val)
	}
	return out
}

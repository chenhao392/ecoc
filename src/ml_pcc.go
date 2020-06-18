package src

import (
	"math"
	"runtime"
	"sync"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
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

//Multiple threads PCC
func ParaCov(data *mat64.Dense, goro int) (covmat *mat64.Dense, err error) {
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

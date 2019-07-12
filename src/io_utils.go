package src

import (
	"bufio"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"os"
	"strconv"
	"strings"
)

func ReadNetwork(inNetworkFile string) (network *mat64.Dense, idIdx map[string]int, idxToId map[int]string) {
	idIdx, idxToId, nID := idIdxGen(inNetworkFile)
	network = mat64.NewDense(nID, nID, nil)
	//file
	file, err := os.Open(inNetworkFile)
	if err != nil {
		return
	}
	//load
	br := bufio.NewReaderSize(file, 32768000)
	for {
		line, isPrefix, err := br.ReadLine()
		if err != nil {
			break
		}
		if isPrefix {
			return
		}

		str := string(line)
		elements := strings.Split(str, "\t")
		a := idIdx[elements[0]]
		b := idIdx[elements[1]]
		v, _ := strconv.ParseFloat(elements[2], 64)
		network.Set(a, b, v)
		network.Set(b, a, v)
	}
	return network, idIdx, idxToId
}
func idIdxGen(inNetworkFile string) (idIdx map[string]int, idxToId map[int]string, count int) {
	idIdx = make(map[string]int)
	idxToId = make(map[int]string)
	//file
	file, err := os.Open(inNetworkFile)
	if err != nil {
		return
	}
	//load
	br := bufio.NewReaderSize(file, 32768000)
	count = 0
	for {
		line, isPrefix, err := br.ReadLine()
		if err != nil {
			break
		}
		if isPrefix {
			return
		}

		str := string(line)
		elements := strings.Split(str, "\t")
		for c, id := range elements {
			if c < 2 {
				_, exist := idIdx[id]
				if !exist {
					idIdx[id] = count
					idxToId[count] = id
					count += 1
				}
			}
		}
	}
	//count +1, so that it is number of unique IDs
	//count += 1
	return idIdx, idxToId, count
}

func ReadFile(inFile string, rowName bool, colName bool) (dataR *mat64.Dense, rName []string, cName []string, err error) {
	//init
	lc, cc, _ := lcCount(inFile)
	if rowName {
		cc -= 1
	}
	if colName {
		lc -= 1
	}
	data := mat64.NewDense(lc, cc, nil)
	rName = make([]string, 0)
	cName = make([]string, 0)

	//file
	file, err := os.Open(inFile)
	if err != nil {
		return
	}
	defer file.Close()

	//load
	br := bufio.NewReaderSize(file, 32768000)
	r := 0
	touchCol := false
	for {
		line, isPrefix, err1 := br.ReadLine()
		if err1 != nil {
			break
		}
		if isPrefix {
			return
		}

		str := string(line)
		elements := strings.Split(str, "\t")
		if rowName {
			value := Shift(&elements)
			rName = append(rName, value)
		}
		//first element already shifted if rowName is true
		if colName && !touchCol {
			cName = elements
			touchCol = true
		} else {
			for c, i := range elements {
				j, _ := strconv.ParseFloat(i, 64)
				data.Set(r, c, j)
			}
			r++
		}
	}
	//shfit first rowName away if colName exist
	if colName {
		Shift(&rName)
	}
	return data, rName, cName, nil
}

//line count(nRow) and column count(nCol) for a tab separeted txt
func lcCount(filename string) (lc int, cc int, err error) {
	lc = 0
	cc = 0
	touch := true

	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	//load
	br := bufio.NewReaderSize(file, 32768000)
	for {
		line, isPrefix, err1 := br.ReadLine()
		if err1 != nil {
			break
		}
		if isPrefix {
			return
		}

		if touch {
			cc = strings.Count(string(line), "\t")
			cc += 1
			touch = false
		}
		lc++
	}
	return lc, cc, nil
}

func WriteFile(outFile string, data *mat64.Dense) (err error) {
	file, err := os.OpenFile(outFile, os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer file.Close()
	file.Truncate(0)
	file.Seek(0, 0)
	wr := bufio.NewWriterSize(file, 192000)
	nRow, nCol := data.Caps()
	var ele string
	for i := 0; i < nRow; i++ {
		ele = strconv.FormatFloat(data.At(i, 0), 'f', 6, 64)
		wr.WriteString(ele)
		for j := 1; j < nCol; j++ {
			ele = strconv.FormatFloat(data.At(i, j), 'f', 6, 64)
			wr.WriteString("\t")
			wr.WriteString(ele)
		}
		wr.WriteString("\n")
	}
	wr.Flush()
	return err
}

func WriteNetwork(outFile string, data *mat64.Dense, idxToId map[int]string) (err error) {
	file, err := os.OpenFile(outFile, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer file.Close()
	file.Truncate(0)
	file.Seek(0, 0)
	wr := bufio.NewWriterSize(file, 192000)
	nRow, nCol := data.Caps()
	var ele string
	for i := 0; i < nRow; i++ {
		for j := i + 1; j < nCol; j++ {
			if data.At(i, j) > 0 {
				wr.WriteString(idxToId[i])
				wr.WriteString("\t")
				wr.WriteString(idxToId[j])
				wr.WriteString("\t")
				ele = strconv.FormatFloat(data.At(i, j), 'f', 6, 64)
				wr.WriteString(ele)
				wr.WriteString("\n")
			}
		}
	}
	wr.Flush()
	return err
}

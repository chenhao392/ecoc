package src

import (
	"bufio"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
)

func AccumIds(idIdx map[string]int, idxToId map[int]string, UpdatedIdIdx map[string]int, UpdatedIdxToId map[int]string) (map[string]int, map[int]string) {
	nEle := len(UpdatedIdIdx)
	for name, _ := range idIdx {
		_, exist := UpdatedIdIdx[name]
		if !exist {
			//nEle is zero based in matrix
			UpdatedIdIdx[name] = nEle
			UpdatedIdxToId[nEle] = name
			nEle += 1
		}
	}
	return UpdatedIdIdx, UpdatedIdxToId

}

func MeanNet(totalNet *mat64.Dense, countNet *mat64.Dense) *mat64.Dense {
	nRow, nCol := totalNet.Caps()
	meanNet := mat64.NewDense(nRow, nCol, nil)
	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			if countNet.At(i, j) > 0.0 {
				meanNet.Set(i, j, totalNet.At(i, j)/countNet.At(i, j))
			}
		}
	}
	return meanNet
}

func UpdateNetwork(inNetworkFile string, idIdx map[string]int, idxToId map[int]string, network *mat64.Dense) *mat64.Dense {
	//file
	file, err := os.Open(inNetworkFile)
	if err != nil {
		return nil
	}
	//load
	br := bufio.NewReaderSize(file, 32768000)
	for {
		line, isPrefix, err := br.ReadLine()
		if err != nil {
			break
		}
		if isPrefix {
			return nil
		}

		str := string(line)
		elements := strings.Split(str, "\t")
		a := idIdx[elements[0]]
		b := idIdx[elements[1]]
		v, _ := strconv.ParseFloat(elements[2], 64)
		network.Set(a, b, v)
		network.Set(b, a, v)

	}
	return network
}

func FillNetwork(inNetworkFile string, idIdx map[string]int, idxToId map[int]string, totalNet *mat64.Dense, countNet *mat64.Dense) (*mat64.Dense, *mat64.Dense) {
	//file
	file, err := os.Open(inNetworkFile)
	if err != nil {
		return nil, nil
	}
	//load
	br := bufio.NewReaderSize(file, 32768000)
	for {
		line, isPrefix, err := br.ReadLine()
		if err != nil {
			break
		}
		if isPrefix {
			return nil, nil
		}

		str := string(line)
		elements := strings.Split(str, "\t")
		a := idIdx[elements[0]]
		b := idIdx[elements[1]]
		v, _ := strconv.ParseFloat(elements[2], 64)
		totalNet.Set(a, b, totalNet.At(a, b)+v)
		totalNet.Set(b, a, totalNet.At(a, b)+v)

		countNet.Set(a, b, countNet.At(a, b)+1.0)
		countNet.Set(b, a, countNet.At(a, b)+1.0)
	}
	return totalNet, countNet
}

func ReadNetwork(inNetworkFile string) (network *mat64.Dense, idIdx map[string]int, idxToId map[int]string) {
	idIdx, idxToId, nID := IdIdxGen(inNetworkFile)
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
func IdIdxGen(inNetworkFile string) (idIdx map[string]int, idxToId map[int]string, count int) {
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

func ReadIDfile(inFile string) (rName []string) {
	rName = make([]string, 0)
	//file
	file, err := os.Open(inFile)
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
		str := string(line)
		elements := strings.Split(str, "\t")
		value := Shift(&elements)
		rName = append(rName, value)
	}
	return rName
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
	if colName && rowName {
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

func WriteFile(outFile string, data *mat64.Dense, name []string, isRowID bool) (err error) {
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
		if isRowID {
			wr.WriteString(name[i])
			wr.WriteString("\t")
		}
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

func WriteOutputFiles(isVerbose bool, resFolder string, trainMeasure *mat64.Dense, testMeasure *mat64.Dense, posLabelRls *mat64.Dense, negLabelRls *mat64.Dense, tsYhat *mat64.Dense, Yhat *mat64.Dense, YhatCalibrated *mat64.Dense, Ylabel *mat64.Dense) {
	oFile := "./" + resFolder + "/test.probMatrix.txt"
	WriteFile(oFile, tsYhat, nil, false)
	if isVerbose {
		oFile = "./" + resFolder + "/cvTraining.measure.txt"
		WriteFile(oFile, trainMeasure, nil, false)
		oFile = "./" + resFolder + "/cvTesting.measure.txt"
		WriteFile(oFile, testMeasure, nil, false)
		oFile = "./" + resFolder + "/posLabelRls.txt"
		WriteFile(oFile, posLabelRls, nil, false)
		oFile = "./" + resFolder + "/negLabelRls.txt"
		WriteFile(oFile, negLabelRls, nil, false)
		oFile = "./" + resFolder + "/train.probMatrix.txt"
		WriteFile(oFile, Yhat, nil, false)
		oFile = "./" + resFolder + "/trainCalibrated.probMatrix.txt"
		WriteFile(oFile, YhatCalibrated, nil, false)
		oFile = "./" + resFolder + "/reorder.trMatrix.txt"
		WriteFile(oFile, Ylabel, nil, false)

		//mem profile
		memprofile := resFolder + "/mem.prof"
		f, err2 := os.Create(memprofile)
		if err2 != nil {
			log.Fatal("could not create memory profile: ", err2)
		}
		defer f.Close()
		runtime.GC() // get up-to-date statistics
		if err2 := pprof.WriteHeapProfile(f); err2 != nil {
			log.Fatal("could not write memory profile: ", err2)
		}
		defer f.Close()
	}

}

func Init(resFolder string) (logFIle *os.File) {
	err := os.MkdirAll("./"+resFolder, 0755)
	if err != nil {
		fmt.Println(err)
		return
	}
	logFile, err := os.OpenFile("./"+resFolder+"/log.txt", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	return logFile
}

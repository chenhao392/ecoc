package main

import (
	"bufio"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"os"
	"strconv"
	"strings"
)

func readFile(inFile string, rowName bool) (dataR *mat64.Dense, rName []string, err error) {
	//init
	lc, cc, _ := lcCount(inFile)
	if rowName {
		cc -= 1
	}
	data := mat64.NewDense(lc, cc, nil)
	rName = make([]string, 0)

	//file
	file, err := os.Open(inFile)
	if err != nil {
		return
	}
	defer file.Close()

	//load
	br := bufio.NewReaderSize(file, 32768000)
	r := 0
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
		for c, i := range elements {
			j, _ := strconv.ParseFloat(i, 64)
			data.Set(r, c, j)
		}
		r++
	}
	return data, rName, nil
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

func writeFile(outFile string, data *mat64.Dense) (err error) {
	file, err := os.OpenFile(outFile, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer file.Close()
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

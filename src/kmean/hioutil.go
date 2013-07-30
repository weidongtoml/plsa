// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hioutil implements higher level of I/O operations.
package kmean

import (
	"bufio"
	"os"
	"strings"
)

// Function WithOpenFileAsBufioReader attempt to open the file of
// given filename, and process the file by calling the given
// processor function with the reader object and param.
// When failed to open the given file, error will be returned,
// otherwise, the return value of processor will be returned.
func WithOpenFileAsBufioReader(filename string,
	processor func(*bufio.Reader) error) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer func() {
		file.Close()
	}()
	return processor(bufio.NewReader(file))
}

// Function ForEachLineInFile attempts to open the given file and
// process each line by invoking processor function with the line.
func ForEachLineInFile(filename string,
	processor func(line string) (bool, error)) error {
	fileProcessor := func(reader *bufio.Reader) error {
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				break
			}
			goOn, err := processor(strings.Trim(line, " \n"))
			if goOn == false {
				return err
			}
		}
		return nil
	}
	return WithOpenFileAsBufioReader(filename, fileProcessor)
}

// Function WithNewOpenFileAsBufioWriter attempts to create a
// new file of the given name and process it by calling the
// processer function with the writer object and param.
// If the file failed to create, error will be returned; else,
// the return value of the processor function will be returned.
func WithNewOpenFileAsBufioWriter(filename string,
	processor func(*bufio.Writer) error) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	defer func() {
		writer.Flush()
		file.Sync()
		file.Close()
	}()
	return processor(writer)
}

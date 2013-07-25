package plsa

import (
	"os"
	"bufio"
	"log"
	"strings"
	"strconv"
	"fmt"
	"errors"
)

type LineFieldExtractor func(string) (docId, word string, count uint64, err error)

func SimpleLineFieldExtractor (docWordSep, wordCountSep string) LineFieldExtractor {
	return func(line string)  (docId, word string, count uint64, err error){
		tokens := strings.SplitN(line, docWordSep, 1)
		if len(tokens) != 2 {
			err = errors.New(fmt.Sprintf("Cannot split [%s] to two fields using docWordSep[%s]", line, docWordSep))
			return
		}
		docId = tokens[0]
		n_tokens := strings.SplitN(tokens[1], wordCountSep, 1)
		if len(n_tokens) != 2 {
			err = errors.New(fmt.Sprintf("Cannot split [%s] to two fields using wordCountSep[%s]", tokens[1], wordCountSep))
			return
		}
		word = n_tokens[0]
		count, err = strconv.ParseUint(n_tokens[1], 10, 64)
		return
	}
}

type LineOrientedLoader struct {
	vocab     []string
	docIds    []string
	count     map[docIdWord]uint64
	extractor LineFieldExtractor
}

func NewLineOrientedLoader (extactor_func LineFieldExtractor) *LineOrientedLoader {
	var loader LineOrientedLoader
	loader.extractor = extactor_func
	return &loader
}

func (loader *LineOrientedLoader) LoadFromFile(docWordFreqFile string) bool {
	fd, err := os.Open(docWordFreqFile)
	if err != nil {
		log.Printf("LineOrientedLoader.LoadFromFile(%s) failed: %s", docWordFreqFile, err);
		return false;
	}
	
	reader := bufio.NewReader(fd)
	vocabMap := make(map[string]bool)
	docIdMap := make(map[string]bool)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		docId, word, count, err := loader.extractor(line)
		if err != nil {
			log.Printf("Failed to extract fields from line [%s]: %s", line, err)
			continue
		}
		
		if !docIdMap[docId] {
			(*loader).docIds = append((*loader).docIds, docId)
		}
		
		if !vocabMap[word] {
			(*loader).vocab = append((*loader).vocab, word)
		}
		
		docIdWordVal := docIdWord{docId, word}
		if countVal, found := (*loader).count[docIdWordVal]; found == true {
			log.Printf("Error, found duplicated definition of %v, old value is %v, new value is %v", 
				docIdWordVal, countVal, count);
		}
		(*loader).count[docIdWordVal] = count;
	}
	
	return true
}

func (loader *LineOrientedLoader) CorpusIds() []string {
	return (*loader).docIds
}

func (loader *LineOrientedLoader) CorpusSize() int {
	return len((*loader).docIds)
}

func (loader *LineOrientedLoader) Vocabulary() []string {
	return (*loader).vocab
}

func (loader *LineOrientedLoader) VocabularySize() int {
	return len((*loader).vocab)
}

func (loader *LineOrientedLoader) DocWordCount(docId, word string) uint64 {
	return (*loader).count[docIdWord{docId, word}]
}

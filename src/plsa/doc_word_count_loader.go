package plsa

import (

)

type LineOrientedLoader struct {
	vocab	[]string
	docIds	[]string
	count map[docIdWord]uint64	
}

func (loader *LineOrientedLoader) LoadFromFile(docWordFreqFile string) bool {
	return false
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

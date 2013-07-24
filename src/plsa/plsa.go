// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package plsa provides a set of methods to train,
// load and save a PLSA model to and from file, plus
// additional methods of applying the model such as
// retrieving the topic probability P(z), probability
// of a word generated from the given topic P(w|z),
// and the probability of a document generated from
// the given topic P(d|z).
//
// Please refer to the following for details of the PLSA model
// and related implementation details:
// Probabilistic Latent Semantic Analysis by Thomas Hofmann.
package plsa

import (
	"math/rand"
)

// DocWordFreqRetriever is the interface that wraps the basic
// methods for retriving document-word frequencies of the
// training corpus for the use of model training.
//
// LoadFromFile loads the document-word frequencies from the given file,
// and returns true on success, false otherwise.
//
// CorpusIds retrieves the entire list of document ids in the training
// corpus and stores the result in docIds.
//
// CorpusSize returns the number of documents in the training set.
//
// Vocabulary retrieves the entire list of words, i.e. the vocabulary in
// the training corpus and stores the result in words.
//
// VocabularySize returns the total number of different words in the training
// set.
//
// GetDocWordCount returns the number of occurrence of word in the document
// indexed by the given docId.
type DocWordFreqRetriever interface {
	LoadFromFile(docWordFreqFile string) bool
	CorpusIds(docIds []string)
	CorpusSize() int
	Vocabulary(words []string)
	VocabularySize() int
	DocWordCount(docId, word string) uint64
}

// Model holds the PLSA model data.
type Model struct {
	topicProb      []float32            //topic probability, P(z)
	docTopicProb  []map[string]float32 //document probability given topic, P(d|z)
	wordTopicProb []map[string]float32 //word probability given topic, P(w|z)
}

// SaveToFile saves the PLSA model to the given file.
func (model *Model) SaveToFile(filename string) {
}

// LoadModelFromFile loads a PLSA model from the given path.
func LoadModelFromFile(filename string) *Model {
	return nil
}

// NumberOfTopics returns the number of topics in the given PLSA model.
func (model *Model) NumberOfTopics() int {
	return len(model.topicProb)
}

// TopicProbability returns the probability of the given topic_id, if
// topic_id greater than or equal to the return value of NumberOfTopics,
// 0 will be returned to signify that the topic does not exist.
func (model *Model) TopicProbability(topicId int) float32 {
	if topicId < len(model.topicProb) {
		return model.topicProb[topicId]
	} else {
		return float32(0)
	}
}

// WordProbabilityGivenTopic returns the probability of the given word
// generated from topic with the given topic_id, if either the given word or topic_id
// is not in the model, 0 will be returned.
func (model *Model) WordProbabilityGivenTopic(word string, topicId int) float32 {
	if topicId < len(model.wordTopicProb) {
		return model.wordTopicProb[topicId][word]
	} else {
		return float32(0)
	}
}

// DocProbabilityGivenTopic returns the probability of the given document
// that is generated from topic with the given topic_id. 0 will be returned
// if either the document with the given id does not exists in the model or
// that the topicId is not in the model.
func (model *Model) DocProbabilityGivenTopic(docId string, topicId int) float32 {
	if topicId < len(model.docTopicProb) {
		return model.docTopicProb[topicId][docId]
	} else {
		return float32(0)
	}
}

// TrainingParameter holds the parameter for training a PLSA model.
type TrainingParameter struct {
	NumberOfTopics	int
}

type docIdWord struct {
	docId	string
	word	string
}

// TrainFromData trains a PLSA model from the given document word frequency
// data using the given training parameter..
func TrainFromData(docWordFreq DocWordFreqRetriever, parameter *TrainingParameter) *Model {
	numTopics := parameter.NumberOfTopics
	numDocs := docWordFreq.CorpusSize()
	numWords := docWordFreq.VocabularySize()
	var docIds []string
	var words []string
	docWordFreq.CorpusIds(docIds)
	docWordFreq.Vocabulary(words)
	
	var m Model
	m.topicProb = make([]float32, numTopics)
	for z, _ := range m.topicProb {
		m.topicProb[z] = float32(1)/float32(numTopics)
	}
	m.docTopicProb = make([]map[string]float32, numTopics)
	for z, _ := range m.docTopicProb {
		m.docTopicProb[z] = make(map[string]float32, numDocs)
		for _, d := range docIds {
			m.docTopicProb[z][d] = rand.Float32()
		}
	}
	m.wordTopicProb = make([]map[string]float32, numTopics)
	for z, _ := range m.wordTopicProb {
		m.wordTopicProb[z] = make(map[string]float32, numWords)
		for _, w := range words {
			m.wordTopicProb[z][w] = rand.Float32()
		}
	}
	
	// EM algorithm for training PLSA model.
	probZgivenDW := make([]map[docIdWord]float32, numTopics)
	for i, _ := range probZgivenDW {
		probZgivenDW[i] = make(map[docIdWord]float32, numDocs*numWords)
	}
	for {//TODO(weidoliang): add convergence test
		// E-step
		norm_constant := float32(0)
		for iter := 0; iter < 2; iter++ {
			for z := 0; z < numTopics; z++ {
				for _, w := range words {
					for _, d := range docIds {
						if iter < 1 {
							p := m.topicProb[z] * m.docTopicProb[z][d] * m.wordTopicProb[z][w]
							probZgivenDW[z][docIdWord{d, w}] = p
							norm_constant += p
						} else {
							probZgivenDW[z][docIdWord{d, w}] /= norm_constant
						}
					}
				}
			}
		}
		// M-step
		for z := 0; z < numTopics; z++ {
			for _, w := range words {
				p_w_z := float32(0)
				for _, d := range docIds {
					p_w_z += float32(docWordFreq.DocWordCount(d, w)) * probZgivenDW[z][docIdWord{d, w}]
				}
				m.wordTopicProb[z][w] = p_w_z
			}
			p_z := float32(0)
			for _, d := range words {
				p_d_z := float32(0)
				for _, w := range words {
					p_d_z += float32(docWordFreq.DocWordCount(d, w)) * probZgivenDW[z][docIdWord{d, w}]
				}
				m.docTopicProb[z][d] = p_d_z
				p_z += p_d_z
			}
			m.topicProb[z] = p_z
		}
	}
}

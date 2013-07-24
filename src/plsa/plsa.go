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
	"math"
	"math/rand"
	"log"
)

// DocWordFreqRetriever is the interface that wraps the basic
// methods for retriving document-word frequencies of the
// training corpus for the use of model training.
//
// LoadFromFile loads the document-word frequencies from the given file,
// and returns true on success, false otherwise.
//
// CorpusIds retrieves the entire list of document ids in the training
// corpus.
//
// CorpusSize returns the number of documents in the training set.
//
// Vocabulary retrieves the entire list of words, i.e. the vocabulary in
// the training corpus.
//
// VocabularySize returns the total number of different words in the training
// set.
//
// GetDocWordCount returns the number of occurrence of word in the document
// indexed by the given docId.
type DocWordFreqRetriever interface {
	LoadFromFile(docWordFreqFile string) bool
	CorpusIds() []string
	CorpusSize() int
	Vocabulary() []string
	VocabularySize() int
	DocWordCount(docId, word string) uint64
}

// Model holds the PLSA model data.
type Model struct {
	topicProb     []float32            //topic probability, P(z)
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
	NumberOfTopics     int     // Number of topics in the PLSA model.
	LikelihoodIncLimit float32 // Minimum likelihood increment reached in training before stopping.
	MaxIteration       int     //Maximum number of steps in the EM training procedure.
}

// TrainFromData trains a PLSA model from the given document word frequency
// data using the given training parameter..
func TrainFromData(docWordFreq DocWordFreqRetriever, param *TrainingParameter) *Model {
	var m Model
	// EM algorithm for training PLSA model.
	probZgivenDW := (&m).randomInit(docWordFreq, param)

	log.Printf("EM training begin: %v.\n", *param)
	prev_likelihood := float32(0)
	iter := 0
	for {
		(&m).eStep(docWordFreq, probZgivenDW)
		(&m).mStep(docWordFreq, probZgivenDW)
		
		likelihood := (&m).Likelihood(docWordFreq)
		likelihood_improvement := math.Abs(float64((likelihood-prev_likelihood)/prev_likelihood))
		
		log.Printf("Iteration: %d, likelihood: %f, improvement: %f\n", 
			iter, likelihood, likelihood_improvement)
		
		if likelihood_improvement < float64(param.LikelihoodIncLimit) {
			break
		} else {
			prev_likelihood = likelihood
		}
		if iter >= param.MaxIteration {
			break
		}
	}
	log.Printf("EM training end.\n")
	
	return &m
}

type docIdWord struct {
	docId string
	word  string
}

func (m *Model) randomInit(docWordFreq DocWordFreqRetriever, param *TrainingParameter) []map[docIdWord]float32 {
	numTopics := param.NumberOfTopics
	numDocs := docWordFreq.CorpusSize()
	numWords := docWordFreq.VocabularySize()
	docIds := docWordFreq.CorpusIds()
	words := docWordFreq.Vocabulary()
	
	(*m).topicProb = make([]float32, numTopics)
	for z, _ := range (*m).topicProb {
		(*m).topicProb[z] = float32(1) / float32(numTopics)
	}
	(*m).docTopicProb = make([]map[string]float32, numTopics)
	for z, _ := range (*m).docTopicProb {
		(*m).docTopicProb[z] = make(map[string]float32, numDocs)
		for _, d := range docIds {
			(*m).docTopicProb[z][d] = rand.Float32()
		}
	}
	(*m).wordTopicProb = make([]map[string]float32, numTopics)
	for z, _ := range (*m).wordTopicProb {
		(*m).wordTopicProb[z] = make(map[string]float32, numWords)
		for _, w := range words {
			(*m).wordTopicProb[z][w] = rand.Float32()
		}
	}

	probZgivenDW := make([]map[docIdWord]float32, numTopics)
	for i, _ := range probZgivenDW {
		probZgivenDW[i] = make(map[docIdWord]float32, numDocs*numWords)
	}
	return probZgivenDW
}

func (m *Model) eStep(docWordFreq DocWordFreqRetriever, probZgivenDW[]map[docIdWord]float32) {
	docIds := docWordFreq.CorpusIds()
	words := docWordFreq.Vocabulary()
	numTopics := m.NumberOfTopics()
	
	norm_constant := float32(0)
	for iter := 0; iter < 2; iter++ {
		for z := 0; z < numTopics; z++ {
			for _, w := range words {
				for _, d := range docIds {
					if iter < 1 {
						p := m.TopicProbability(z) * m.DocProbabilityGivenTopic(d, z) * m.WordProbabilityGivenTopic(w, z)
						probZgivenDW[z][docIdWord{d, w}] = p
						norm_constant += p
					} else {
						probZgivenDW[z][docIdWord{d, w}] /= norm_constant
					}
				}
			}
		}
	}
}

func (m *Model) mStep(docWordFreq DocWordFreqRetriever, probZgivenDW[]map[docIdWord]float32) {
	docIds := docWordFreq.CorpusIds()
	words := docWordFreq.Vocabulary()
	numTopics := m.NumberOfTopics()
	
	for z := 0; z < numTopics; z++ {
		for _, w := range words {
			p_w_z := float32(0)
			for _, d := range docIds {
				p_w_z += float32(docWordFreq.DocWordCount(d, w)) * probZgivenDW[z][docIdWord{d, w}]
			}
			(*m).wordTopicProb[z][w] = p_w_z
		}
		p_z := float32(0)
		for _, d := range words {
			p_d_z := float32(0)
			for _, w := range words {
				p_d_z += float32(docWordFreq.DocWordCount(d, w)) * probZgivenDW[z][docIdWord{d, w}]
			}
			(*m).docTopicProb[z][d] = p_d_z
			p_z += p_d_z
		}
		(*m).topicProb[z] = p_z
	}
}

// Likelihood computes the log likelihood of reconstruction of data from
// docWordFreq using the current model.
func (m *Model) Likelihood(docWordFreq DocWordFreqRetriever) float32 {
	docIds := docWordFreq.CorpusIds()
	words := docWordFreq.Vocabulary()
	numTopics := m.NumberOfTopics()

	likelihood := float64(0)
	for _, d := range docIds {
		for _, w := range words {
			count := docWordFreq.DocWordCount(d, w)
			if count > 0 {
				p_d_w := float32(0)
				for z := 0; z < numTopics; z++ {
					p_d_w += m.WordProbabilityGivenTopic(w, z) * m.DocProbabilityGivenTopic(d, z)
				}
				likelihood += float64(count) * math.Log(float64(p_d_w))
			}
		}
	}
	return float32(likelihood)
}

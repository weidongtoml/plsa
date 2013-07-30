// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plsa

import (
	"fmt"
	"math"
	"sort"
)

// WordFrequencyRetriever is an interface for retrieving single word probabilities and
// word co-occurence probabilities.
type WordFrequencyRetriever interface {
	WordProb(string) float64
	WordCooccurenceProb(string, string) float64
}

// PMIScorer is an object for calculating the PMI scores
type PMIScorer struct {
	WordFrequencyRetriever
}

// PMIScore returns the Pointwise Mutual Information Score of the given list of words.
func (s *PMIScorer) PMIScore(wordList []string) float64 {
	numWords := len(wordList)
	if numWords < 2 {
		panic(fmt.Sprintf("PMIScorer.PMIScore expects a slice having at least 2 elements but got %v", wordList))
	}
	var scores []float64
	for i, w1 := range wordList {
		for _, w2 := range wordList[i+1:] {
			scores = append(scores, s.PointwiseMutualInformation(w1, w2))
		}
	}
	sort.Float64s(scores)
	numScores := len(scores)
	if numScores%2 == 0 {
		return scores[numScores/2-1] + scores[numWords/2]
	} else {
		return scores[numScores/2]
	}
}

// PointwiseMutualInformation calculates the pointwise mutual information of word1 and word2.
func (s *PMIScorer) PointwiseMutualInformation(word1 string, word2 string) float64 {
	p := s.WordCooccurenceProb(word1, word2) / (s.WordProb(word1) * s.WordProb(word2))
	return math.Log(p)
}

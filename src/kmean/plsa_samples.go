// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kmean

import (
	"fmt"
	"log"
	"math"
	"sort"
	"strconv"
	"strings"
)

type PlsaSample struct {
	topicId  int
	repTerms map[string]float64
	norm     float64
}

func AssertAsPlsaSample(c SampleContainer) *PlsaSample {
	a, ok := c.(*PlsaSample)
	if !ok {
		panic(fmt.Sprintf("PlsaSample.Add(SampleContainer) expected parameter to be *PlsaSample but got %v", c))
	}
	return a
}

func (s *PlsaSample) Normalize() {
	totalP := float64(0)
	for _, p := range s.repTerms {
		totalP += p
	}
	for k, _ := range s.repTerms {
		s.repTerms[k] /= totalP
	}
	s.norm = 1.0
}

type termWeightT struct {
	term   string
	weight float64
}

type termWeightsT []*termWeightT

func (t termWeightsT) Len() int {
	return len(t)
}

func (t termWeightsT) Swap(i, j int) {
	t[i], t[j] = t[j], t[i]
}

type byName struct {
	termWeightsT
}

func (s byName) Less(i, j int) bool {
	return s.termWeightsT[i].term < s.termWeightsT[j].term
}

type byWeight struct {
	termWeightsT
}

func (s byWeight) Less(i, j int) bool {
	return s.termWeightsT[i].weight > s.termWeightsT[j].weight
}

func (s *PlsaSample) String() string {
	str := fmt.Sprintf("TopicId: %d, Terms: ", s.topicId)
	var r []*termWeightT
	for k, w := range s.repTerms {
		r = append(r, &termWeightT{k, w})
	}
	sort.Sort(byWeight{r})
	for _, v := range r {
		str += fmt.Sprintf(" %s(%f)", v.term, v.weight)
	}
	return str
}

func (s *PlsaSample) Id() int {
	return s.topicId
}

func (s *PlsaSample) Equals(c SampleContainer) bool {
	a := AssertAsPlsaSample(c)
	return s.topicId == a.topicId
}

func (s *PlsaSample) Zero() SampleContainer {
	return SampleContainer(&PlsaSample{0, make(map[string]float64), 0})
}

func (s *PlsaSample) Add(c SampleContainer) {
	a := AssertAsPlsaSample(c)
	for k, v := range a.repTerms {
		s.repTerms[k] += v
	}
}

func (s *PlsaSample) ScalarMul(a float64) {
	for k, _ := range s.repTerms {
		s.repTerms[k] *= a
	}
}

func (s *PlsaSample) Norm() float64 {
	if s.norm == 0 {
		n := float64(0)
		for _, v := range s.repTerms {
			n += v * v
		}
		s.norm = math.Sqrt(n)
	}
	return s.norm
}

func (s *PlsaSample) DistanceFrom(c SampleContainer) float64 {
	a := AssertAsPlsaSample(c)
	dist := float64(0)
	// Parts that are common to both
	for k, v := range s.repTerms {
		if u, found := a.repTerms[k]; found {
			dist += (v - u) * (v - u)
		}
	}
	// Parts that present only in 1
	for k, v := range s.repTerms {
		if _, found := a.repTerms[k]; !found {
			dist += v * v
		}
	}
	for k, u := range a.repTerms {
		if _, found := s.repTerms[k]; !found {
			dist += u * u
		}
	}
	return dist
}

func (s *PlsaSample) CosineSim(c SampleContainer) float64 {
	a := AssertAsPlsaSample(c)
	sDota := float64(0)
	for k, v := range s.repTerms {
		if u, found := a.repTerms[k]; found {
			sDota += u * v
		}
	}
	return sDota / (s.Norm() * a.Norm())
}

type PlsaSampleSupplier struct {
	samples []PlsaSample
}

func (sp *PlsaSampleSupplier) Load(filename string) error {
	return ForEachLineInFile(filename, func(line string) (bool, error) {
		fields := strings.Split(line, " ")
		if len(fields) < 4 {
			log.Printf("Invalid line: %s", line)
		} else {
			topicId, err := strconv.ParseInt(fields[0], 10, 64)
			if err != nil {
				log.Printf("Invalid topic id: %s", fields[0])
			}
			repTerms := make(map[string]float64)
			for i := 2; i < len(fields); i += 2 {
				p, err := strconv.ParseFloat(fields[i+1], 64)
				if err == nil {
					repTerms[fields[i]] = p
				} else {
					log.Printf("Invalid field: %s %s", fields[i], fields[i+1])
				}
			}
			sp.samples = append(sp.samples, PlsaSample{int(topicId), repTerms, float64(0)})
		}
		return true, nil
	})
}

func (sp PlsaSampleSupplier) SampleSize() int {
	return len(sp.samples)
}

func (sp PlsaSampleSupplier) Sample(i int) SampleContainer {
	return SampleContainer(&sp.samples[i])
}

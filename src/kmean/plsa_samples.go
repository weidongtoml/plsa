// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kmean

import (
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"
)

type PlsaSample struct {
	topicId  int
	repTerms map[string]float64
	norm     float64
}

func (s *PlsaSample) String() string {
	return fmt.Sprintf("%d", s.topicId)
}

func (s *PlsaSample) Norm() float64 {
	if s.norm == 0 {
		n := float64(0)
		for _, v := range s.repTerms {
			n += v
		}
		s.norm = math.Sqrt(n)
	}
	return s.norm
}

func (s *PlsaSample) DistanceFrom(a *PlsaSample) float64 {
	dist := float64(0)
	for k, v := range s.repTerms {
		if u, found := a.repTerms[k]; found {
			dist += (v - u) * (v - u)
		}
	}
	return dist
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

func (sp *PlsaSampleSupplier) SampleSize() int {
	return len(sp.samples)
}

func (sp *PlsaSampleSupplier) Sample(i int) PlsaSample {
	return sp.samples[i]
}

func (sp *PlsaSampleSupplier) Mean(s []PlsaSample) PlsaSample {
	var sample PlsaSample
	for _, v := range s {
		for k, u := range v.repTerms {
			sample.repTerms[k] += u
		}
	}
	for k, _ := range sample.repTerms {
		sample.repTerms[k] /= float64(len(s))
	}
	return sample
}

func (sp *PlsaSampleSupplier) Equals(aC []SampleContainer, bC []SampleContainer) bool {
	a := []PlsaSample(aC)

	b := []PlsaSample(bC)
	return contains(a, b) && contains(b, a)

}

func contains(a []PlsaSample, b []PlsaSample) bool {
	for _, v := range a {
		found := false
		for _, u := range b {
			if v.topicId == u.topicId {
				found = true
			}
		}
		if !found {
			return false
		}
	}
	return true
}

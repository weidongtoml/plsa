// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package kmean implements the kmean++ algorithm for clustering.
// The kmean++ cluster algorithm starts by choosing the k initial centers using
// the following procedure:
//   1. Choose one center uniformly at random from among the data points.
//   2. For each data point x, compute D(x), the distane between x and the 
//      nearest center that has been chosen.
//   3. Choose one new data point at random as a new center, using a weighted
//      probability distribution where a point x is chosen with probability
//      proportional to D(x)^2.
//   4. Repeat Steps 2 and 3 until k centers have been chosen.
// After that, proceed using the standard k-means clustering as follow:
//   1. Assigment step: assign each observation to the cluster whose mean is
//		closest to it.
//      S_i_t = {x_p : ||x_p - m_i_t|| <= ||x_p - m_j_t|| for all 1<=j <= k}
//   2. Update step: calculate the new means to be the centroids of the observations
//      in the new clusters.
//		m_i(t+1) = sum(x_j in S_i_j){x_j}/(1/|S_i_t|)
// repeat the above 2 steps until assignments no longer change.
package kmean

import (
	"math"
	"math/rand"
)

type SampleContainer interface {
	String() string
	DistanceFrom(SampleContainer) float64
}

type SampleSupplier interface {
	SampleSize() int
	Sample(int) SampleContainer
	NewSample() SampleContainer
	Mean([]SampleContainer) SampleContainer
}

type Cluster struct {
	Centroid SampleContainer
	Members  []SampleContainer
}

type indexDist struct {
	index int
	dist  float64
}

// Function KMeanCluster clusters the given sample into k clusters.
func KMeanCluster(s SampleSupplier, k int) *[]Cluster {
	//Use kmean++ to select the k initial centers.
	var clusters []Cluster
	ind := rand.Int() % s.SampleSize()
	clusters = append(clusters, Cluster{s.Sample(ind), nil})
	indList := map[int]bool{ind: true}
	for i := 1; i < k; i++ {
		var indexD []indexDist
		for sIndex := 0; sIndex < s.SampleSize(); sIndex++ {
			if indList[sIndex] == false {
				sample := s.Sample(sIndex)
				shortestDist := math.MaxFloat64
				for _, c := range clusters {
					d := sample.DistanceFrom(c.Centroid)
					if d < shortestDist {
						shortestDist = d
					}
				}
				indexD = append(indexD, indexDist{sIndex, shortestDist * shortestDist})
			}
		}
		for j, v := range indexD[1:] {
			v.dist += indexD[j-1].dist
		}
		maxDist := indexD[len(indexD)-1].dist
		newProb := rand.Float64()
		for j, v := range indexD {
			if v.dist/maxDist > newProb {
				clusters = append(clusters, Cluster{s.Sample(j), nil})
			}
		}
	}
	// Use kmean to adjust the clusters till no re-assignment has been made.

	return &clusters
}

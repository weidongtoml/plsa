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

// Interface SampleContainer represents one data sample.
type SampleContainer interface {
	String() string
	DistanceFrom(SampleContainer) float64
}

// Interface SampleSupplier specifies methods for accessing data samples and
// different calculations based on array of samples.
type SampleSupplier interface {
	SampleSize() int
	Sample(int) SampleContainer
	Mean([]SampleContainer) SampleContainer
	Equals([]SampleContainer, []SampleContainer) bool
}

// Representation of a cluster.
type Cluster struct {
	Centroid SampleContainer
	Members  []SampleContainer
}

type indexDist struct {
	index int
	dist  float64
}

// Function KMeanCluster clusters the given sample into k clusters.
func KMeanCluster(s SampleSupplier, k int) []Cluster {
	//Use kmean++ to select the k initial centers.
	clusters := kMeanPlusPlus(s, k)
	// Use kmean to adjust the clusters till no re-assignment has been made.
	return kMean(s, clusters)
}

func kMeanPlusPlus(s SampleSupplier, k int) []Cluster {
	var clusters []Cluster
	ind := rand.Int() % s.SampleSize()
	clusters = append(clusters, Cluster{s.Sample(ind), nil})
	indList := map[int]bool{ind: true}
	for i := 1; i < k; i++ {
		var indexD []indexDist
		for sIndex := 0; sIndex < s.SampleSize(); sIndex++ {
			if !indList[sIndex] {
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
	return clusters
}

func kMean(s SampleSupplier, clusters []Cluster) []Cluster {
	for {
		newClusters := cloneClusterCentroids(clusters)
		//1. Assignment
		for i := 0; i < s.SampleSize(); i++ {
			sample := s.Sample(i)
			index := 0
			dist := math.MaxFloat64
			for j, c := range clusters {
				if cDist := sample.DistanceFrom(c.Centroid); dist < cDist {
					dist = cDist
					index = j
				}
			}
			newClusters[index].add(sample)
		}
		//2. update
		for _, c := range newClusters {
			c.Centroid = s.Mean(c.Members)
		}
		//Check for convergence
		clustersAreEqual := true
		for i, _ := range newClusters {
			if !s.Equals(newClusters[i].Members, clusters[i].Members) {
				clustersAreEqual = false
				break
			}
		}
		if clustersAreEqual {
			break
		} else {
			clusters = newClusters
		}
	}
	return clusters
}

func cloneClusterCentroids(clusters []Cluster) []Cluster {
	var newCluster []Cluster
	for _, c := range clusters {
		newCluster = append(newCluster, Cluster{c.Centroid, nil})
	}
	return newCluster
}

func (cluster *Cluster) add(sample SampleContainer) {
	cluster.Members = append(cluster.Members, sample)
}

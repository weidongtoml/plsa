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
	"log"
	"math"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

// Interface SampleContainer represents one data sample.
type SampleContainer interface {
	Id() int
	Equals(SampleContainer) bool
	DistanceFrom(SampleContainer) float64
	CosineSim(SampleContainer) float64
	Norm() float64
	Add(SampleContainer)
	ScalarMul(float64)
	Zero() SampleContainer
	Normalize()
}

// Interface SampleSupplier specifies methods for accessing data samples and
// different calculations based on array of samples.
type SampleSupplier interface {
	SampleSize() int
	Sample(int) SampleContainer
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
	return kMean(s, clusters, false)
}

// Function SphericalKMeanCluster clusters the given sample into k clusters using
// the spherical kmeans algorithm
func SphericalKMeanCluster(s SampleSupplier, k int) []Cluster {
	// Normalize all samples
	for i := 0; i < s.SampleSize(); i++ {
		s.Sample(i).Normalize()
	}
	//Use kmean++ to select the k initial centers.
	clusters := kMeanPlusPlus(s, k)
	return kMean(s, clusters, true)
}

func normalizeIndexDist(indexD []indexDist) []indexDist {
	for j, _ := range indexD {
		if j > 0 {
			indexD[j].dist += indexD[j-1].dist
		}
	}
	maxDist := indexD[len(indexD)-1].dist
	for j, _ := range indexD {
		indexD[j].dist /= maxDist
	}
	return indexD
}

func kMeanPlusPlus(s SampleSupplier, k int) []Cluster {
	var clusters []Cluster
	indList := make(map[int]bool)
	var ind int
	for i := 1; i <= k; i++ {
		if i == 1 {
			ind = rand.Int() % s.SampleSize()
		} else {
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
			indexD = normalizeIndexDist(indexD)
			newProb := rand.Float64()
			for _, v := range indexD {
				if v.dist > newProb {
					ind = v.index
					break
				}
			}
		}
		clusters = append(clusters, Cluster{i, s.Sample(ind), nil})
		indList[ind] = true

		log.Printf("kmean++: %dth centroid: %d\n", i, ind)
	}

	log.Printf("Inital Centeroids used are: \n")
	for _, c := range clusters {
		log.Printf("%v\n", c.Centroid)
	}
	log.Printf("\n")

	return clusters
}

func nearestCentroid(sample SampleContainer, clusters []Cluster, isSpherical bool) int {
	index := 0
	dist := math.MaxFloat64
	sim := float64(0)
	for j, c := range clusters {
		if isSpherical {
			if cSim := sample.CosineSim(c.Centroid); sim < cSim {
				sim = cSim
				index = j
			}
		} else {
			if cDist := sample.DistanceFrom(c.Centroid); cDist < dist {
				dist = cDist
				index = j
			}
		}
	}
	return index
}

func kMean(s SampleSupplier, clusters []Cluster, isSpherical bool) []Cluster {
	iter := 0
	for {
		log.Printf("Iteration: %v\n", iter)
		newClusters := cloneClusterCentroids(clusters)
		//1. Assignment
		for i := 0; i < s.SampleSize(); i++ {
			sample := s.Sample(i)
			index := nearestCentroid(sample, clusters, isSpherical)
			newClusters[index].add(sample)
		}
		//2. update
		for i, _ := range newClusters {
			newClusters[i].RecalcCentroid(isSpherical)
		}

		log.Printf("Clusters: \n")
		for _, c := range clusters {
			log.Printf("%s\n", c.String())
		}

		//Check for convergence
		clustersAreEqual := true
		for i, _ := range newClusters {
			if !newClusters[i].Equals(&clusters[i]) {
				clustersAreEqual = false
				break
			}
		}
		if clustersAreEqual {
			break
		} else {
			clusters = newClusters
		}
		iter++
	}
	return clusters
}

func cloneClusterCentroids(clusters []Cluster) []Cluster {
	var newCluster []Cluster
	for _, c := range clusters {
		newCluster = append(newCluster, Cluster{c.Id, c.Centroid, nil})
	}
	return newCluster
}

func (cluster *Cluster) add(sample SampleContainer) {
	cluster.Members = append(cluster.Members, sample)
}

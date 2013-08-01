// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./kmean"
	"flag"
	"fmt"
	"log"
)

var (
	corpus = flag.String("corpus", "../data/top_rep_terms/20W_z_top_w_top100.dat",
		"Path of the corpus file for doing the clustering.")
	numCluster = flag.Int("num_cluster", 100, "Number of clusters")
	output     = flag.String("output", "./cluster_result.txt", "file to store the result")
)

func main() {
	var sampleSupplier kmean.PlsaSampleSupplier
	err := sampleSupplier.Load(*corpus)
	if err != nil {
		log.Printf("Error: failed to load corpus file[%s]: %s.\n", *corpus, err)
	} else {
		clusters := kmean.SphericalKMeanCluster(sampleSupplier, *numCluster)

		// Output Result
		for _, c := range clusters {
			avg, stdev := c.PairwiseConsineSimStats()
			fmt.Printf("Pairwise Consine Sim Stats:\nAvg:%f, stdev: %f\n", avg, stdev)
			for _, m := range c.Members {
				fmt.Printf("%v\n", m)
			}
			fmt.Printf("\n..........................\n")
		}

		fmt.Printf("\nInter Cluster Consine Similarity:\nClusterA ClusterB AvgSim StdevSim\n")
		num := len(clusters)
		interAvg := float64(0)
		for i := 0; i < num; i++ {
			for j := i + 1; j < num; j++ {
				avg, stdev := clusters[i].InterClusterConsineSimStats(&clusters[j])
				if avg > 0 {
					interAvg += avg
					fmt.Printf("%d %d %f %f\n", i, j, avg, stdev)
				}

			}
		}
		fmt.Printf(".................................\n")
		fmt.Printf("Inter Cluster Avg Sim: %f\n\n", interAvg/float64(num*(num-1)))
	}
}

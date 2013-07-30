// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"kmean"
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
		log.Printf("Error: failed to load corpus file[%s]: %s", *corpus, err)
	} else {
		clusters := kmean.KMeanCluster(sampleSupplier, *numCluster)
		for _, c := range clusters {
			for _, m := range c.Members {
				fmt.Printf("%v ", m)
			}
			fmt.Printf("\n..........................\n")
		}
	}
}

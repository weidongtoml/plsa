// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kmean

import (
	"fmt"
	"math"
)

// Representation of a cluster.
type Cluster struct {
	Id       int
	Centroid SampleContainer
	Members  []SampleContainer
}

func (c *Cluster) Quality() float64 {
	q := float64(0)
	if len(c.Members) > 0 {
		for _, m := range c.Members {
			q += c.Centroid.CosineSim(m)
		}
		q /= float64(len(c.Members))
	}
	return q
}

func (c *Cluster) String() string {
	str := fmt.Sprintf("Id: %d, Members:", c.Id)
	for _, m := range c.Members {
		str += fmt.Sprintf(" %d", m.Id())
	}
	return str
}

func (c *Cluster) Equals(b *Cluster) bool {
	return c.Contains(b) && b.Contains(c)
}

func (c *Cluster) Contains(b *Cluster) bool {
	for _, u := range b.Members {
		found := false
		for _, v := range c.Members {
			if u.Equals(v) {
				found = true
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (c *Cluster) RecalcCentroid(isSperical bool) {
	z := c.Centroid.Zero()
	for _, m := range c.Members {
		z.Add(m)
	}
	z.ScalarMul(float64(1) / float64(len(c.Members)))
	if isSperical {
		z.ScalarMul(float64(1) / z.Norm())
	}
	c.Centroid = z
}

func (c *Cluster) PairwiseConsineSimStats() (avg, stdev float64) {
	var sim []float64
	var total float64
	n := len(c.Members)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			s := c.Members[i].CosineSim(c.Members[j])
			sim = append(sim, s)
			total += s
		}
	}
	avg = total / float64(n)
	dev := float64(0)
	for _, s := range sim {
		dev += (s - avg) * (s - avg)
	}
	stdev = math.Sqrt(dev)
	return
}

func (c *Cluster) InterClusterConsineSimStats(d *Cluster) (avg, stdev float64) {
	var sim []float64
	var total float64
	for _, cM := range c.Members {
		for _, dM := range d.Members {
			s := cM.CosineSim(dM)
			sim = append(sim, s)
			total += s
		}
	}
	avg = total / float64(len(c.Members)*len(d.Members))
	dev := float64(0)
	for _, s := range sim {
		dev += (s - avg) * (s - avg)
	}
	stdev = math.Sqrt(dev)
	return
}

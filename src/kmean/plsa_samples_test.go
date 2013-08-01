// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kmean

import (
	"bufio"
	"math"
	"os"
	"testing"
)

func Float64Equals(a, b float64) bool {
	return math.Abs(a-b) < 0.00000000001
}

func Square(a float64) float64 {
	return a * a
}

func TestPlsaSample(t *testing.T) {
	var sample PlsaSample
	c0 := SampleContainer(&sample)
	// Test for Zero
	c1 := c0.Zero()
	if &c0 == &c1 || c0 == c1 {
		t.Errorf("PlsaSample.Zero should produce a new PlsaSample object.")
	}
	s1 := AssertAsPlsaSample(c1)
	if s1.topicId != 0 || s1.repTerms == nil || s1.norm != 0 {
		t.Errorf("PlsaSamle.Zero produces invalid result %v.", s1)
	}
	// Test for Equality
	if !c0.Equals(c1) {
		t.Errorf("PlsaSample.Equals failed.")
	}
	// Test for Addition
	s2 := &PlsaSample{2, map[string]float64{"鲜花": 0.12, "快递": 0.22}, float64(0)}
	c2 := SampleContainer(s2)
	c1.Add(c2)
	if s1.repTerms["鲜花"] != 0.12 || s1.repTerms["快递"] != 0.22 {
		t.Errorf("PlsaSample.Add failed, expected %v but got %v.", s2, s1)
	}
	s3 := &PlsaSample{2, map[string]float64{"游戏": 0.99, "快递": 0.22}, float64(0)}
	c3 := SampleContainer(s3)
	c1.Add(c3)
	if s1.repTerms["鲜花"] != 0.12 || s1.repTerms["快递"] != 0.44 || s1.repTerms["游戏"] != 0.99 {
		t.Errorf("PlsaSample.Add failed, got %v.", s1)
	}
	// Test for Scalar Multiplication
	c1.ScalarMul(0.2)
	if !Float64Equals(s1.repTerms["鲜花"], 0.12*0.2) ||
		!Float64Equals(s1.repTerms["快递"], 0.44*0.2) ||
		!Float64Equals(s1.repTerms["游戏"], 0.99*0.2) {
		t.Errorf("PlsaSample.Add failed, got %v.", s1)
	}
	// Test for DistanceFrom
	if !Float64Equals(c1.DistanceFrom(c2), c2.DistanceFrom(c1)) {
		t.Errorf("Expected (a)DistanceFrom(b) to be equal to (b)DistanceFrom(a) but found otherwise.")
	}
	expectedDist := Square(0.12*0.2-0.12) + Square(0.44*0.2-0.22) + Square(0.99*0.2-0)
	if !Float64Equals(c1.DistanceFrom(c2), expectedDist) {
		t.Errorf("Expected distance to be %f, but got %f.", expectedDist, c1.DistanceFrom(c2))
	}
	// Test for CosineSim
	expectedSim := (0.12*0.2*0.12 + 0.44*0.2*0.22) /
		(math.Sqrt(Square(0.12)+Square(0.22)) * math.Sqrt(Square(0.12*0.2)+Square(0.44*0.2)+Square(0.99*0.2)))
	if !Float64Equals(c1.CosineSim(c2), expectedSim) {
		t.Errorf("Expected consine sim to be %f, but got %f.", expectedSim, c1.CosineSim(c2))
	}
	// Test for normalization
	c1.Normalize()
	if !Float64Equals(c1.Norm(), 1.0) {
		t.Errorf("Expected normalized sample to have norm of 1.0 but got %f.", c1.Norm())
	}
}

func TestPlsaSampleSupplier(t *testing.T) {
	var supplier PlsaSampleSupplier
	fileContent := []string{
		"0 0.1 鲜花 0.1 玫瑰 0.2 百合 0.3",
		"1 0.2 游戏 0.2 动画 0.3",
	}
	testFile := "kmean_test.txt"
	defer func() {
		os.Remove(testFile)
	}()
	err := WithNewOpenFileAsBufioWriter(testFile, func(w *bufio.Writer) error {
		for _, c := range fileContent {
			w.WriteString(c + "\n")
		}
		return nil
	})
	if err != nil {
		t.Errorf("Failed to create test file [%s] : %s", testFile, err)
	}
	err = supplier.Load(testFile)
	if err != nil {
		t.Errorf("PlsaSampleSupplier.Load(%s) failed: %s", testFile, err)
	} else {
		if supplier.SampleSize() != len(fileContent) {
			t.Errorf("Number of samples loaded is not the same as in file.")
		}
		c0 := supplier.Sample(0)
		s0 := AssertAsPlsaSample(c0)
		if !Float64Equals(s0.repTerms["鲜花"], 0.1) ||
			!Float64Equals(s0.repTerms["玫瑰"], 0.2) ||
			!Float64Equals(s0.repTerms["百合"], 0.3) ||
			len(s0.repTerms) != 3 {
			//TODO(weidoliang): fix this
			t.Errorf("Sample 0 did not load correctly: %v.", s0)
		}
		c1 := supplier.Sample(1)
		s1 := AssertAsPlsaSample(c1)
		if !Float64Equals(s1.repTerms["游戏"], 0.2) ||
			!Float64Equals(s1.repTerms["动画"], 0.3) ||
			len(s1.repTerms) != 2 {
			//TODO(weidoliang): fix this
			t.Errorf("Sample 0 did not load correctly: %v.", s1)
		}
	}
}

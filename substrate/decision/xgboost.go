// Package decision is the decision stage (design §15.3): for an ordered pair of the points of one
// case and one front, the probability that the first dominates the second, and from those
// probabilities each point's predicted membership of the front.
//
// The stage is an ensemble of gradient-boosted trees, one per front, saved by XGBoost as JSON.
// This file evaluates such a model without XGBoost. The rest of the package builds the model's
// inputs from two points, normalises its answers and applies the front rule.
package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// Model is a binary classifier saved by XGBoost 2.x (Booster.save_model("m.json")): a gbtree
// booster with the binary:logistic objective. Predict is exact to the precision of XGBoost's own
// single-precision sums.
type Model struct {
	// NumFeatures is the length of the input row.
	NumFeatures int
	baseMargin  float64
	trees       []tree
}

type tree struct {
	left, right, feature []int32
	// cond is the split threshold of an inner node and the leaf value of a leaf.
	cond        []float32
	defaultLeft []bool
}

// The layout of an XGBoost JSON model, reduced to what evaluation reads.
type modelFile struct {
	Learner struct {
		Param struct {
			BaseScore  string `json:"base_score"`
			NumClass   string `json:"num_class"`
			NumFeature string `json:"num_feature"`
			NumTarget  string `json:"num_target"`
		} `json:"learner_model_param"`
		Objective struct {
			Name string `json:"name"`
		} `json:"objective"`
		Booster struct {
			Name  string `json:"name"`
			Model struct {
				Param struct {
					NumTrees string `json:"num_trees"`
				} `json:"gbtree_model_param"`
				Trees []struct {
					Left        []int32   `json:"left_children"`
					Right       []int32   `json:"right_children"`
					SplitIndex  []int32   `json:"split_indices"`
					SplitCond   []float64 `json:"split_conditions"`
					DefaultLeft []int     `json:"default_left"`
					SplitType   []int     `json:"split_type"`
				} `json:"trees"`
				TreeInfo []int `json:"tree_info"`
			} `json:"model"`
		} `json:"gradient_booster"`
	} `json:"learner"`
}

// LoadModel reads an XGBoost JSON model from a file.
func LoadModel(path string) (*Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m, err := ParseModel(b)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

// ParseModel reads an XGBoost JSON model. It refuses anything but a gbtree booster with the
// binary:logistic objective and numerical splits, rather than evaluate it wrongly.
func ParseModel(data []byte) (*Model, error) {
	var f modelFile
	dec := json.NewDecoder(bytes.NewReader(data))
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("not an XGBoost JSON model: %w", err)
	}
	l := f.Learner
	if l.Booster.Name != "gbtree" {
		return nil, fmt.Errorf("booster %q: only gbtree is evaluated", l.Booster.Name)
	}
	if l.Objective.Name != "binary:logistic" {
		return nil, fmt.Errorf("objective %q: only binary:logistic is evaluated", l.Objective.Name)
	}
	if nc, _ := strconv.Atoi(l.Param.NumClass); nc > 1 {
		return nil, fmt.Errorf("num_class %d: a binary classifier has none", nc)
	}
	nf, err := strconv.Atoi(l.Param.NumFeature)
	if err != nil || nf < 1 {
		return nil, fmt.Errorf("num_feature %q is not a positive integer", l.Param.NumFeature)
	}
	base, err := parseBaseScore(l.Param.BaseScore)
	if err != nil {
		return nil, err
	}
	m := &Model{NumFeatures: nf, baseMargin: math.Log(base / (1 - base))}
	for i, t := range l.Booster.Model.Trees {
		n := len(t.Left)
		if n == 0 || len(t.Right) != n || len(t.SplitIndex) != n || len(t.SplitCond) != n || len(t.DefaultLeft) != n {
			return nil, fmt.Errorf("tree %d: node arrays differ in length", i)
		}
		if len(l.Booster.Model.TreeInfo) > i && l.Booster.Model.TreeInfo[i] != 0 {
			return nil, fmt.Errorf("tree %d: belongs to class %d of a multi-class model", i, l.Booster.Model.TreeInfo[i])
		}
		tr := tree{left: t.Left, right: t.Right, feature: t.SplitIndex, cond: make([]float32, n), defaultLeft: make([]bool, n)}
		for k := 0; k < n; k++ {
			tr.cond[k] = float32(t.SplitCond[k])
			tr.defaultLeft[k] = t.DefaultLeft[k] != 0
			if len(t.SplitType) == n && t.SplitType[k] != 0 {
				return nil, fmt.Errorf("tree %d node %d: categorical splits are not evaluated", i, k)
			}
			if t.Left[k] == -1 {
				continue
			}
			if int(t.Left[k]) >= n || int(t.Right[k]) >= n || t.Left[k] < 0 || t.Right[k] < 0 {
				return nil, fmt.Errorf("tree %d node %d: child out of range", i, k)
			}
			if int(t.SplitIndex[k]) >= nf || t.SplitIndex[k] < 0 {
				return nil, fmt.Errorf("tree %d node %d: splits on feature %d of %d", i, k, t.SplitIndex[k], nf)
			}
		}
		m.trees = append(m.trees, tr)
	}
	if want, err := strconv.Atoi(l.Booster.Model.Param.NumTrees); err == nil && want != len(m.trees) {
		return nil, fmt.Errorf("num_trees is %d and the file holds %d trees", want, len(m.trees))
	}
	return m, nil
}

// parseBaseScore reads the base score, a probability, written as "5E-1" (XGBoost 2.1) or as a
// one-element list "[5E-1]" (later versions).
func parseBaseScore(s string) (float64, error) {
	s = strings.Trim(strings.TrimSpace(s), "[]")
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || !(v > 0 && v < 1) {
		return 0, fmt.Errorf("base_score %q is not a probability strictly between 0 and 1", s)
	}
	return v, nil
}

// Margin is the model's raw score of a row: the logit of the base score plus the leaf values.
// A NaN entry is a missing value and follows each split's default direction. The row is
// compared in single precision, as XGBoost does.
func (m *Model) Margin(row []float32) float64 {
	sum := float32(0)
	for i := range m.trees {
		t := &m.trees[i]
		n := int32(0)
		for t.left[n] != -1 {
			v := row[t.feature[n]]
			switch {
			case v != v:
				if t.defaultLeft[n] {
					n = t.left[n]
				} else {
					n = t.right[n]
				}
			case v < t.cond[n]:
				n = t.left[n]
			default:
				n = t.right[n]
			}
		}
		sum += t.cond[n]
	}
	return m.baseMargin + float64(sum)
}

// Predict is the probability of the positive class for a row of NumFeatures values.
func (m *Model) Predict(row []float32) (float64, error) {
	if len(row) != m.NumFeatures {
		return 0, fmt.Errorf("row has %d values and the model takes %d", len(row), m.NumFeatures)
	}
	return 1 / (1 + math.Exp(-m.Margin(row))), nil
}

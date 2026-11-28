package decision

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// Python XGBoost's probabilities for 300 rows, a third of them with every kind of missing value,
// are reproduced to 1e-6.
func TestTheEvaluatorMatchesPythonXGBoost(t *testing.T) {
	m, err := LoadModel(filepath.Join("testdata", "model.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c struct {
		Inputs [][]*float64 `json:"inputs"`
		Proba  []float64    `json:"proba"`
	}
	readJSON(t, filepath.Join("testdata", "cases.json"), &c)
	if len(c.Inputs) < 100 {
		t.Fatalf("fixture has %d cases", len(c.Inputs))
	}
	missing, worst := 0, 0.0
	for i, in := range c.Inputs {
		row := make([]float32, len(in))
		for j, v := range in {
			if v == nil {
				row[j] = float32(math.NaN())
				missing++
			} else {
				row[j] = float32(*v)
			}
		}
		p, err := m.Predict(row)
		if err != nil {
			t.Fatal(err)
		}
		worst = math.Max(worst, math.Abs(p-c.Proba[i]))
	}
	if missing == 0 {
		t.Fatal("the fixture has no missing values")
	}
	if worst > 1e-6 {
		t.Errorf("largest difference from Python is %g, want at most 1e-6", worst)
	}
}

// A hand-written stump: x0 < 0.5 goes left to a leaf of -1, else right to +2; a missing x0 goes
// the way default_left says. The base score of 0.25 is a margin of ln(1/3).
func stump(defaultLeft int, extra string) string {
	d := "0"
	if defaultLeft == 1 {
		d = "1"
	}
	return `{"learner":{"learner_model_param":{"base_score":"2.5E-1","num_class":"0","num_feature":"2","num_target":"1"},
	 "objective":{"name":"binary:logistic"},
	 "gradient_booster":{"name":"gbtree","model":{"gbtree_model_param":{"num_trees":"1"},"tree_info":[0],
	 "trees":[{"left_children":[1,-1,-1],"right_children":[2,-1,-1],"split_indices":[0,0,0],
	  "split_conditions":[0.5,-1,2],"default_left":[` + d + `,0,0]` + extra + `}]}}}}`
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func TestAMissingValueFollowsTheDefaultDirectionAndTheBaseScoreShiftsTheMargin(t *testing.T) {
	base := math.Log(0.25 / 0.75)
	for _, c := range []struct {
		name        string
		defaultLeft int
		x0          float32
		leaf        float64
	}{
		{"below the threshold", 0, 0.4, -1},
		{"at the threshold goes right", 0, 0.5, 2},
		{"above the threshold", 1, 0.9, 2},
		{"missing, default left", 1, float32(math.NaN()), -1},
		{"missing, default right", 0, float32(math.NaN()), 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := ParseModel([]byte(stump(c.defaultLeft, "")))
			if err != nil {
				t.Fatal(err)
			}
			p, err := m.Predict([]float32{c.x0, 0})
			if err != nil {
				t.Fatal(err)
			}
			if want := sigmoid(base + c.leaf); math.Abs(p-want) > 1e-9 {
				t.Errorf("got %v, want %v", p, want)
			}
		})
	}
}

func TestTheEvaluatorRefusesWhatItWouldEvaluateWrongly(t *testing.T) {
	good := stump(0, "")
	for _, c := range []struct{ name, doc, want string }{
		{"dart", strings.Replace(good, `"gbtree"`, `"dart"`, 1), "only gbtree"},
		{"another objective", strings.Replace(good, "binary:logistic", "reg:squarederror", 1), "binary:logistic"},
		{"multi-class", strings.Replace(good, `"num_class":"0"`, `"num_class":"3"`, 1), "num_class"},
		{"a categorical split", stump(0, `,"split_type":[1,0,0]`), "categorical"},
		{"a feature past the end", strings.Replace(good, `"split_indices":[0,0,0]`, `"split_indices":[5,0,0]`, 1), "feature 5"},
		{"a base score of 1", strings.Replace(good, "2.5E-1", "1", 1), "base_score"},
		{"not JSON", "tree", "not an XGBoost"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := ParseModel([]byte(c.doc)); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("got %v, want an error containing %q", err, c.want)
			}
		})
	}
	m, _ := ParseModel([]byte(good))
	if _, err := m.Predict([]float32{1}); err == nil {
		t.Error("a row of the wrong length was accepted")
	}
}

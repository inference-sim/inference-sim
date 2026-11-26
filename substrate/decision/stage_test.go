package decision

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"
)

// The normalised pair probabilities, the memberships and the front of the stage equal what the
// the training pipeline computes with the same models on seven real points.
func TestTheStageMatchesTheTrainingPipelinesFrontOnRealPoints(t *testing.T) {
	s := loadArtifactStage(t)
	_, all := loadFeatureFixture(t)
	var par struct {
		Points []int `json:"points"`
		Fronts map[Front]struct {
			Pi          [][]float64 `json:"pi"`
			OnFront     []bool      `json:"on_front"`
			Probability []float64   `json:"probability"`
		} `json:"fronts"`
	}
	readJSON(t, filepath.Join("testdata", "artifact", "parity.json"), &par)
	var pts []Point
	for _, k := range par.Points {
		pts = append(pts, all[k])
	}
	for _, f := range Fronts {
		m, pi, err := s.Front(f, pts)
		if err != nil {
			t.Fatal(err)
		}
		want := par.Fronts[f]
		for i := range pts {
			if m[i].OnFront != want.OnFront[i] {
				t.Errorf("front %s point %d: on front %v, Python %v", f, i, m[i].OnFront, want.OnFront[i])
			}
			if math.Abs(m[i].Probability-want.Probability[i]) > 1e-6 {
				t.Errorf("front %s point %d: probability %v, Python %v", f, i, m[i].Probability, want.Probability[i])
			}
			for j := range pts {
				if i != j && math.Abs(pi[i][j]-want.Pi[i][j]) > 1e-6 {
					t.Errorf("front %s pair (%d,%d): %v, Python %v", f, i, j, pi[i][j], want.Pi[i][j])
				}
			}
		}
	}
}

// randomPoints draws points near the real ones: each takes a real point's configuration and
// gets random predictions.
func randomPoints(t testing.TB, r *rand.Rand, n int) []Point {
	_, base := loadFeatureFixture(t)
	out := make([]Point, n)
	for i := range out {
		p := base[r.Intn(len(base))]
		p.ID = string(rune('a' + i))
		p.LogX, p.LogI, p.LogT = r.NormFloat64(), r.NormFloat64(), r.NormFloat64()
		out[i] = p
	}
	return out
}

// DEC-4: one constant per objective added to the predictions of every point of a case leaves
// every probability, membership and front unchanged.
func TestAddingAConstantToEveryPointsPredictionsChangesNothing(t *testing.T) {
	s := loadArtifactStage(t)
	r := rand.New(rand.NewSource(1))
	for trial := 0; trial < 25; trial++ {
		pts := randomPoints(t, r, 8)
		cx, ci, ct := 3*r.NormFloat64(), 3*r.NormFloat64(), 3*r.NormFloat64()
		moved := append([]Point(nil), pts...)
		for i := range moved {
			moved[i].LogX += cx
			moved[i].LogI += ci
			moved[i].LogT += ct
		}
		for _, f := range Fronts {
			m0, p0, err := s.Front(f, pts)
			if err != nil {
				t.Fatal(err)
			}
			m1, p1, _ := s.Front(f, moved)
			for i := range pts {
				if m0[i].OnFront != m1[i].OnFront || math.Abs(m0[i].Probability-m1[i].Probability) > 1e-6 {
					t.Fatalf("trial %d front %s point %d: %+v became %+v", trial, f, i, m0[i], m1[i])
				}
				for j := range pts {
					if i != j && math.Abs(p0[i][j]-p1[i][j]) > 1e-6 {
						t.Fatalf("trial %d front %s pair (%d,%d): %v became %v", trial, f, i, j, p0[i][j], p1[i][j])
					}
				}
			}
		}
	}
}

// DEC-4: the two probabilities of a pair sum to at most one, and asking in the other order
// returns the same two numbers swapped.
func TestThePairProbabilitiesSumToAtMostOneAndDoNotDependOnOrder(t *testing.T) {
	s := loadArtifactStage(t)
	r := rand.New(rand.NewSource(2))
	for trial := 0; trial < 300; trial++ {
		pts := randomPoints(t, r, 2)
		for _, f := range Fronts {
			pq, qp, err := s.Pair(f, pts[0], pts[1])
			if err != nil {
				t.Fatal(err)
			}
			rq, rp, _ := s.Pair(f, pts[1], pts[0])
			if pq < 0 || qp < 0 || pq+qp > 1+1e-12 {
				t.Fatalf("front %s: %v + %v", f, pq, qp)
			}
			if pq != rp || qp != rq {
				t.Fatalf("front %s: (%v,%v) asked one way, (%v,%v) the other", f, pq, qp, rq, rp)
			}
		}
	}
}

// DEC-4: a larger predicted advantage of the first point never lowers the probability that it
// dominates (X and I up, T down).
func TestAFirstPointsLargerPredictedAdvantageNeverLowersItsRawProbability(t *testing.T) {
	s := loadArtifactStage(t)
	r := rand.New(rand.NewSource(3))
	for trial := 0; trial < 200; trial++ {
		pts := randomPoints(t, r, 2)
		for _, f := range Fronts {
			x, i, tt := f.uses()
			base, _ := s.Raw(f, pts[0], pts[1])
			better := pts[0]
			step := 0.05 + r.Float64()
			if x {
				better.LogX += step
			}
			if i {
				better.LogI += step
			}
			if tt {
				better.LogT -= step
			}
			after, _ := s.Raw(f, better, pts[1])
			if after < base-1e-12 {
				t.Fatalf("front %s: advantage grew by %v and the probability fell from %v to %v", f, step, base, after)
			}
		}
	}
}

// The front rule on a stub whose raw estimates are set by the caller: a point is off the front
// exactly when another dominates it with probability above one half.
func TestAPointIsOnTheFrontUnlessAnotherDominatesItAboveOneHalf(t *testing.T) {
	// A model of one stump on dX: a is better than b by more than 0 gives 0.9, else 0.05.
	m, err := ParseModel([]byte(`{"learner":{"learner_model_param":{"base_score":"5E-1","num_class":"0","num_feature":"1","num_target":"1"},
	 "objective":{"name":"binary:logistic"},"gradient_booster":{"name":"gbtree","model":{"gbtree_model_param":{"num_trees":"1"},"tree_info":[0],
	 "trees":[{"left_children":[1,-1,-1],"right_children":[2,-1,-1],"split_indices":[0,0,0],
	 "split_conditions":[0.0,-3,3],"default_left":[0,0,0]}]}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	s := &Stage{Epsilon: DefaultEpsilon, Features: []string{"dX"}, Models: map[Front]*Model{FrontXI: m}}
	pts := []Point{{ID: "best", LogX: 2, LogI: 0}, {ID: "mid", LogX: 1, LogI: 0}, {ID: "worst", LogX: 0, LogI: 0}, {ID: "refused", LogX: math.NaN(), LogI: 0}}
	mem, pi, err := s.Front(FrontXI, pts)
	if err != nil {
		t.Fatal(err)
	}
	want := []bool{true, false, false}
	for i, w := range want {
		if mem[i].OnFront != w {
			t.Errorf("%s: on front %v, want %v (pi %v)", pts[i].ID, mem[i].OnFront, w, pi[0][i])
		}
	}
	if mem[0].Probability < 0.9 || mem[2].Probability > 0.1 {
		t.Errorf("probabilities %+v", mem)
	}
	if mem[3].Answered || !math.IsNaN(pi[0][3]) {
		t.Errorf("a point without a prediction was placed: %+v", mem[3])
	}
}

// A model that thinks each point dominates the other (0.88 each way) is normalised to one half
// each: two points cannot dominate each other, and neither is then called dominated.
func TestAModelThatSaysBothDominateIsNormalisedToHalfEach(t *testing.T) {
	m, err := ParseModel([]byte(`{"learner":{"learner_model_param":{"base_score":"5E-1","num_class":"0","num_feature":"1","num_target":"1"},
	 "objective":{"name":"binary:logistic"},"gradient_booster":{"name":"gbtree","model":{"gbtree_model_param":{"num_trees":"1"},"tree_info":[0],
	 "trees":[{"left_children":[1,-1,-1],"right_children":[2,-1,-1],"split_indices":[0,0,0],
	 "split_conditions":[-1e9,0,2],"default_left":[0,0,0]}]}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	s := &Stage{Epsilon: DefaultEpsilon, Features: []string{"dX"}, Models: map[Front]*Model{FrontXI: m}}
	p, q := Point{LogX: 1, LogI: 1}, Point{LogX: 0, LogI: 0}
	if g1, _ := s.Raw(FrontXI, p, q); g1 < 0.8 {
		t.Fatalf("stub raw estimate %v", g1)
	}
	pq, qp, _ := s.Pair(FrontXI, p, q)
	if math.Abs(pq-0.5) > 1e-12 || math.Abs(qp-0.5) > 1e-12 {
		t.Errorf("got %v and %v, want one half each", pq, qp)
	}
	mem, _, _ := s.Front(FrontXI, []Point{p, q})
	if !mem[0].OnFront || !mem[1].OnFront {
		t.Errorf("a probability of exactly one half dominates: %+v", mem)
	}
}

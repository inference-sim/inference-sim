package decision

import (
	"fmt"
	"math"
)

// Front names a set of objectives on which dominance is judged: X throughput per GPU, I
// interactivity, T the P90 time to first token.
type Front string

const (
	FrontXI  Front = "XI"
	FrontXT  Front = "XT"
	FrontIT  Front = "IT"
	FrontXIT Front = "XIT"
)

// Fronts lists the four fronts of a case, in the order outputs use.
var Fronts = []Front{FrontXI, FrontXT, FrontIT, FrontXIT}

// Objectives of a front: which of the point's three predictions it judges.
func (f Front) uses() (x, i, t bool) {
	return f == FrontXI || f == FrontXT || f == FrontXIT,
		f == FrontXI || f == FrontIT || f == FrontXIT,
		f == FrontXT || f == FrontIT || f == FrontXIT
}

// Answered reports whether the simulator predicted every objective the front judges. A point
// it did not answer on a front cannot be placed on it.
func (p Point) Answered(f Front) bool {
	x, i, t := f.uses()
	ok := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	return (!x || ok(p.LogX)) && (!i || ok(p.LogI)) && (!t || ok(p.LogT))
}

// Stage is the decision stage: one fitted model per front, and the inputs each takes.
type Stage struct {
	// Epsilon is the dominance tolerance the models were fitted with: a point dominates another
	// when it is better on every objective of a front by more than a factor 1+Epsilon.
	Epsilon  float64
	Features []string
	Models   map[Front]*Model
}

// Raw is the model's estimate g_pq that p dominates q on the front, before normalisation.
func (s *Stage) Raw(f Front, p, q Point) (float64, error) {
	m := s.Models[f]
	if m == nil {
		return 0, fmt.Errorf("the stage has no model for front %s", f)
	}
	row, err := PairFeatures(s.Features, p, q)
	if err != nil {
		return 0, err
	}
	return m.Predict(row)
}

// Pair returns the probabilities that p dominates q and that q dominates p. The model is trained on both
// orderings of every pair, and two points cannot dominate each other, so its two estimates are
// normalised: pi_pq = g_pq / max(1, g_pq + g_qp). Hence pi_pq + pi_qp <= 1, and Pair(f, q, p)
// returns the same two values swapped.
func (s *Stage) Pair(f Front, p, q Point) (pq, qp float64, err error) {
	gpq, err := s.Raw(f, p, q)
	if err != nil {
		return 0, 0, err
	}
	gqp, err := s.Raw(f, q, p)
	if err != nil {
		return 0, 0, err
	}
	d := math.Max(1, gpq+gqp)
	return gpq / d, gqp / d, nil
}

// Membership is the stage's answer for one point on one front.
type Membership struct {
	// Answered is false when the simulator gave no prediction for an objective of the front. The
	// point then has no membership: OnFront and Probability are zero and meaningless.
	Answered bool
	// OnFront: no answered point dominates this one with probability above one half.
	OnFront bool
	// Probability that the point is on the front: the product over the other answered points q
	// of 1 - pi_qp.
	Probability float64
}

// Front places each point of one case on front f. pi[p][q] is the probability that p dominates
// q (NaN where either point is unanswered, and on the diagonal). Only answered points are compared.
func (s *Stage) Front(f Front, pts []Point) (m []Membership, pi [][]float64, err error) {
	n := len(pts)
	m = make([]Membership, n)
	pi = make([][]float64, n)
	for i := range pi {
		pi[i] = make([]float64, n)
		for j := range pi[i] {
			pi[i][j] = math.NaN()
		}
		m[i] = Membership{Answered: pts[i].Answered(f), OnFront: true, Probability: 1}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if !m[i].Answered || !m[j].Answered {
				continue
			}
			a, b, err := s.Pair(f, pts[i], pts[j])
			if err != nil {
				return nil, nil, err
			}
			pi[i][j], pi[j][i] = a, b
		}
	}
	for j := 0; j < n; j++ {
		if !m[j].Answered {
			m[j] = Membership{}
			continue
		}
		for i := 0; i < n; i++ {
			if i == j || !m[i].Answered {
				continue
			}
			m[j].Probability *= 1 - pi[i][j]
			if pi[i][j] > 0.5 {
				m[j].OnFront = false
			}
		}
	}
	return m, pi, nil
}

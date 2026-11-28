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

// Stage is the decision stage: per front, the simulator model and the inputs each model takes, and
// the fallback model for the pairs the simulator does not answer.
type Stage struct {
	// Epsilon is the dominance tolerance the models were fitted with: a point dominates another
	// when it is better on every objective of a front by more than a factor 1+Epsilon.
	Epsilon  float64
	Features []string
	// Models is the simulator model of each front: fitted on pairs whose two points the simulator
	// answers, monotone in its predicted differences.
	Models map[Front]*Model
	// Fallback is the fallback model of each front, or nil. It takes the same inputs with the
	// simulator's differences missing, and was fitted on every pair. A stage without it places
	// only the points the simulator answered.
	Fallback map[Front]*Model
	masked   []int // positions in Features that the fallback reads as missing
}

// Basis says which model answered a pair.
type Basis string

const (
	// BasisSimulator: the simulator answered both points on the front, and the simulator model gave the probabilities.
	BasisSimulator Basis = "simulator"
	// BasisFallback: the simulator did not answer one point or both; the fallback gave the probabilities, and they are estimates (DEC-3).
	BasisFallback Basis = "fallback"
)

// setMasked records which features the fallback models read as missing.
func (s *Stage) setMasked(names []string) {
	s.masked = nil
	for i, f := range s.Features {
		for _, n := range names {
			if f == n {
				s.masked = append(s.masked, i)
			}
		}
	}
}

// HasFallback reports whether the stage can place a point the simulator did not answer.
func (s *Stage) HasFallback() bool { return len(s.Fallback) > 0 }

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

// RawFallback is the fallback model's estimate g_pq that p dominates q on the front, before normalisation.
// The simulator's differences are passed as missing, as they were in every row the model was fitted on.
func (s *Stage) RawFallback(f Front, p, q Point) (float64, error) {
	m := s.Fallback[f]
	if m == nil {
		return 0, fmt.Errorf("the stage has no fallback model for front %s", f)
	}
	row, err := PairFeatures(s.Features, p, q)
	if err != nil {
		return 0, err
	}
	for _, i := range s.masked {
		row[i] = float32(math.NaN())
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

// Combined returns the probabilities that p dominates q and that q dominates p, and which model gave
// them. A pair whose two points the simulator answers on the front goes to the simulator model; any other
// pair goes to the fallback. Either way the two estimates of the pair are normalised together, so the
// two probabilities sum to at most one and asking in the other order returns the same two numbers
// swapped (DEC-4). Without a fallback a pair the simulator does not answer is an error.
func (s *Stage) Combined(f Front, p, q Point) (pq, qp float64, basis Basis, err error) {
	if p.Answered(f) && q.Answered(f) {
		pq, qp, err = s.Pair(f, p, q)
		return pq, qp, BasisSimulator, err
	}
	if !s.HasFallback() {
		return 0, 0, "", fmt.Errorf("the simulator did not answer %s or %s on front %s, and the stage has no fallback", p.ID, q.ID, f)
	}
	gpq, err := s.RawFallback(f, p, q)
	if err != nil {
		return 0, 0, "", err
	}
	gqp, err := s.RawFallback(f, q, p)
	if err != nil {
		return 0, 0, "", err
	}
	d := math.Max(1, gpq+gqp)
	return gpq / d, gqp / d, BasisFallback, nil
}

// PairBasis names the model that answers the pair on the front.
func (s *Stage) PairBasis(f Front, p, q Point) Basis {
	if p.Answered(f) && q.Answered(f) {
		return BasisSimulator
	}
	return BasisFallback
}

// Membership is the stage's answer for one point on one front.
type Membership struct {
	// Answered is false when the simulator gave no prediction for an objective of the front. Without a
	// fallback the point then has no membership: OnFront and Probability are zero and meaningless.
	Answered bool
	// Estimated is true when the stage placed a point the simulator did not answer, by its
	// fallback (DEC-3). Every value of such a point is an estimate. Exactly one of Answered and
	// Estimated is true when the point is placed.
	Estimated bool
	// OnFront: no answered point dominates this one with probability above one half.
	OnFront bool
	// Probability that the point is on the front: the product over the other answered points q
	// of 1 - pi_qp.
	Probability float64
}

// Front places each point of one case on front f. pi[p][q] is the probability that p dominates
// q (NaN on the diagonal, and, for a stage without a fallback, where either point is unanswered).
// With a fallback every point is placed: a pair the simulator answers on both points is judged by
// the simulator model, any other pair by the fallback, and a point the simulator did not answer
// is marked Estimated. Without a fallback only the answered points are compared.
func (s *Stage) Front(f Front, pts []Point) (m []Membership, pi [][]float64, err error) {
	n := len(pts)
	m = make([]Membership, n)
	pi = make([][]float64, n)
	fb := s.HasFallback()
	placed := make([]bool, n)
	for i := range pi {
		pi[i] = make([]float64, n)
		for j := range pi[i] {
			pi[i][j] = math.NaN()
		}
		a := pts[i].Answered(f)
		placed[i] = a || fb
		m[i] = Membership{Answered: a, Estimated: !a && fb, OnFront: true, Probability: 1}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if !placed[i] || !placed[j] {
				continue
			}
			a, b, _, err := s.Combined(f, pts[i], pts[j])
			if err != nil {
				return nil, nil, err
			}
			pi[i][j], pi[j][i] = a, b
		}
	}
	for j := 0; j < n; j++ {
		if !placed[j] {
			m[j] = Membership{}
			continue
		}
		for i := 0; i < n; i++ {
			if i == j || !placed[i] {
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

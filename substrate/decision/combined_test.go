package decision

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// combinedFixture is testdata/combined/parity.json: real points of decision cases, with the
// simulator's predictions (null where it refused), and the combined stage's output on them as the
// training pipeline's own code computes it with the shipped models.
type combinedFixture struct {
	Cases []struct {
		Case   string `json:"case"`
		Points []struct {
			ID     string   `json:"id"`
			Kind   float64  `json:"kind"`
			TP     nullable `json:"tp"`
			EP     nullable `json:"ep"`
			DP     nullable `json:"dp"`
			GPUs   nullable `json:"gpus"`
			PD     nullable `json:"pd"`
			Spec   nullable `json:"spec"`
			Seqs   nullable `json:"seqs"`
			Btok   nullable `json:"btok"`
			PDFrac nullable `json:"pdfrac"`
			LConc  nullable `json:"lconc"`
			MTot   nullable `json:"m_tot"`
			MAct   nullable `json:"m_act"`
			CHBM   nullable `json:"c_hbm"`
			CBW    nullable `json:"c_bw"`
			CFP8   nullable `json:"c_fp8"`
			Engine nullable `json:"engine"`
			LISL   nullable `json:"lisl"`
			LOSL   nullable `json:"losl"`
			Src    nullable `json:"src"`
			SimX   nullable `json:"sim_log_x"`
			SimI   nullable `json:"sim_log_i"`
			SimT   nullable `json:"sim_log_t"`
		} `json:"points"`
		Fronts map[Front]struct {
			Pi          [][]float64 `json:"pi"`
			Answered    []bool      `json:"answered"`
			OnFront     []bool      `json:"on_front"`
			Probability []float64   `json:"probability"`
		} `json:"fronts"`
	} `json:"cases"`
}

func (c combinedFixture) points(k int) []Point {
	var pts []Point
	for _, p := range c.Cases[k].Points {
		f := func(n nullable) float64 { return float64(n) }
		pts = append(pts, Point{ID: p.ID, LogX: f(p.SimX), LogI: f(p.SimI), LogT: f(p.SimT),
			Kind: int(p.Kind), TP: f(p.TP), EP: f(p.EP), DP: f(p.DP), GPUs: f(p.GPUs), PD: f(p.PD), Spec: f(p.Spec),
			Seqs: f(p.Seqs), Btok: f(p.Btok), PDFrac: f(p.PDFrac), LConc: f(p.LConc),
			MTot: f(p.MTot), MAct: f(p.MAct), CHBM: f(p.CHBM), CBW: f(p.CBW), CFP8: f(p.CFP8),
			Engine: f(p.Engine), LISL: f(p.LISL), LOSL: f(p.LOSL), Src: f(p.Src)})
	}
	return pts
}

func loadCombinedFixture(t testing.TB) combinedFixture {
	t.Helper()
	var fx combinedFixture
	readJSON(t, filepath.Join("testdata", "combined", "parity.json"), &fx)
	if len(fx.Cases) == 0 {
		t.Fatal("the fixture holds no case")
	}
	return fx
}

func loadShippedStage(t testing.TB) *Stage {
	t.Helper()
	s, _, err := LoadShipped(DefaultEpsilon)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestTheShippedArtifactLoadsWithItsFallback(t *testing.T) {
	s, a, err := LoadShipped(DefaultEpsilon)
	if err != nil {
		t.Fatal(err)
	}
	if a.SchemaVersion != ArtifactVersionFallback || len(s.Models) != 4 || len(s.Fallback) != 4 || !s.HasFallback() {
		t.Fatalf("stage %+v artifact %+v", s, a)
	}
	if len(a.Features) != len(PairFeatureNames) || a.RoleCounts.Fit < 1 || a.Epsilon != DefaultEpsilon || len(s.masked) != 3 {
		t.Errorf("artifact %+v masked %v", a, s.masked)
	}
	// The directory form reads the same artifact.
	d, _, err := LoadStage("artifact", DefaultEpsilon)
	if err != nil || len(d.Fallback) != 4 {
		t.Errorf("directory: %v", err)
	}
	if _, _, err := LoadShipped(0.1); err == nil || !strings.Contains(err.Error(), "epsilon") {
		t.Errorf("another epsilon: %v", err)
	}
}

// Parity: the combined probabilities, memberships and front equal what the training pipeline computes with
// the same models on real points, to 1e-6. The points include ones the simulator refused.
func TestTheCombinedStageMatchesTheTrainingPipelineOnRealPoints(t *testing.T) {
	s := loadShippedStage(t)
	fx := loadCombinedFixture(t)
	refused, compared := 0, 0
	for k, c := range fx.Cases {
		pts := fx.points(k)
		for _, f := range Fronts {
			want := c.Fronts[f]
			m, pi, err := s.Front(f, pts)
			if err != nil {
				t.Fatal(err)
			}
			for i := range pts {
				if m[i].Answered != want.Answered[i] || m[i].Estimated == want.Answered[i] {
					t.Errorf("%s front %s point %d: answered %v estimated %v, simulator answered it in Python: %v", c.Case, f, i, m[i].Answered, m[i].Estimated, want.Answered[i])
				}
				if !want.Answered[i] {
					refused++
				}
				if m[i].OnFront != want.OnFront[i] {
					t.Errorf("%s front %s point %d: on front %v, Python %v", c.Case, f, i, m[i].OnFront, want.OnFront[i])
				}
				if math.Abs(m[i].Probability-want.Probability[i]) > 1e-6 {
					t.Errorf("%s front %s point %d: probability %v, Python %v", c.Case, f, i, m[i].Probability, want.Probability[i])
				}
				for j := range pts {
					if i == j {
						continue
					}
					compared++
					if math.Abs(pi[i][j]-want.Pi[i][j]) > 1e-6 {
						t.Errorf("%s front %s pair (%d,%d): %v, Python %v", c.Case, f, i, j, pi[i][j], want.Pi[i][j])
					}
				}
			}
		}
	}
	if refused == 0 {
		t.Error("the fixture has no point the simulator refused: the fallback is not exercised")
	}
	t.Logf("%d ordered pairs compared, %d point-front answers from the fallback", compared, refused)
}

// refuse returns a copy of the points with the simulator's predictions removed from the points picked.
func refuse(pts []Point, pick func(i int) bool) []Point {
	out := append([]Point(nil), pts...)
	for i := range out {
		if pick(i) {
			out[i].LogX, out[i].LogI, out[i].LogT = math.NaN(), math.NaN(), math.NaN()
		}
	}
	return out
}

// randomCombined draws points near the real ones with random predictions, some of them refused.
func randomCombined(t testing.TB, r *rand.Rand, n int, refusedShare float64) []Point {
	base := loadCombinedFixture(t).points(0)
	out := make([]Point, n)
	for i := range out {
		p := base[r.Intn(len(base))]
		p.ID = string(rune('a' + i))
		p.LogX, p.LogI, p.LogT = r.NormFloat64(), r.NormFloat64(), r.NormFloat64()
		if r.Float64() < refusedShare {
			p.LogX, p.LogI, p.LogT = math.NaN(), math.NaN(), math.NaN()
		}
		out[i] = p
	}
	return out
}

// DEC-3: a point the simulator did not answer is placed by the fallback and labelled estimated, on
// every front; the points it did answer are labelled as answered; an artifact without the fallback
// leaves the unanswered point unplaced, as before.
func TestAPointTheSimulatorRefusedIsPlacedByTheFallbackAndLabelledEstimated(t *testing.T) {
	s := loadShippedStage(t)
	pts := loadCombinedFixture(t).points(0)
	cut := refuse(pts, func(i int) bool { return i == 0 || i == 3 })
	for _, f := range Fronts {
		m, pi, err := s.Front(f, cut)
		if err != nil {
			t.Fatal(err)
		}
		for i, p := range cut {
			switch {
			case i == 0 || i == 3:
				if m[i].Answered || !m[i].Estimated || m[i].Probability <= 0 || m[i].Probability > 1 {
					t.Errorf("front %s refused point %d: %+v", f, i, m[i])
				}
			case p.Answered(f):
				if !m[i].Answered || m[i].Estimated {
					t.Errorf("front %s answered point %d: %+v", f, i, m[i])
				}
			}
			for j := range cut {
				if i != j && (math.IsNaN(pi[i][j]) && (!cut[i].Answered(f) || !cut[j].Answered(f)) || s.PairBasis(f, cut[i], cut[j]) == BasisFallback && cut[i].Answered(f) && cut[j].Answered(f)) {
					t.Errorf("front %s pair (%d,%d) was not answered by the fallback", f, i, j)
				}
			}
		}
	}
	// The same stage without its fallback places neither point.
	bare := &Stage{Epsilon: s.Epsilon, Features: s.Features, Models: s.Models}
	m, _, err := bare.Front(FrontXI, cut)
	if err != nil {
		t.Fatal(err)
	}
	if m[0] != (Membership{}) || m[0].Estimated {
		t.Errorf("without a fallback: %+v", m[0])
	}
	if _, _, _, err := bare.Combined(FrontXI, cut[0], cut[1]); err == nil {
		t.Error("a pair the simulator does not answer was estimated without a fallback")
	}
}

// DEC-3 at the pair: what the fallback says about a pair does not depend on any prediction of the
// simulator, only on the configurations and the case.
func TestTheFallbackIgnoresTheSimulatorsPredictions(t *testing.T) {
	s := loadShippedStage(t)
	r := rand.New(rand.NewSource(11))
	for trial := 0; trial < 50; trial++ {
		pts := randomCombined(t, r, 2, 0)
		pts[1].LogX, pts[1].LogI, pts[1].LogT = math.NaN(), math.NaN(), math.NaN() // the simulator refused the second
		moved := append([]Point(nil), pts...)
		moved[0].LogX, moved[0].LogI, moved[0].LogT = 5*r.NormFloat64(), 5*r.NormFloat64(), 5*r.NormFloat64()
		for _, f := range Fronts {
			a, b, basis, err := s.Combined(f, pts[0], pts[1])
			c, d, _, _ := s.Combined(f, moved[0], moved[1])
			if err != nil || basis != BasisFallback || a != c || b != d {
				t.Fatalf("front %s: (%v,%v) became (%v,%v), basis %s, err %v", f, a, b, c, d, basis, err)
			}
		}
	}
}

// DEC-4 for the combined rule: one constant per objective added to every point's predictions of a
// case, some points refused, leaves every probability, membership and front unchanged.
func TestCombinedAddingAConstantToEveryPointsPredictionsChangesNothing(t *testing.T) {
	s := loadShippedStage(t)
	r := rand.New(rand.NewSource(21))
	for trial := 0; trial < 25; trial++ {
		pts := randomCombined(t, r, 9, 0.3)
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
				if m0[i].OnFront != m1[i].OnFront || m0[i].Estimated != m1[i].Estimated || math.Abs(m0[i].Probability-m1[i].Probability) > 1e-6 {
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

// DEC-4 for the combined rule: the two probabilities of a pair sum to at most one, and asking in
// the other order returns the same two numbers swapped, whichever model answers.
func TestCombinedPairProbabilitiesSumToAtMostOneAndDoNotDependOnOrder(t *testing.T) {
	s := loadShippedStage(t)
	r := rand.New(rand.NewSource(22))
	seen := map[Basis]int{}
	for trial := 0; trial < 400; trial++ {
		pts := randomCombined(t, r, 2, 0.4)
		for _, f := range Fronts {
			pq, qp, basis, err := s.Combined(f, pts[0], pts[1])
			if err != nil {
				t.Fatal(err)
			}
			rq, rp, rbasis, _ := s.Combined(f, pts[1], pts[0])
			seen[basis]++
			if pq < 0 || qp < 0 || pq+qp > 1+1e-12 {
				t.Fatalf("front %s: %v + %v", f, pq, qp)
			}
			if pq != rp || qp != rq || basis != rbasis {
				t.Fatalf("front %s: (%v,%v) asked one way, (%v,%v) the other", f, pq, qp, rq, rp)
			}
		}
	}
	if seen[BasisSimulator] == 0 || seen[BasisFallback] == 0 {
		t.Errorf("a model was never exercised: %v", seen)
	}
}

// DEC-4 for the combined rule: where the simulator answers both points, a larger predicted advantage
// of the first never lowers the probability that it dominates (X and I up, T down).
func TestCombinedALargerPredictedAdvantageNeverLowersTheProbabilityItDominates(t *testing.T) {
	s := loadShippedStage(t)
	r := rand.New(rand.NewSource(23))
	for trial := 0; trial < 200; trial++ {
		pts := randomCombined(t, r, 2, 0)
		for _, f := range Fronts {
			x, i, tt := f.uses()
			base, _, _, _ := s.Combined(f, pts[0], pts[1])
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
			after, _, basis, _ := s.Combined(f, better, pts[1])
			rawBase, _ := s.Raw(f, pts[0], pts[1])
			rawAfter, _ := s.Raw(f, better, pts[1])
			if basis != BasisSimulator || rawAfter < rawBase-1e-12 {
				t.Fatalf("front %s: advantage grew by %v and the raw probability fell from %v to %v (basis %s; normalised %v to %v)", f, step, rawBase, rawAfter, basis, base, after)
			}
		}
	}
}

// DEC-5 of the schema: an artifact of version 2 states the fallback models and the features they
// do not read, and the loader refuses one that does not.
func TestAnArtifactWithAFallbackIsRefusedWhenItsFallbackIsWrong(t *testing.T) {
	copyShipped := func(t *testing.T) (string, map[string]any) {
		dir := t.TempDir()
		ents, err := os.ReadDir("artifact")
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range ents {
			b, _ := os.ReadFile(filepath.Join("artifact", e.Name()))
			if err := os.WriteFile(filepath.Join(dir, e.Name()), b, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		var doc map[string]any
		b, _ := os.ReadFile(filepath.Join(dir, "artifact.json"))
		if err := json.Unmarshal(b, &doc); err != nil {
			t.Fatal(err)
		}
		return dir, doc
	}
	for _, c := range []struct {
		name string
		edit func(dir string, m map[string]any)
		want string
	}{
		{"a missing fallback model", func(dir string, m map[string]any) { os.Remove(filepath.Join(dir, FallbackFile(FrontXT))) }, "fallback-XT.json"},
		{"no masked features", func(dir string, m map[string]any) { delete(m, "fallback_masked_features") }, "fallback_masked_features is empty"},
		{"a masked name that is no feature", func(dir string, m map[string]any) { m["fallback_masked_features"] = []any{"dX", "model_id"} }, `"model_id"`},
		{"version 1 with a masked list", func(dir string, m map[string]any) { m["schema_version"] = 1 }, "schema version 1 has no fallback"},
		{"a fallback that takes other inputs", func(dir string, m map[string]any) {
			b, _ := os.ReadFile(filepath.Join("testdata", "artifact", ModelFile(FrontIT)))
			os.WriteFile(filepath.Join(dir, FallbackFile(FrontIT)), b, 0o644)
		}, "takes 54 inputs"},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir, doc := copyShipped(t)
			c.edit(dir, doc)
			writeArtifact(t, dir, doc)
			if _, _, err := LoadStage(dir, DefaultEpsilon); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("got %v, want an error containing %q", err, c.want)
			}
		})
	}
}

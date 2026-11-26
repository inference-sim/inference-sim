package decision

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// nullable is a float that JSON null reads as NaN.
type nullable float64

func (n *nullable) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*n = nullable(math.NaN())
		return nil
	}
	var f float64
	if err := json.Unmarshal(b, &f); err != nil {
		return err
	}
	*n = nullable(f)
	return nil
}

func readJSON(t testing.TB, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

// featuresFixture is testdata/features.json: real points and the features the training pipeline computes for pairs.
type featuresFixture struct {
	Feats  []string `json:"feats"`
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
	Pairs []struct {
		A, B int
		X    []nullable
	} `json:"pairs"`
}

func loadFeatureFixture(t testing.TB) (fixture featuresFixture, pts []Point) {
	readJSON(t, filepath.Join("testdata", "features.json"), &fixture)
	for _, p := range fixture.Points {
		f := func(n nullable) float64 { return float64(n) }
		pts = append(pts, Point{ID: p.ID, LogX: f(p.SimX), LogI: f(p.SimI), LogT: f(p.SimT),
			Kind: int(p.Kind), TP: f(p.TP), EP: f(p.EP), DP: f(p.DP), GPUs: f(p.GPUs), PD: f(p.PD), Spec: f(p.Spec),
			Seqs: f(p.Seqs), Btok: f(p.Btok), PDFrac: f(p.PDFrac), LConc: f(p.LConc),
			MTot: f(p.MTot), MAct: f(p.MAct), CHBM: f(p.CHBM), CBW: f(p.CBW), CFP8: f(p.CFP8),
			Engine: f(p.Engine), LISL: f(p.LISL), LOSL: f(p.LOSL), Src: f(p.Src)})
	}
	return fixture, pts
}

func loadArtifactStage(t testing.TB) *Stage {
	t.Helper()
	s, _, err := LoadStage(filepath.Join("testdata", "artifact"), DefaultEpsilon)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

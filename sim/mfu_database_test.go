package sim

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// --- Helper: synthetic MFUDatabase for testing without bench_data/ ---

func syntheticMFUDatabase() *MFUDatabase {
	return &MFUDatabase{
		attentionConfig: "32-8-128",
		gpu:             "h100",
		prefillData: map[string][]MHAPrefillRow{
			"32-8-128": {
				{SeqLen: 128, MFU: 0.10},
				{SeqLen: 512, MFU: 0.30},
				{SeqLen: 1024, MFU: 0.50},
				{SeqLen: 4096, MFU: 0.70},
			},
		},
		decodeData: map[string][]MHADecodeRow{
			"32-8-128-tp1": {
				{BatchSize: 1, KVLen: 128, MFU: 0.05},
				{BatchSize: 1, KVLen: 512, MFU: 0.10},
				{BatchSize: 8, KVLen: 128, MFU: 0.15},
				{BatchSize: 8, KVLen: 512, MFU: 0.20},
				{BatchSize: 32, KVLen: 128, MFU: 0.25},
				{BatchSize: 32, KVLen: 512, MFU: 0.30},
			},
		},
		gemmData: []GEMMRow{
			{M: 1, K: 4096, N: 4096, MFU: 0.05},
			{M: 8, K: 4096, N: 4096, MFU: 0.20},
			{M: 32, K: 4096, N: 4096, MFU: 0.40},
			{M: 128, K: 4096, N: 4096, MFU: 0.60},
		},
	}
}

// --- lerp ---

func TestLerp(t *testing.T) {
	tests := []struct {
		a, b, param, want float64
	}{
		{0, 10, 0, 0},      // t=0 → a
		{0, 10, 1, 10},     // t=1 → b
		{0, 10, 0.5, 5},    // midpoint
		{2, 8, 0.25, 3.5},  // quarter
		{10, 10, 0.5, 10},  // a==b
	}
	for _, tc := range tests {
		got := lerp(tc.a, tc.b, tc.param)
		if math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("lerp(%v, %v, %v) = %v, want %v", tc.a, tc.b, tc.param, got, tc.want)
		}
	}
}

// --- bracketIndex ---

func TestBracketIndex(t *testing.T) {
	sorted := []int{1, 4, 8, 16, 32}
	tests := []struct {
		target     int
		wantLo, wantHi int
	}{
		{0, 0, 0},   // below min → clamp
		{1, 0, 0},   // exact min
		{4, 1, 1},   // exact match
		{32, 4, 4},  // exact max
		{100, 4, 4}, // above max → clamp
		{5, 1, 2},   // between 4 and 8
		{10, 2, 3},  // between 8 and 16
		{2, 0, 1},   // between 1 and 4
	}
	for _, tc := range tests {
		lo, hi := bracketIndex(sorted, tc.target)
		if lo != tc.wantLo || hi != tc.wantHi {
			t.Errorf("bracketIndex(%v, %d) = (%d, %d), want (%d, %d)",
				sorted, tc.target, lo, hi, tc.wantLo, tc.wantHi)
		}
	}
}

// --- computeAttentionConfig ---

func TestComputeAttentionConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  ModelConfig
		want string
	}{
		{
			name: "standard MHA (heads==kv_heads)",
			cfg:  ModelConfig{NumHeads: 32, NumKVHeads: 32, HiddenDim: 4096},
			want: "32-32-128", // 4096/32 = 128
		},
		{
			name: "GQA (kv_heads < heads)",
			cfg:  ModelConfig{NumHeads: 32, NumKVHeads: 8, HiddenDim: 4096},
			want: "32-8-128",
		},
		{
			name: "zero kv_heads defaults to num_heads",
			cfg:  ModelConfig{NumHeads: 16, NumKVHeads: 0, HiddenDim: 2048},
			want: "16-16-128",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := computeAttentionConfig(tc.cfg)
			if got != tc.want {
				t.Errorf("computeAttentionConfig(%+v) = %q, want %q", tc.cfg, got, tc.want)
			}
		})
	}
}

// --- parseAttentionConfig ---

func TestParseAttentionConfig(t *testing.T) {
	shape := parseAttentionConfig("32-8-128")
	if shape.NumHeads != 32 || shape.NumKVHeads != 8 || shape.HeadDim != 128 {
		t.Errorf("parseAttentionConfig(32-8-128) = %+v, want {32, 8, 128}", shape)
	}
	if shape.ConfigKey != "32-8-128" {
		t.Errorf("ConfigKey = %q, want %q", shape.ConfigKey, "32-8-128")
	}
}

// --- euclideanDistance ---

func TestEuclideanDistance(t *testing.T) {
	a := AttentionShape{NumHeads: 32, NumKVHeads: 8, HeadDim: 128}
	b := AttentionShape{NumHeads: 32, NumKVHeads: 8, HeadDim: 128}
	if d := euclideanDistance(a, b); d != 0 {
		t.Errorf("same shapes should have distance 0, got %f", d)
	}

	c := AttentionShape{NumHeads: 64, NumKVHeads: 8, HeadDim: 128}
	d := euclideanDistance(a, c)
	if d != 32 {
		t.Errorf("expected distance 32, got %f", d)
	}
}

// --- findNearestConfig ---

func TestFindNearestConfig(t *testing.T) {
	available := []AttentionShape{
		{NumHeads: 32, NumKVHeads: 32, HeadDim: 128, ConfigKey: "32-32-128"},
		{NumHeads: 32, NumKVHeads: 8, HeadDim: 128, ConfigKey: "32-8-128"},
		{NumHeads: 64, NumKVHeads: 8, HeadDim: 128, ConfigKey: "64-8-128"},
	}

	// Exact match
	got := findNearestConfig("32-8-128", available)
	if got != "32-8-128" {
		t.Errorf("exact match: got %q, want 32-8-128", got)
	}

	// Nearest to 28-4-128 should be 32-8-128 (distance ~5.66)
	got = findNearestConfig("28-4-128", available)
	if got != "32-8-128" {
		t.Errorf("nearest to 28-4-128: got %q, want 32-8-128", got)
	}

	// Empty available → return target unchanged
	got = findNearestConfig("99-99-99", nil)
	if got != "99-99-99" {
		t.Errorf("empty available: got %q, want 99-99-99", got)
	}
}

// --- GetAttnPrefillMFU ---

func TestGetAttnPrefillMFU_Interpolation(t *testing.T) {
	db := syntheticMFUDatabase()

	// Exact grid point
	mfu := db.GetAttnPrefillMFU(512)
	if math.Abs(mfu-0.30) > 1e-6 {
		t.Errorf("exact 512: got %f, want 0.30", mfu)
	}

	// Below min → clamp to first
	mfu = db.GetAttnPrefillMFU(1)
	if math.Abs(mfu-0.10) > 1e-6 {
		t.Errorf("below min: got %f, want 0.10", mfu)
	}

	// Above max → clamp to last
	mfu = db.GetAttnPrefillMFU(10000)
	if math.Abs(mfu-0.70) > 1e-6 {
		t.Errorf("above max: got %f, want 0.70", mfu)
	}

	// Midpoint between 512 (0.30) and 1024 (0.50) → 0.40
	mfu = db.GetAttnPrefillMFU(768)
	if math.Abs(mfu-0.40) > 1e-6 {
		t.Errorf("midpoint 768: got %f, want 0.40", mfu)
	}

	// Monotonicity: more tokens → higher MFU (for this synthetic data)
	prev := 0.0
	for _, seq := range []int{64, 128, 256, 512, 1024, 2048, 4096, 8192} {
		m := db.GetAttnPrefillMFU(seq)
		if m < prev {
			t.Errorf("monotonicity violated: MFU(%d)=%f < MFU(prev)=%f", seq, m, prev)
		}
		prev = m
	}
}

// --- GetAttnDecodeMFU ---

func TestGetAttnDecodeMFU_Interpolation(t *testing.T) {
	db := syntheticMFUDatabase()

	// Exact grid point
	mfu := db.GetAttnDecodeMFU(8, 128, 1)
	if math.Abs(mfu-0.15) > 1e-6 {
		t.Errorf("exact (8,128,tp1): got %f, want 0.15", mfu)
	}

	// Interpolated between batch=8 and batch=32 at kvLen=128
	// batch=8 → 0.15, batch=32 → 0.25; midpoint batch=20 → ~0.1875
	mfu = db.GetAttnDecodeMFU(20, 128, 1)
	if mfu < 0.15 || mfu > 0.25 {
		t.Errorf("interpolated (20,128,tp1): got %f, want in [0.15, 0.25]", mfu)
	}
}

// --- GetGEMMmfu ---

func TestGetGEMMmfu_Interpolation(t *testing.T) {
	db := syntheticMFUDatabase()

	// Exact grid point
	mfu := db.GetGEMMmfu(8, 4096, 4096)
	if math.Abs(mfu-0.20) > 1e-6 {
		t.Errorf("exact (8,4096,4096): got %f, want 0.20", mfu)
	}

	// Below min → clamp
	mfu = db.GetGEMMmfu(1, 4096, 4096)
	if math.Abs(mfu-0.05) > 1e-6 {
		t.Errorf("min clamp: got %f, want 0.05", mfu)
	}

	// Above max → clamp
	mfu = db.GetGEMMmfu(1000, 4096, 4096)
	if math.Abs(mfu-0.60) > 1e-6 {
		t.Errorf("max clamp: got %f, want 0.60", mfu)
	}
}

// --- NewMFUDatabase error paths ---

func TestNewMFUDatabase_MissingDirectory(t *testing.T) {
	cfg := ModelConfig{NumHeads: 32, NumKVHeads: 8, HiddenDim: 4096}
	_, err := NewMFUDatabase(cfg, "/nonexistent/path", "h100")
	if err == nil {
		t.Fatal("expected error for missing bench_data directory")
	}
}

func TestNewMFUDatabase_EmptyDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	// Create the expected directory structure but with no CSV files
	mhaDir := filepath.Join(tmpDir, "mha", "prefill", "h100")
	if err := os.MkdirAll(mhaDir, 0o755); err != nil {
		t.Fatal(err)
	}

	cfg := ModelConfig{NumHeads: 32, NumKVHeads: 8, HiddenDim: 4096}
	_, err := NewMFUDatabase(cfg, tmpDir, "h100")
	// Should fail because there are no CSV files with data
	if err == nil {
		t.Fatal("expected error for empty bench_data directory")
	}
}

// --- CSV loading with real bench_data ---

func TestNewMFUDatabase_WithRealBenchData(t *testing.T) {
	benchPath := filepath.Join("..", "bench_data")
	if _, err := os.Stat(benchPath); os.IsNotExist(err) {
		t.Skip("bench_data not available")
	}

	cfg := ModelConfig{
		NumHeads:   32,
		NumKVHeads: 8,
		HiddenDim:  4096,
	}
	db, err := NewMFUDatabase(cfg, benchPath, "h100")
	if err != nil {
		t.Fatalf("NewMFUDatabase: %v", err)
	}

	// Verify data was loaded
	if len(db.prefillData) == 0 {
		t.Error("no prefill data loaded")
	}
	if len(db.decodeData) == 0 {
		t.Error("no decode data loaded")
	}
	if len(db.gemmData) == 0 {
		t.Error("no GEMM data loaded")
	}

	// MFU values should be positive and bounded
	mfu := db.GetAttnPrefillMFU(512)
	if mfu <= 0 || mfu > 1 {
		t.Errorf("prefill MFU(512) = %f, expected in (0, 1]", mfu)
	}

	mfu = db.GetGEMMmfu(32, 4096, 4096)
	if mfu <= 0 || mfu > 1 {
		t.Errorf("GEMM MFU(32,4096,4096) = %f, expected in (0, 1]", mfu)
	}
}

package latency_test

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// TestDPEPConfig_RealFixtures is the #1417 acceptance test: it parses the four
// real MoE config.json fixtures through the production GetModelConfig path and
// asserts the parsed expert fields plus the DP/EP group-size helpers across
// several (TP, DP, EnableExpertParallel) triples.
//
// The fixtures are real upstream HuggingFace configs committed under
// model_configs/. Expected values were verified against upstream config.json on
// 2026-06-10 (design §2.2).
//
// numMoELayersGap records, but does NOT assert as parity, the known dense-layer
// placement gap: BLIS does not yet parse first_k_dense_replace /
// decoder_sparse_step / mlp_only_layers, so for those models BLIS treats every
// layer as MoE and over-counts MoE layers. This is documented (design §2.2) and
// resolved by the future distill-model-config effort, not this PR. The field
// keeps the fixture honest — it does not claim layer-count parity.
func TestDPEPConfig_RealFixtures(t *testing.T) {
	type fixture struct {
		name    string
		dir     string
		experts int // NumLocalExperts (expert count after alias resolution)
		perTok  int // NumExpertsPerTok
		moeFFN  int // MoEExpertFFNDim (0 = falls back to IntermediateDim)
		sharedF int // SharedExpertFFNDim (0 = no/unparsed shared expert)
		// numMoELayersGapNote records, as a human-readable note (NOT asserted),
		// the known dense-layer-placement gap: where BLIS treats every layer as
		// MoE but the model upstream has some dense layers, BLIS over-counts MoE
		// layers. Logged via t.Logf to keep the fixture honest about parity.
		numMoELayersGapNote string
	}

	fixtures := []fixture{
		{
			name: "mixtral-8x7b", dir: "mixtral-8x7b-v0.1",
			experts: 8, perTok: 2, moeFFN: 0, sharedF: 0,
			numMoELayersGapNote: "none (uniform MoE, no shared expert)",
		},
		{
			name: "deepseek-v2-lite", dir: "deepseek-v2-lite",
			experts: 64, perTok: 6, moeFFN: 1408, sharedF: 2816, // 2 × 1408
			numMoELayersGapNote: "first_k_dense_replace=1 → 1 of 27 layers is dense; BLIS over-counts MoE layers by 1",
		},
		{
			name: "qwen3-30b-a3b", dir: "qwen3-30b-a3b",
			experts: 128, perTok: 8, moeFFN: 768, sharedF: 0,
			numMoELayersGapNote: "decoder_sparse_step=1, mlp_only_layers=[] → all-MoE, no gap",
		},
		{
			name: "llama-4-scout", dir: "llama-4-scout-17b-16e-instruct-fp8-dynamic",
			experts: 16, perTok: 1, moeFFN: 0, sharedF: 0, // shared-expert dim not exposed → parser gap
			numMoELayersGapNote: "shared expert present upstream but no config field exposes its dim → SharedExpertFFNDim=0 (parser gap)",
		},
	}

	// (TP, DP, EP) triples and the expected flattened-group / EP sizes for an MoE model.
	triples := []struct {
		tp, dp       int
		ep           bool
		wantMoEGroup int
		wantEP       int
	}{
		{1, 1, false, 1, 1},
		{2, 1, false, 2, 1},
		{2, 1, true, 2, 2},
		{2, 2, false, 4, 1}, // EP-off flattened group = TP·DP
		{2, 2, true, 4, 4},  // EP-on uses the same flattened group as EP
		{4, 2, true, 8, 8},
	}

	for _, f := range fixtures {
		t.Run(f.name, func(t *testing.T) {
			path := filepath.Join("..", "..", "model_configs", f.dir, "config.json")
			mc, err := latency.GetModelConfig(path)
			if err != nil {
				t.Fatalf("GetModelConfig(%s): %v", path, err)
			}

			// Parsed expert fields (the architecture facts the helpers depend on).
			if mc.NumLocalExperts != f.experts {
				t.Errorf("NumLocalExperts: got %d, want %d", mc.NumLocalExperts, f.experts)
			}
			if mc.NumExpertsPerTok != f.perTok {
				t.Errorf("NumExpertsPerTok: got %d, want %d", mc.NumExpertsPerTok, f.perTok)
			}
			if mc.MoEExpertFFNDim != f.moeFFN {
				t.Errorf("MoEExpertFFNDim: got %d, want %d", mc.MoEExpertFFNDim, f.moeFFN)
			}
			if mc.SharedExpertFFNDim != f.sharedF {
				t.Errorf("SharedExpertFFNDim: got %d, want %d", mc.SharedExpertFFNDim, f.sharedF)
			}

			// All four fixtures are MoE (NumLocalExperts > 1).
			for _, tr := range triples {
				hw := sim.NewModelHardwareConfig(*mc, sim.HardwareCalib{}, f.name, "H100",
					tr.tp, tr.dp, tr.ep, "", "trained-physics", 0)
				if got := hw.EffectiveMoEGroupSize(); got != tr.wantMoEGroup {
					t.Errorf("(TP=%d,DP=%d,EP=%t) EffectiveMoEGroupSize: got %d, want %d",
						tr.tp, tr.dp, tr.ep, got, tr.wantMoEGroup)
				}
				if got := hw.EffectiveEP(); got != tr.wantEP {
					t.Errorf("(TP=%d,DP=%d,EP=%t) EffectiveEP: got %d, want %d",
						tr.tp, tr.dp, tr.ep, got, tr.wantEP)
				}
			}

			t.Logf("known numMoELayers gap: %s", f.numMoELayersGapNote)
		})
	}
}

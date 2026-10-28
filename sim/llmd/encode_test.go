package llmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// runEncode runs llm-d's sample E/P/D configuration (file) on one encode, one prefill and two
// decode pods, every other request carrying an image.
func runEncode(t *testing.T, file string, render bool, encoders int) (runResult, map[string]string) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(forkConfigDir(t), file))
	if err != nil {
		t.Fatal(err)
	}
	config := string(raw)
	if encoders > 1 {
		// The encode profile picks up to two encoders.
		config = strings.Replace(config, "plugins:\n", "plugins:\n- type: max-score-picker\n  name: two-encoders\n  parameters:\n    maxNumOfEndpoints: 2\n", 1)
		config = strings.Replace(config, "- name: encode\n  plugins:\n  - pluginRef: encode-filter\n",
			"- name: encode\n  plugins:\n  - pluginRef: encode-filter\n  - pluginRef: two-encoders\n", 1)
	}
	if render {
		// A vLLM render token-producer: the simulated renderer renders the chat message
		// (renderChat), media items included.
		config = strings.Replace(config, "plugins:\n", "plugins:\n- type: token-producer\n  parameters:\n    modelName: test-model\n"+
			"    vllm:\n      url: http://render.blis.local:8000\n", 1)
	}
	reqs := testRequests(30, 9, 5000, 0)
	for i, r := range reqs {
		if i%2 == 0 {
			r.ImageTokenCount = len(r.InputTokens) / 2 // the prompt's second half is the image's
		}
	}
	dep := testDeployment(3 + encoders)
	dep.EncodeInstances, dep.PrefillInstances, dep.DecodeInstances = encoders, 1, 2
	dep.TraceLevel = "decisions"
	dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 + tokens/50 }
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", PodLabels: roleLabels, EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)})
	encoded := map[string]string{}
	rendered := 0
	gw.observe = func(kind, detail string) {
		if kind == "render" {
			rendered++
			if !render {
				t.Errorf("rendered %s without a render token-producer", detail)
			}
		}
		if kind == "encode" {
			f := strings.Fields(detail)
			encoded[f[0]] = f[1]
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if render && rendered == 0 {
		t.Error("the vLLM render token-producer never rendered a multimodal request")
	}
	return runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}, encoded
}

// TestEncode_RouterPlacesMultimodalRequestsOnEncoders: with an encode pool and llm-d's E/P/D
// configuration, the router's always-disagg-multimodal-decider sends each request carrying media
// (a chat message with an image part, rendered with its placeholder span) to an encode pod, and
// no text-only request; prefill and decode never run on the encode pod; every request completes
// (encoding costs no simulated time, as BLIS's own encode stage); the run is deterministic.
func TestEncode_RouterPlacesMultimodalRequestsOnEncoders(t *testing.T) {
	for _, render := range []bool{false, true} {
		t.Run(fmt.Sprintf("vllm-render=%v", render), func(t *testing.T) { checkEncode(t, render, 1) })
	}
	// An encode profile that picks two encoders: every one the router chose is recorded, in its
	// order (x-llm-d-encoder-hosts-ports), none dropped.
	t.Run("two-encoders", func(t *testing.T) { checkEncode(t, false, 2) })
}

func checkEncode(t *testing.T, render bool, encoders int) {
	res, encoded := runEncode(t, "sim-e-p-d-epp-config.yaml", render, encoders)
	assertConservation(t, res, len(res.reqs))
	if res.metrics.CompletedRequests != len(res.reqs) {
		t.Fatalf("completed %d of %d", res.metrics.CompletedRequests, len(res.reqs))
	}
	role := res.cs.PoolMembership()
	for _, r := range res.reqs {
		enc, ok := encoded[r.ID]
		switch {
		case r.IsMultimodal() && !ok:
			t.Errorf("%s carries an image but was not encoded", r.ID)
		case !r.IsMultimodal() && ok:
			t.Errorf("%s is text only but was encoded on %s", r.ID, enc)
		}
		if !ok {
			continue
		}
		ids := strings.Split(enc, ",")
		if len(ids) != encoders {
			t.Errorf("%s encoded on %v; want %d encoders", r.ID, ids, encoders)
		}
		seen := map[string]bool{}
		for _, id := range ids {
			if role[id] != cluster.PoolRoleEncode || seen[id] {
				t.Errorf("%s encoded on %v: want distinct encode pods", r.ID, ids)
			}
			seen[id] = true
		}
	}
	for _, p := range res.cs.ParentRequests() {
		if role[string(p.PrefillInstanceID)] == cluster.PoolRoleEncode || role[string(p.DecodeInstanceID)] == cluster.PoolRoleEncode {
			t.Errorf("%s prefilled on %s, decoded on %s: an encode pod", p.ID, p.PrefillInstanceID, p.DecodeInstanceID)
		}
		if r := p.OriginalRequest; r.IsMultimodal() {
			var got []string
			for _, id := range p.EncodeInstanceIDs {
				got = append(got, string(id))
			}
			if strings.Join(got, ",") != encoded[r.ID] {
				t.Errorf("%s: the parent records encoders %v, the router chose %q", r.ID, got, encoded[r.ID])
			}
		}
	}
	for _, r := range res.reqs {
		if role[r.AssignedInstance] == cluster.PoolRoleEncode {
			t.Errorf("%s served on the encode pod %s", r.ID, r.AssignedInstance)
		}
	}
	// The decision trace has one encode record per chosen encoder, in the router's order.
	byReq := map[string][]string{}
	for _, rec := range res.cs.Trace().EncodeRoutings {
		byReq[rec.ParentRequestID] = append(byReq[rec.ParentRequestID], rec.ChosenInstance)
	}
	for id, enc := range encoded {
		if got := strings.Join(byReq[id], ","); got != enc {
			t.Errorf("%s: the trace records encoders %q, the router chose %q", id, got, enc)
		}
	}
	if len(byReq) != len(encoded) {
		t.Errorf("the trace has encode records for %d requests, %d were encoded", len(byReq), len(encoded))
	}
	again, encoded2 := runEncode(t, "sim-e-p-d-epp-config.yaml", render, encoders)
	if len(encoded2) != len(encoded) || again.metrics.CompletedRequests != res.metrics.CompletedRequests {
		t.Fatalf("two identical runs differ: %d vs %d encoded", len(encoded), len(encoded2))
	}
	for id, enc := range encoded {
		if encoded2[id] != enc {
			t.Fatalf("%s encoded on %s, then on %s", id, enc, encoded2[id])
		}
	}
	t.Logf("%d of %d requests encoded", len(encoded), len(res.reqs))
}

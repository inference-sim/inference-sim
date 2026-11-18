package cmd

import (
	"math"
	"testing"
)

// instanceWindows returns each instance's measured_window block.
func instanceWindows(t *testing.T, d capDoc) map[string]map[string]any {
	t.Helper()
	out := map[string]map[string]any{}
	for id, in := range d.instances() {
		mw, ok := in["measured_window"].(map[string]any)
		if !ok {
			t.Fatalf("instance %s states no measured_window", id)
		}
		out[id] = mw
	}
	return out
}

// In a disaggregated run an instance's records are sub-requests (a prefill leg and a decode leg),
// which are not what a user measured. Each instance's measured_window states the user-level figures
// the cluster's own use (inference-sim#2218): a decode instance the requests it completed, and a
// prefill instance the requests it prefilled, as those users saw them.
func TestPDInstancesStateUserLevelMeasuredWindow(t *testing.T) {
	t.Parallel()
	d := capExampleRun(t, capExamplesDir, "run", "--scenario", "pd-unequal-tp.yaml", "--num-instances", "5",
		"--prefill-instances", "4", "--decode-instances", "1", "--rate", "5", "--num-requests", "20", "--seed", "3")
	windows := instanceWindows(t, d)
	requests := int(d.num("window.requests"))
	if requests != 20 {
		t.Fatalf("fixture: window holds %d requests, want 20", requests)
	}
	var completions, userRequests int
	for id, mw := range windows {
		completions += int(rowNum(mw, "measured_completed_requests"))
		// instance_0 to instance_3 are the prefill pool, instance_4 the decode instance.
		role := "prefill"
		if id == "instance_4" {
			role = "decode"
		}
		if mw["role"] != role {
			t.Errorf("instance %s states role %v, want %s", id, mw["role"], role)
		}
		switch role {
		case "prefill":
			userRequests += int(rowNum(mw, "measured_user_requests"))
			if rowNum(mw, "measured_completed_requests") != 0 || rowNum(mw, "output_tok_s") != 0 {
				t.Errorf("prefill instance %s completes %v requests and emits %v tok/s, want none", id, mw["measured_completed_requests"], mw["output_tok_s"])
			}
			if n := rowNum(mw, "measured_user_requests"); n > 0 {
				// The user waited for the decode leg too, so the instance's E2E is far above its TTFT.
				if e2e, ttft := rowNum(mw, "e2e_p99_ms"), rowNum(mw, "ttft_p99_ms"); e2e <= 2*ttft {
					t.Errorf("prefill instance %s: e2e_p99 %.2f ms is the prefill leg's, not the user's (ttft_p99 %.2f ms)", id, e2e, ttft)
				}
				if _, ok := mw["tpot_p99_ms"]; !ok {
					t.Errorf("prefill instance %s states no tpot", id)
				}
			}
		case "decode":
			// The decode instance completed every user request, so its figures are the cluster's.
			for _, k := range []string{"ttft_p50_ms", "ttft_p99_ms", "e2e_p50_ms", "e2e_p99_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms", "itl_p50_ms", "itl_p99_ms"} {
				if got, want := rowNum(mw, k), d.num("metrics."+k+".value"); math.Abs(got-want) > 1e-9 {
					t.Errorf("decode instance %s %s = %v, the cluster's is %v", id, k, got, want)
				}
			}
		}
	}
	if completions != requests {
		t.Errorf("the instances' measured completions add to %d, the window holds %d requests", completions, requests)
	}
	if userRequests != requests {
		t.Errorf("the prefill instances' user requests add to %d, want the window's %d", userRequests, requests)
	}
}

// With two decode instances each states only the user requests it completed, and together they
// are the cluster's: the mean of the cluster's E2E is the completion-weighted mean of theirs.
func TestPDDecodeInstancesPartitionTheUserRequests(t *testing.T) {
	t.Parallel()
	d := capExampleRun(t, capExamplesDir, "run", "--scenario", "pd-unequal-dp.yaml", "--num-instances", "6",
		"--prefill-instances", "4", "--decode-instances", "2", "--rate", "5", "--num-requests", "20", "--seed", "3")
	var n, e2e float64
	served := 0
	for _, id := range []string{"instance_4", "instance_5"} {
		mw := instanceWindows(t, d)[id]
		c := rowNum(mw, "measured_completed_requests")
		n += c
		e2e += c * rowNum(mw, "e2e_mean_ms")
		if c > 0 {
			served++
		}
	}
	if int(n) != int(d.num("window.requests")) || served != 2 {
		t.Fatalf("decode instances completed %v requests (%d serving), want all %v split over both", n, served, d.num("window.requests"))
	}
	if got, want := e2e/n, d.num("metrics.e2e_mean_ms.value"); math.Abs(got-want) > 1e-6*want {
		t.Errorf("completion-weighted decode-instance mean E2E %.4f ms, the cluster's is %.4f ms", got, want)
	}
}

// A colocated instance's figures are unchanged and gain the time-per-output-token and
// inter-token figures: without a disaggregated role they state neither role nor user requests.
func TestColocatedInstancesStateTPOTAndITL(t *testing.T) {
	t.Parallel()
	d := capExampleRun(t, capExamplesDir, "run", "--scenario", "node-pools.yaml", "--num-instances", "4",
		"--policy-config", capExamplesDir+"/node-pools-bundle.yaml", "--rate", "4", "--num-requests", "40")
	for id, mw := range instanceWindows(t, d) {
		if _, ok := mw["role"]; ok {
			t.Errorf("colocated instance %s states a P/D role", id)
		}
		if rowNum(mw, "measured_completed_requests") == 0 {
			continue
		}
		for _, k := range []string{"tpot_p50_ms", "tpot_p99_ms", "itl_p50_ms", "itl_p99_ms"} {
			if v, ok := mw[k].(float64); !ok || v <= 0 {
				t.Errorf("instance %s %s = %v, want a positive figure", id, k, mw[k])
			}
		}
	}
}

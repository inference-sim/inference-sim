// Original to blis-sglang.

package config

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// The surface list (surface.json, X9) and the start-up refusal of a setting (accepted.go) both name
// the open issue that models a feature, and a user reads one or the other. They are generated and
// hand-kept apart, so this test holds them to one answer (SGL-6, X28): for every surface item that
// one ServerArgs setting selects, the issue the list cites is the issue the refusal of that setting
// at that value cites. Items an externally loaded class or file selects ("dynamic" storage, a
// customized dispatcher) are not selected by a value of a closed set and are left to their parent
// issue (inference-sim/inference-sim#2460).
func TestSurfaceListCitesTheIssueTheSettingRefusalCites(t *testing.T) {
	raw, err := os.ReadFile("../surface.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Items []struct {
			ID       string `json:"id"`
			Category string `json:"category"`
			Name     string `json:"name"`
			Status   string `json:"status"`
			Issue    string `json:"issue"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	fieldOf := map[string]string{
		"hicache_storage":      "hicache_storage_backend",
		"transfer_backend":     "disaggregation_transfer_backend",
		"moe_a2a_backend":      "moe_a2a_backend",
		"pd_allocation_policy": "disaggregation_decode_allocation_policy",
		"spec_algo":            "speculative_algorithm",
	}
	external := map[string]bool{"hicache_storage:dynamic": true, "moe_a2a_backend:customized": true}
	known := map[string]bool{}
	for _, n := range FieldNames() {
		known[n] = true
	}
	checked := 0
	for _, it := range doc.Items {
		var field, value string
		switch {
		case it.Category == "hicache_option":
			f, v, ok := strings.Cut(it.Name, "=")
			if !ok {
				t.Fatalf("%s: a hicache_option without a value", it.ID)
			}
			field, value = f, v
		case fieldOf[it.Category] != "":
			field, value = fieldOf[it.Category], it.Name
		default:
			continue
		}
		if it.Status != "refused" || external[it.ID] {
			continue
		}
		if !known[field] {
			t.Errorf("%s: names setting %q, which ServerArgs does not have", it.ID, field)
			continue
		}
		if AcceptsValue(field, value) || (it.Category == "moe_a2a_backend" && value == "none") {
			continue // the setting's default or an accepted value, not a refusal
		}
		want := "https://github.com/" + strings.Replace(ValueRefusalIssue(field, value), "#", "/issues/", 1)
		if it.Issue != want {
			t.Errorf("%s: the list cites %s, but refusing %s=%s cites %s", it.ID, it.Issue, field, value, want)
		}
		checked++
	}
	if checked < 45 {
		t.Fatalf("only %d surface items were held to a setting refusal", checked)
	}
}

// The worker class and the argument-resolution handler an algorithm selects are refused with the
// algorithm (the surface list cites the issue the refusal of speculative_algorithm cites). The
// EAGLE family is accepted at the setting, and its items cite the issue that models it.
func TestSurfaceSpeculativeWorkersCiteTheirAlgorithmsIssue(t *testing.T) {
	raw, err := os.ReadFile("../surface.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Items []struct {
			ID    string `json:"id"`
			Issue string `json:"issue"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	algorithmOf := map[string]string{
		"spec_worker:DFlashWorkerV2":                  "DFLASH",
		"spec_worker:UnoWorkerV2":                     "UNO",
		"spec_worker:DSparkWorkerV2":                  "DSPARK",
		"spec_worker:FrozenKVMTPWorkerV2":             "FROZEN_KV_MTP",
		"spec_worker:StandaloneWorkerV2":              "STANDALONE",
		"spec_worker:NGRAMWorker":                     "NGRAM",
		"spec_running_requests:_handle_dflash":        "DFLASH",
		"spec_running_requests:_handle_dspark":        "DSPARK",
		"spec_running_requests:_handle_frozen_kv_mtp": "FROZEN_KV_MTP",
		"spec_running_requests:_handle_ngram":         "NGRAM",
		"drafter_arch:DeepseekV4ForCausalLMDSpark":    "DSPARK",
	}
	seen := map[string]bool{}
	for _, it := range doc.Items {
		algo, ok := algorithmOf[it.ID]
		if !ok {
			continue
		}
		seen[it.ID] = true
		want := "https://github.com/" + strings.Replace(ValueRefusalIssue("speculative_algorithm", algo), "#", "/issues/", 1)
		if it.Issue != want {
			t.Errorf("%s: the list cites %s, but refusing speculative_algorithm=%s cites %s", it.ID, it.Issue, algo, want)
		}
	}
	for id := range algorithmOf {
		if !seen[id] {
			t.Errorf("%s is not in the surface list", id)
		}
	}
}

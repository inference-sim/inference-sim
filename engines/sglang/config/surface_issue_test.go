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
	if checked < 40 {
		t.Fatalf("only %d surface items were held to a setting refusal", checked)
	}
}

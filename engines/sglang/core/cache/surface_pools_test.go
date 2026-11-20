// Original to blis-sglang.

package cache

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// The surface list (surface.json, X9) and the start-up refusal (pool_classes.json, SGL-6) both
// say which allocators, KV pools and request pools the port models. They are generated and
// hand-kept apart, so this test holds them to one answer (blis-sglang#44): a class the refusal
// models is modelled in the list; every other in-scope class is refused in the list citing the
// issue the refusal cites, not a parent stage issue; and an out-of-scope class (a hardware backend
// SGLang does not run on a CUDA target) is still refused by name at start-up.
func TestSurfaceListAgreesWithPoolRefusal(t *testing.T) {
	raw, err := os.ReadFile("../../surface.json")
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
	wantIssue := "https://github.com/" + RefusedPoolsIssue[:strings.Index(RefusedPoolsIssue, "#")] + "/issues/" +
		RefusedPoolsIssue[strings.Index(RefusedPoolsIssue, "#")+1:]
	checked := 0
	for _, it := range doc.Items {
		if it.Category != "allocator" && it.Category != "kv_pool" && it.Category != "req_pool" {
			continue
		}
		checked++
		pc, listed := poolClasses[it.Name]
		switch it.Status {
		case "modelled":
			if !listed || pc.Status != "modelled" {
				t.Errorf("%s: the list says modelled, but the start-up check %s", it.ID, describe(listed, pc))
			}
		case "refused":
			if listed && pc.Status == "modelled" {
				t.Errorf("%s: the list refuses it, but the start-up check models it", it.ID)
			}
			if it.Issue != wantIssue {
				t.Errorf("%s: refused citing %s, want %s (the issue the start-up refusal cites)", it.ID, it.Issue, wantIssue)
			}
		case "out_of_scope":
			if listed && pc.Status != "refused" {
				t.Errorf("%s: out of scope in the list, but the start-up check %s", it.ID, describe(listed, pc))
			}
		default:
			t.Errorf("%s: status %q", it.ID, it.Status)
		}
	}
	if checked == 0 {
		t.Fatal("surface.json holds no allocator, KV pool or request pool")
	}
	// Every class the refusal knows is on the list, so neither side can drop one unnoticed.
	onList := map[string]bool{}
	for _, it := range doc.Items {
		onList[it.Name] = true
	}
	for name := range poolClasses {
		if !onList[name] {
			t.Errorf("%s: the start-up check knows it, the surface list does not", name)
		}
	}
}

func describe(listed bool, pc poolClassEntry) string {
	if !listed {
		return "does not list it"
	}
	return "lists it as " + pc.Status
}

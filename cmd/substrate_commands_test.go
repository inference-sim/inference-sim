package cmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
	"github.com/inference-sim/inference-sim/internal/substratedata"
)

// invariants --json prints every record of the registry the pinned substrate embeds, in its
// order, with the registry's own keys.
func TestInvariantsPrintsThePinnedRegistry(t *testing.T) {
	reg, err := invariants.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := printInvariants(&out, true); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Invariants []map[string]any `json:"invariants"`
	}
	dec := json.NewDecoder(&out)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got.Invariants) != len(reg.Invariants) || len(got.Invariants) < 100 {
		t.Fatalf("printed %d invariants; the registry holds %d", len(got.Invariants), len(reg.Invariants))
	}
	for i, inv := range reg.Invariants {
		if got.Invariants[i]["id"] != inv.ID || got.Invariants[i]["statement"] != inv.Statement {
			t.Fatalf("record %d is %v, want %s", i, got.Invariants[i]["id"], inv.ID)
		}
	}
	sim3, ok := reg.Get("SIM-3")
	if !ok {
		t.Fatal("the registry has no SIM-3")
	}
	var text bytes.Buffer
	if err := printInvariants(&text, false); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text.String(), "SIM-3 ") || !strings.Contains(text.String(), sim3.Status) {
		t.Errorf("the text form lacks SIM-3:\n%s", text.String()[:200])
	}
}

func resolved(t *testing.T, r catalogResolution, err error) string {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	if r.Identity == nil {
		return ""
	}
	return *r.Identity
}

// A model resolves by its entry name or by its source repository; anything else is null with
// a reason. A chip resolves by its hardware file. A query is never read as a path.
func TestCatalogResolve(t *testing.T) {
	root := substratedata.MustCatalog()
	for _, c := range []struct{ query, want string }{
		{"llama-3.1-70b-instruct", "llama-3.1-70b-instruct"},
		{"meta-llama/Llama-3.1-70B-Instruct", "llama-3.1-70b-instruct"},
		{"META-LLAMA/llama-3.1-70b-instruct", "llama-3.1-70b-instruct"},
		{"llama70b", ""},
		{"../hardware", ""},
		{"llama-3.1-70b-instruct/../qwen3-14b", ""},
	} {
		r, err := resolveCatalogModel(root, c.query)
		if got := resolved(t, r, err); got != c.want {
			t.Errorf("model %q resolves to %q, want %q", c.query, got, c.want)
		}
		if r.Reason == "" || r.Query != c.query {
			t.Errorf("model %q: %+v lacks the query or a reason", c.query, r)
		}
	}
	for _, c := range []struct{ query, want string }{
		{"h200", "h200"},
		{"nvidia-h200", ""},
		{"../models", ""},
	} {
		r, err := resolveCatalogChip(root, c.query)
		if got := resolved(t, r, err); got != c.want {
			t.Errorf("chip %q resolves to %q, want %q", c.query, got, c.want)
		}
	}
}

// A source repository two entries claim is refused, naming both, rather than resolved to one.
func TestCatalogResolveRefusesAnAmbiguousRepository(t *testing.T) {
	src := substratedata.MustCatalog()
	root := t.TempDir()
	body, err := os.ReadFile(filepath.Join(src, "models", "qwen3-14b", "model.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"qwen3-14b", "qwen3-14b-copy"} {
		dir := filepath.Join(root, "models", name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		b := strings.Replace(string(body), "name: qwen3-14b", "name: "+name, 1)
		if err := os.WriteFile(filepath.Join(dir, "model.yaml"), []byte(b), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	_, err = resolveCatalogModel(root, "Qwen/Qwen3-14B")
	if err == nil || !strings.Contains(err.Error(), "qwen3-14b, qwen3-14b-copy") {
		t.Errorf("an ambiguous repository must be refused naming both entries, got %v", err)
	}
}

// The command's JSON has exactly the three keys, with null for no identity, and refuses
// neither-or-both of --model and --chip.
func TestCatalogResolveCommand(t *testing.T) {
	run := func(args ...string) (string, error) {
		t.Helper()
		var out bytes.Buffer
		catalogPath, resolveModel, resolveChip, substrateJSON = "", "", "", false
		t.Cleanup(func() { catalogPath, resolveModel, resolveChip, substrateJSON = "", "", "", false })
		rootCmd.SetOut(&out)
		rootCmd.SetErr(&bytes.Buffer{})
		rootCmd.SetArgs(args)
		t.Cleanup(func() { rootCmd.SetOut(nil); rootCmd.SetErr(nil); rootCmd.SetArgs(nil) })
		err := rootCmd.Execute()
		return out.String(), err
	}
	cat := substratedata.MustCatalog()
	out, err := run("catalog", "resolve", "--catalog", cat, "--chip", "nvidia-h200", "--json")
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatal(err)
	}
	if v, ok := got["identity"]; !ok || v != nil || got["query"] != "nvidia-h200" || got["reason"] == "" || len(got) != 3 {
		t.Errorf("unresolved chip JSON = %s", out)
	}
	if _, err := run("catalog", "resolve", "--catalog", cat); err == nil {
		t.Error("neither --model nor --chip must be refused")
	}
	if _, err := run("catalog", "resolve", "--catalog", cat, "--model", "x", "--chip", "y"); err == nil {
		t.Error("both --model and --chip must be refused")
	}
}

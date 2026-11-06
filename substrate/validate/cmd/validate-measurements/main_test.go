package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const runFile = "synthetic-h100-tp4.yaml"

func golden(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "..", "contracts", "spec", "measurement", "testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// tree writes a measurements directory from a path->contents map.
func tree(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for rel, body := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func exercise(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := run(args, &out, &errb)
	return code, out.String(), errb.String()
}

func goodTree(t *testing.T) map[string]string {
	return map[string]string{
		"sources.yaml":                 golden(t, "sources.yaml"),
		"synthetic-fixture/" + runFile: golden(t, "run.yaml"),
	}
}

func TestValidTreePasses(t *testing.T) {
	code, out, errb := exercise(tree(t, goodTree(t)))
	if code != 0 || errb != "" {
		t.Fatalf("exit = %d, want 0; stderr:\n%s", code, errb)
	}
	if !strings.Contains(out, "EXACT/RESOLVED") || !strings.Contains(out, "all 2 file(s) validate") {
		t.Errorf("stdout is missing the verdict or the summary:\n%s", out)
	}
}

// Each case breaks one directory-level rule; the run must fail with a message naming it.
func TestTreeRejects(t *testing.T) {
	cases := []struct {
		name, want string
		mutate     func(map[string]string)
	}{
		{"no registry", "sources.yaml", func(m map[string]string) { delete(m, "sources.yaml") }},
		{"run filed under another licence", "filed under", func(m map[string]string) {
			m["other/"+runFile] = m["synthetic-fixture/"+runFile]
			delete(m, "synthetic-fixture/"+runFile)
		}},
		{"file name differs from run name", "its file is", func(m map[string]string) {
			m["synthetic-fixture/renamed.yaml"] = m["synthetic-fixture/"+runFile]
			delete(m, "synthetic-fixture/"+runFile)
		}},
		{"run at the top level", "only sources.yaml", func(m map[string]string) {
			m[runFile] = m["synthetic-fixture/"+runFile]
		}},
		{"stated verdict contradicts", "computed, not asserted", func(m map[string]string) {
			m["synthetic-fixture/"+runFile] = strings.Replace(m["synthetic-fixture/"+runFile],
				"resolution: RESOLVED", "resolution: STRICT", 1)
		}},
		{"unknown field", "not found in type", func(m map[string]string) {
			m["synthetic-fixture/"+runFile] = strings.Replace(m["synthetic-fixture/"+runFile],
				"schema_version: 1", "schema_version: 1\ntrusted: true", 1)
		}},
		{"dangling duplicate_of", "no run in this directory", func(m map[string]string) {
			m["synthetic-fixture/"+runFile] += "duplicate_of: nowhere\n"
		}},
		{"unverified licence", "does not permit redistribution", func(m map[string]string) {
			m["hud/"+runFile] = strings.Replace(strings.Replace(m["synthetic-fixture/"+runFile],
				"license_ref: synthetic-fixture", "license_ref: hud", 1),
				"verdict: EXACT\nresolution: RESOLVED\n", "verdict: PARTIAL\n", 1)
			delete(m, "synthetic-fixture/"+runFile)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := goodTree(t)
			tc.mutate(m)
			code, _, errb := exercise(tree(t, m))
			if code != 1 || !strings.Contains(errb, tc.want) {
				t.Fatalf("exit = %d, want 1 with %q; stderr:\n%s", code, tc.want, errb)
			}
		})
	}
}

// -local admits a run whose licence permits scoring but not redistribution, and still
// refuses one whose licence permits nothing.
func TestLocalAllowsScoreOnly(t *testing.T) {
	m := goodTree(t)
	m["sources.yaml"] = strings.Replace(m["sources.yaml"],
		"      - derive_coefficients\n      - redistribute\n", "", 1)
	dir := tree(t, m)
	if code, _, errb := exercise(dir); code != 1 || !strings.Contains(errb, "redistribution") {
		t.Fatalf("without -local: exit = %d, want 1; stderr:\n%s", code, errb)
	}
	if code, _, errb := exercise("-local", dir); code != 0 {
		t.Fatalf("with -local: exit = %d, want 0; stderr:\n%s", code, errb)
	}
}

func TestMisuse(t *testing.T) {
	for _, args := range [][]string{nil, {"a", "b"}, {filepath.Join(t.TempDir(), "absent")}} {
		if code, _, _ := exercise(args...); code != 2 {
			t.Errorf("%v: exit = %d, want 2", args, code)
		}
	}
}

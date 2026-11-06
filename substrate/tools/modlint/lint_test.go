package main

import (
	"strings"
	"testing"
	"time"
)

const simGoMod = `module github.com/inference-sim/inference-sim

go 1.27.2

require (
	github.com/inference-sim/blis-vllm v0.32.0-blis.2
	github.com/llm-d/llm-d-router v0.0.0-00010101000000-000000000000
)

exclude github.com/inference-sim/blis-vllm v0.32.0

replace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router v1.0.0-blis.2
`

func simModules() []Module {
	return []Module{
		{Path: "github.com/inference-sim/inference-sim", Main: true, GoVersion: "1.27.2"},
		{Path: "github.com/inference-sim/blis-vllm", Version: "v0.32.0-blis.2", GoVersion: "1.27.2"},
		{Path: "github.com/inference-sim/blis-substrate", Version: "v0.11.0", GoVersion: "1.24"},
		{Path: "github.com/llm-d/llm-d-router", Version: "v0.0.0-00010101000000-000000000000",
			Replace: &Module{Path: "github.com/inference-sim/llm-d-router", Version: "v1.0.0-blis.2", GoVersion: "1.27.2"}},
		{Path: "gopkg.in/yaml.v3", Version: "v3.0.1"},
	}
}

func simInput() Input {
	return Input{
		GoMod:   []byte(simGoMod),
		Modules: simModules(),
		Graph: [][2]string{
			{"github.com/inference-sim/inference-sim", "github.com/llm-d/llm-d-router@v0.0.0-00010101000000-000000000000"},
			{"github.com/inference-sim/blis-vllm@v0.32.0-blis.2", "github.com/inference-sim/blis-substrate@v0.11.0"},
		},
		Toolchain: "go1.27.2",
		GoVersion: "go1.27.2",
	}
}

func TestInferenceSimAsDesignedIsClean(t *testing.T) {
	bad, err := Lint(simInput())
	if err != nil {
		t.Fatal(err)
	}
	if len(bad) > 0 {
		t.Fatalf("a go.mod that follows X11 was refused:\n%s", strings.Join(bad, "\n"))
	}
}

// Each case is a known-bad input (X18): one change to a clean inference-sim input that
// breaks one rule of X11 or LIF-2. The lint must report it.
func TestEachRuleHasAKnownBadInput(t *testing.T) {
	cases := []struct {
		name string
		edit func(*Input)
		want string
	}{
		{"a branch replace target", func(in *Input) {
			in.GoMod = []byte(strings.Replace(simGoMod, "v1.0.0-blis.2", "v1.0.1-0.20261101000000-abcdefabcdef", 1))
		}, "is not a -blis.N tag"},
		{"a local replace", func(in *Input) {
			in.GoMod = []byte(strings.Replace(simGoMod, "github.com/inference-sim/llm-d-router v1.0.0-blis.2", "../llm-d-router", 1))
		}, "replace target ../llm-d-router"},
		{"a second replace", func(in *Input) {
			in.GoMod = append([]byte(simGoMod), "replace github.com/inference-sim/blis-vllm => ../blis-vllm\n"...)
		}, "has 2 replace directives"},
		{"no replace", func(in *Input) {
			in.GoMod = []byte(strings.Split(simGoMod, "replace ")[0])
		}, "has 0 replace directives"},
		{"an exclude of a -blis.N tag", func(in *Input) {
			in.GoMod = []byte(strings.Replace(simGoMod, "blis-vllm v0.32.0\n", "blis-vllm v0.32.0-blis.1\n", 1))
		}, "an exclude may name only a plain tag"},
		{"an exclude of a non-engine module", func(in *Input) {
			in.GoMod = []byte(strings.Replace(simGoMod, "exclude github.com/inference-sim/blis-vllm v0.32.0", "exclude github.com/inference-sim/blis-substrate v0.10.0", 1))
		}, "an exclude may name only an engine module"},
		{"a committed go.work", func(in *Input) { in.GoWork = []string{"go.work"} }, "go.work is committed"},
		{"another module requires the router", func(in *Input) {
			in.Graph = append(in.Graph, [2]string{"github.com/inference-sim/blis-vllm@v0.32.0-blis.2", "github.com/llm-d/llm-d-router@v1.0.0"})
		}, "only inference-sim may require the router"},
		{"a BLIS module at a pseudo-version", func(in *Input) {
			in.Modules[1].Version = "v0.32.1-0.20261101000000-abcdefabcdef"
		}, "is not a release tag"},
		{"an engine at a plain upstream tag", func(in *Input) { in.Modules[1].Version = "v0.32.0" }, "must be a -blis.N tag"},
		{"an engine at a release candidate", func(in *Input) { in.Modules[1].Version = "v0.32.0-rc.1" }, "must be a -blis.N tag"},
		{"the substrate at a release candidate", func(in *Input) { in.Modules[2].Version = "v0.12.0-rc.1" }, "must be a plain vX.Y.Z tag"},
		{"the substrate at a -blis.N tag", func(in *Input) { in.Modules[2].Version = "v0.12.0-blis.1" }, "must be a plain vX.Y.Z tag"},
		{"the substrate with build metadata", func(in *Input) { in.Modules[2].Version = "v0.12.0+meta" }, "must be a plain vX.Y.Z tag"},
		{"the router fork at a plain tag", func(in *Input) {
			in.Modules = append(in.Modules, Module{Path: "github.com/inference-sim/llm-d-router", Version: "v1.0.0"})
		}, "must be a -blis.N tag"},
		{"the router selected without its replace", func(in *Input) { in.Modules[3].Replace = nil }, "without the router fork's replace"},
		{"a module above the toolchain", func(in *Input) { in.Modules[2].GoVersion = "1.28" }, "above contracts/toolchain"},
		{"a go command other than the toolchain", func(in *Input) { in.GoVersion = "go1.27.3" }, "go version reports go1.27.3"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			in := simInput()
			in.Modules = simModules()
			c.edit(&in)
			bad, err := Lint(in)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(strings.Join(bad, "\n"), c.want) {
				t.Fatalf("lint did not report %q; it reported:\n%s", c.want, strings.Join(bad, "\n"))
			}
		})
	}
}

func TestOutsideInferenceSimNoReplaceOrExclude(t *testing.T) {
	gomod := "module github.com/inference-sim/blis-vllm\n\ngo 1.27.2\n\nreplace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router v1.0.0-blis.2\n\nexclude github.com/inference-sim/blis-sglang v0.5.23\n"
	bad, err := Lint(Input{
		GoMod:     []byte(gomod),
		Modules:   []Module{{Path: "github.com/inference-sim/blis-vllm", Main: true, GoVersion: "1.27.2"}},
		Toolchain: "go1.27.2",
	})
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(bad, "\n")
	for _, want := range []string{"only inference-sim may carry a replace directive", "only inference-sim may carry exclude directives"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
}

func TestTagForms(t *testing.T) {
	for _, v := range []string{"v0.32.0-blis.1", "v0.32.0-blis.10", "v0.5.6-post.1.blis.2", "v0.10.1-rev.1.blis.1"} {
		if !blisTag.MatchString(v) {
			t.Errorf("%s is a LIF-5 tag", v)
		}
	}
	for _, v := range []string{"v0.32.0", "v0.32.0-blis.0", "v0.32.0-blis", "v0.32.0-rc.1", "v0.5.6-post.0.blis.1", "v0.32.0-blis.01"} {
		if blisTag.MatchString(v) {
			t.Errorf("%s is not a LIF-5 tag", v)
		}
	}
	for v, want := range map[string]bool{
		"v0.0.0-20261101000000-abcdefabcdef":       true,
		"v0.32.1-0.20261101000000-abcdefabcdef":    true,
		"v1.2.4-pre.0.20261101000000-abcdefabcdef": true,
		"v0.32.0-blis.2":                           false,
		"v0.11.0":                                  false,
	} {
		if isPseudo(v) != want {
			t.Errorf("isPseudo(%s) = %v", v, !want)
		}
	}
}

// This repository, as committed, passes its own lint, and contracts/toolchain is a valid
// version no lower than any go directive here.
func TestThisRepositoryIsClean(t *testing.T) {
	bad, err := lintDir("../..", "")
	if err != nil {
		t.Fatal(err)
	}
	// The go command running the tests may differ from contracts/toolchain on a developer
	// machine; CI installs exactly that version and runs modlint itself.
	var real []string
	for _, b := range bad {
		if !strings.HasPrefix(b, "go version reports") {
			real = append(real, b)
		}
	}
	if len(real) > 0 {
		t.Fatalf("blis-substrate fails its own selected-version lint:\n%s", strings.Join(real, "\n"))
	}
}

// The pin-by-commit deferral: a pseudo-version pin is deferred while the file holds; every
// other finding fails; an expired, unowned or malformed file defers nothing.
func TestApplyDeferral(t *testing.T) {
	day := func(s string) time.Time { d, _ := time.Parse("2006-01-02", s); return d }
	const file = "issue: inference-sim/blis-substrate#14\nadded: 2026-11-05\nexpires: 2026-12-05\nmodules: [blis-substrate, blis-vllm, llm-d-router]\nreason: tags follow review\n"
	pin := "github.com/inference-sim/blis-substrate resolves to v0.0.0-20261105193227-4a3becf2d09a, which is not a release tag (X11, LIF-2)"
	other := "go version reports go1.24.0; contracts/toolchain names go1.27.2 (LIF-2)"
	fail, deferred := ApplyDeferral([]byte(file), day("2026-11-05"), []string{pin, other})
	if len(deferred) != 1 || len(fail) != 1 || fail[0] != other {
		t.Fatalf("held: fail=%v deferred=%v", fail, deferred)
	}
	if fail, deferred = ApplyDeferral(nil, day("2026-11-05"), []string{pin}); len(fail) != 1 || len(deferred) != 0 {
		t.Fatalf("no file: fail=%v deferred=%v", fail, deferred)
	}
	// The last day before the expiry date still defers, at any time of day; the date itself
	// does not, from its first instant.
	for _, at := range []string{"2026-12-04T00:00:00Z", "2026-12-04T23:59:59Z"} {
		now, err := time.Parse(time.RFC3339, at)
		if err != nil {
			t.Fatal(err)
		}
		if fail, deferred = ApplyDeferral([]byte(file), now, []string{pin}); len(deferred) != 1 || len(fail) != 0 {
			t.Fatalf("%s: fail=%v deferred=%v, want 1 deferred", at, fail, deferred)
		}
	}
	if fail, deferred = ApplyDeferral([]byte(file), day("2026-12-04"), []string{pin}); len(deferred) != 1 || len(fail) != 0 {
		t.Fatalf("2026-12-04: fail=%v deferred=%v, want 1 deferred", fail, deferred)
	}
	if now, _ := time.Parse(time.RFC3339, "2026-12-05T00:00:01Z"); !now.IsZero() {
		if fail, deferred = ApplyDeferral([]byte(file), now, []string{pin}); len(deferred) != 0 || len(fail) != 2 {
			t.Fatalf("2026-12-05T00:00:01Z: fail=%v deferred=%v, want expired", fail, deferred)
		}
	}
	// A deferral file that is missing is no file: nothing is deferred and every finding fails.
	if fail, deferred = ApplyDeferral([]byte{}, day("2026-11-05"), []string{pin}); len(fail) != 1 || len(deferred) != 0 {
		t.Fatalf("empty file: fail=%v deferred=%v", fail, deferred)
	}
	if fail, deferred = ApplyDeferral([]byte(file), day("2026-12-05"), []string{pin}); len(deferred) != 0 || len(fail) != 2 || !strings.Contains(fail[1], "expired") {
		t.Fatalf("expired: fail=%v deferred=%v", fail, deferred)
	}
	if fail, _ = ApplyDeferral([]byte("issue: x\nadded: 2026-11-05\nexpires: 2026-12-05\nmodules: [blis-substrate]\nreason: y\n"), day("2026-11-05"), []string{pin}); len(fail) != 2 || !strings.Contains(fail[1], "owning issue") {
		t.Fatalf("bad issue: %v", fail)
	}
}

// The deferral lasts at most 30 days from the day the file was added (plan §3.3), covers
// only the BLIS modules the file lists, and is not dated in the future.
func TestDeferralBounds(t *testing.T) {
	day := func(s string) time.Time { d, _ := time.Parse("2006-01-02", s); return d }
	file := func(added, expires, modules string) []byte {
		return []byte("issue: inference-sim/blis-substrate#14\nadded: " + added + "\nexpires: " + expires + "\nmodules: " + modules + "\nreason: r\n")
	}
	pin := func(m string) string {
		return "github.com/inference-sim/" + m + " resolves to v0.0.0-20261105193227-4a3becf2d09a, which is not a release tag (X11, LIF-2)"
	}
	today := day("2026-11-06")
	if fail, def := ApplyDeferral(file("2026-11-05", "2099-01-01", "[blis-substrate]"), today, []string{pin("blis-substrate")}); len(def) != 0 || len(fail) != 2 || !strings.Contains(fail[1], "30 days") {
		t.Errorf("expires 2099: fail=%v deferred=%v", fail, def)
	}
	if fail, def := ApplyDeferral(file("2026-11-05", "2026-12-05", "[blis-substrate]"), today, []string{pin("blis-substrate")}); len(def) != 1 || len(fail) != 0 {
		t.Errorf("exactly 30 days: fail=%v deferred=%v", fail, def)
	}
	if fail, def := ApplyDeferral(file("2026-11-05", "2026-12-06", "[blis-substrate]"), today, []string{pin("blis-substrate")}); len(def) != 0 || len(fail) != 2 {
		t.Errorf("31 days: fail=%v deferred=%v", fail, def)
	}
	if fail, def := ApplyDeferral(file("2026-11-07", "2026-11-20", "[blis-substrate]"), today, []string{pin("blis-substrate")}); len(def) != 0 || len(fail) != 2 || !strings.Contains(fail[1], "future") {
		t.Errorf("added in the future: fail=%v deferred=%v", fail, def)
	}
	// A module the file does not list, or one outside BLIS, is not deferred.
	f := file("2026-11-05", "2026-12-05", "[blis-substrate]")
	other := "github.com/inference-sim/blis-vllm resolves to v0.0.0-20261105193227-4a3becf2d09a, which is not a release tag (X11, LIF-2)"
	foreign := "github.com/example/blis-substrate resolves to v0.0.0-20261105193227-4a3becf2d09a, which is not a release tag (X11, LIF-2)"
	if fail, def := ApplyDeferral(f, today, []string{other, foreign}); len(def) != 0 || len(fail) != 2 {
		t.Errorf("unlisted module: fail=%v deferred=%v", fail, def)
	}
	if fail, _ := ApplyDeferral(file("2026-11-05", "2026-12-05", "[]"), today, []string{pin("blis-substrate")}); len(fail) != 2 || !strings.Contains(fail[1], "modules") {
		t.Errorf("empty modules: %v", fail)
	}
	if fail, _ := ApplyDeferral(file("2026-11-05", "2026-12-05", "[inference-sim]"), today, []string{pin("blis-substrate")}); len(fail) != 2 || !strings.Contains(fail[1], "modules") {
		t.Errorf("unknown module name: %v", fail)
	}
}

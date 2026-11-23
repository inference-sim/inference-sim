package main

import (
	"strings"
	"testing"
	"time"
)

const simGoMod = `module github.com/inference-sim/inference-sim

go 1.27.2

require github.com/llm-d/llm-d-router v0.0.0-00010101000000-000000000000

replace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router v1.0.0-blis.2
`

func simModules() []Module {
	return []Module{
		{Path: "github.com/inference-sim/inference-sim", Main: true, GoVersion: "1.27.2"},
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
		{"an exclude of the retired engine module", func(in *Input) {
			in.GoMod = append([]byte(simGoMod), "exclude github.com/inference-sim/blis-vllm v0.32.0\n"...)
		}, "has no module version to exclude"},
		{"a committed go.work", func(in *Input) { in.GoWork = []string{"go.work"} }, "go.work is committed"},
		{"another module requires the router", func(in *Input) {
			in.Graph = append(in.Graph, [2]string{"github.com/example/x@v1.0.0", "github.com/llm-d/llm-d-router@v1.0.0"})
		}, "only inference-sim may require the router"},
		{"a retired substrate module in the build", func(in *Input) {
			in.Modules = append(in.Modules, Module{Path: "github.com/inference-sim/blis-substrate", Version: "v0.11.0"})
		}, "blis-substrate is selected"},
		{"a retired engine module in the build", func(in *Input) {
			in.Modules = append(in.Modules, Module{Path: "github.com/inference-sim/blis-vllm", Version: "v0.33.0-blis.1"})
		}, "blis-vllm is selected"},
		{"the router fork at a plain tag", func(in *Input) {
			in.Modules = append(in.Modules, Module{Path: "github.com/inference-sim/llm-d-router", Version: "v1.0.0"})
		}, "must be a -blis.N tag"},
		{"the router selected without its replace", func(in *Input) { in.Modules[1].Replace = nil }, "without the router fork's replace"},
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

// This repository, as committed, passes its own lint, and substrate/contracts/toolchain is a
// valid version no lower than any go directive here.
func TestThisRepositoryIsClean(t *testing.T) {
	bad, err := lintDir("../../..", "")
	if err != nil {
		t.Fatal(err)
	}
	// The go command running the tests may differ from contracts/toolchain on a developer
	// machine; CI installs exactly that version and runs modlint itself. The router fork may be
	// pinned by commit while a draft is open, which the deferral file covers in CI.
	var real []string
	for _, b := range bad {
		if !strings.HasPrefix(b, "go version reports") && !deferrable(b, []string{"llm-d-router"}) {
			real = append(real, b)
		}
	}
	if len(real) > 0 {
		t.Fatalf("this module fails its own selected-version lint:\n%s", strings.Join(real, "\n"))
	}
}

func day(s string) time.Time { d, _ := time.Parse("2006-01-02", s); return d }

// The pin-by-commit deferral covers the router fork at a commit and nothing else; an expired,
// unowned, over-long, future-dated or malformed file defers nothing and adds a finding.
func TestApplyDeferral(t *testing.T) {
	file := func(added, expires, modules string) []byte {
		return []byte("issue: inference-sim/inference-sim#2137\nadded: " + added + "\nexpires: " + expires + "\nmodules: " + modules + "\nreason: tags follow review\n")
	}
	finding := func(path, version string) string {
		return "github.com/llm-d/llm-d-router resolves to " + path + " " + version + "; it must resolve to github.com/inference-sim/llm-d-router at a -blis.N tag (LIF-2)"
	}
	rep := func(v string) string {
		return `go.mod:132: replace target version "` + v + `" is not a -blis.N tag (X11, LIF-5)`
	}
	const pv = "v1.0.1-0.20261105040605-fd1830412b30"
	pin := finding("github.com/inference-sim/llm-d-router", pv)
	other := "go version reports go1.24.0; contracts/toolchain names go1.27.2 (LIF-2)"
	today := day("2026-11-06")
	held := file("2026-11-05", "2026-12-05", "[llm-d-router]")

	if fail, def := ApplyDeferral(held, today, []string{pin, rep(pv), other}); len(def) != 2 || len(fail) != 1 || fail[0] != other {
		t.Errorf("held: fail=%v deferred=%v", fail, def)
	}
	for name, f := range map[string]string{
		"a replace to another module":      finding("github.com/someone/llm-d-router", pv),
		"a branch name, not a commit":      finding("github.com/inference-sim/llm-d-router", "llm-d-r"),
		"a tag-shaped version":             finding("github.com/inference-sim/llm-d-router", "v1.0.0-blis.4"),
		"a replace target at a plain tag":  rep("v1.0.1"),
		"a retired module at a pseudo-tag": "github.com/inference-sim/blis-vllm resolves to " + pv + ", which is not a release tag (X11, LIF-2)",
	} {
		if fail, def := ApplyDeferral(held, today, []string{f}); len(def) != 0 || len(fail) != 1 {
			t.Errorf("%s: fail=%v deferred=%v, want it to fail", name, fail, def)
		}
	}
	for name, c := range map[string]struct {
		src   []byte
		on    time.Time
		extra string
	}{
		"no file":                {nil, today, ""},
		"an empty file":          {[]byte{}, today, ""},
		"expired":                {held, day("2026-12-05"), "expired"},
		"more than 30 days":      {file("2026-11-05", "2026-12-06", "[llm-d-router]"), today, "30 days"},
		"added in the future":    {file("2026-11-07", "2026-11-20", "[llm-d-router]"), today, "future"},
		"no modules":             {file("2026-11-05", "2026-12-05", "[]"), today, "modules"},
		"a retired module named": {file("2026-11-05", "2026-12-05", "[blis-substrate]"), today, "modules"},
		"no owning issue":        {[]byte("issue: x\nadded: 2026-11-05\nexpires: 2026-12-05\nmodules: [llm-d-router]\nreason: y\n"), today, "owning issue"},
	} {
		fail, def := ApplyDeferral(c.src, c.on, []string{pin})
		if len(def) != 0 || len(fail) == 0 || (c.extra != "" && !strings.Contains(strings.Join(fail, "\n"), c.extra)) {
			t.Errorf("%s: fail=%v deferred=%v", name, fail, def)
		}
	}
	// The last day before the expiry date still defers, at any time of day; the date itself does not.
	for _, at := range []string{"2026-12-04T00:00:00Z", "2026-12-04T23:59:59Z"} {
		now, err := time.Parse(time.RFC3339, at)
		if err != nil {
			t.Fatal(err)
		}
		if fail, def := ApplyDeferral(held, now, []string{pin}); len(def) != 1 || len(fail) != 0 {
			t.Errorf("%s: fail=%v deferred=%v, want 1 deferred", at, fail, def)
		}
	}
	if fail, def := ApplyDeferral(held, day("2026-12-05"), []string{pin}); len(def) != 0 || len(fail) != 2 {
		t.Errorf("2026-12-05: fail=%v deferred=%v", fail, def)
	}
}

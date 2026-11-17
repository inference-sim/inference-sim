// Original to blis-sglang.

// Command metricsmanifest lists every sglang:* metric family SGLang's serving code defines, from
// the release's own source, and writes it as docs/metrics-manifest.json with each family's class:
//
//   - exact: the scheduler alone decides the value and core/metrics reproduces it;
//   - derived: it needs an input the simulator supplies, such as time or a transfer outcome;
//   - unavailable: it needs a device, a process or a feature outside the simulation;
//   - later: a feature a later stage models (the stage is in "why").
//
// A family is defined where a Gauge, Counter or Histogram constructor takes a "sglang:" name as its
// name= argument or first positional argument, under python/sglang/srt. Every other "sglang:"
// string literal there must be listed in the classes file's not_families with why, and every
// family must match a class rule (or be listed in exact_series). A family a release adds or drops
// therefore fails the run until docs/metrics-classes.json says what it is, which is the point: a
// family nobody ported is a listed gap, never a silent one.
//
// Usage: go run ./cmd/metricsmanifest -sglang <checkout> [-classes docs/metrics-classes.json]
// [-out docs/metrics-manifest.json] [-check]
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Family is one metric family as the release defines it.
type Family struct {
	Name   string   `json:"name"`
	Type   string   `json:"type"`
	Labels []string `json:"labels,omitempty"`
	Site   string   `json:"site"`
	Class  string   `json:"class"`
	Why    string   `json:"why,omitempty"`
}

// Rule classifies the families whose name matches.
type Rule struct {
	Match string `json:"match"`
	Class string `json:"class"`
	Why   string `json:"why"`
}

// Classes is docs/metrics-classes.json.
type Classes struct {
	ExactSeries []string          `json:"exact_series"`
	Rules       []Rule            `json:"rules"`
	NotFamilies map[string]string `json:"not_families"`
}

// Manifest is docs/metrics-manifest.json.
type Manifest struct {
	Upstream string   `json:"upstream"`
	Families []Family `json:"families"`
}

var (
	ctor    = regexp.MustCompile(`(Gauge|Counter|Histogram|Summary)\(\s*(?:name\s*=\s*)?"(sglang:[A-Za-z0-9_:]+)"`)
	literal = regexp.MustCompile(`"(sglang:[A-Za-z0-9_:]+)`)
	labelsR = regexp.MustCompile(`labelnames\s*=\s*([^\n]*)`)
)

func main() {
	root := flag.String("sglang", "", "SGLang checkout")
	classesPath := flag.String("classes", "docs/metrics-classes.json", "classes file")
	out := flag.String("out", "docs/metrics-manifest.json", "manifest")
	check := flag.Bool("check", false, "fail if the manifest on disk differs")
	flag.Parse()
	if err := run(*root, *classesPath, *out, *check); err != nil {
		fmt.Fprintln(os.Stderr, "metricsmanifest:", err)
		os.Exit(1)
	}
}

func run(root, classesPath, out string, check bool) error {
	if root == "" {
		return fmt.Errorf("-sglang is required")
	}
	var cl Classes
	b, err := os.ReadFile(classesPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &cl); err != nil {
		return err
	}
	rules := make([]*regexp.Regexp, len(cl.Rules))
	for i, r := range cl.Rules {
		if rules[i], err = regexp.Compile(r.Match); err != nil {
			return fmt.Errorf("rule %q: %w", r.Match, err)
		}
	}
	exact := map[string]bool{}
	for _, n := range cl.ExactSeries {
		exact[n] = true
	}
	srt := filepath.Join(root, "python", "sglang", "srt")
	fams := map[string]Family{}
	seenLiteral := map[string]string{}
	err = filepath.WalkDir(srt, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".py") {
			return err
		}
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(filepath.Join(root, "python", "sglang"), p)
		text := string(src)
		for _, m := range ctor.FindAllStringSubmatchIndex(text, -1) {
			name := text[m[4]:m[5]]
			line := 1 + strings.Count(text[:m[0]], "\n")
			f := Family{Name: name, Type: strings.ToLower(text[m[2]:m[3]]), Site: fmt.Sprintf("%s:%d", rel, line)}
			end := min(len(text), m[1]+1200)
			if close := strings.Index(text[m[1]:end], "\n        )"); close >= 0 {
				end = m[1] + close
			}
			if lm := labelsR.FindStringSubmatch(text[m[1]:end]); lm != nil {
				f.Labels = []string{strings.TrimRight(strings.TrimSpace(lm[1]), ",")}
			}
			if old, dup := fams[name]; dup && old.Site < f.Site {
				continue
			}
			fams[name] = f
		}
		for _, m := range literal.FindAllStringSubmatchIndex(text, -1) {
			n := text[m[2]:m[3]]
			if !strings.HasPrefix(n, "sglang::") {
				seenLiteral[n] = rel
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	var names []string
	for n := range fams {
		names = append(names, n)
	}
	sort.Strings(names)
	var problems []string
	man := Manifest{Upstream: "SGLang v0.5.23 297e54fd105e974cdf401e5b4d20253a22c16f90"}
	for _, n := range names {
		f := fams[n]
		switch {
		case exact[n]:
			f.Class, f.Why = "exact", "the scheduler decides it; core/metrics reproduces it"
		default:
			for i, r := range rules {
				if r.MatchString(n) {
					f.Class, f.Why = cl.Rules[i].Class, cl.Rules[i].Why
					break
				}
			}
		}
		if f.Class == "" {
			problems = append(problems, "no class for "+n+" ("+f.Site+")")
		}
		man.Families = append(man.Families, f)
	}
	for n := range exact {
		if _, ok := fams[n]; !ok {
			if _, lit := seenLiteral[n]; !lit {
				problems = append(problems, "exact series "+n+" is not defined by the release")
			}
		}
	}
	for n, where := range seenLiteral {
		if _, ok := fams[n]; ok {
			continue
		}
		if _, ok := cl.NotFamilies[n]; !ok {
			problems = append(problems, fmt.Sprintf("literal %q (%s) is neither a family nor listed in not_families", n, where))
		}
	}
	for n := range cl.NotFamilies {
		if _, ok := seenLiteral[n]; !ok {
			problems = append(problems, "not_families lists "+n+", which the release no longer contains")
		}
	}
	sort.Strings(problems)
	if len(problems) > 0 {
		return fmt.Errorf("%d problem(s):\n  %s", len(problems), strings.Join(problems, "\n  "))
	}
	enc, err := json.MarshalIndent(man, "", " ")
	if err != nil {
		return err
	}
	enc = append(enc, '\n')
	if check {
		cur, err := os.ReadFile(out)
		if err != nil || !bytes.Equal(cur, enc) {
			return fmt.Errorf("%s differs from a fresh manifest; run without -check", out)
		}
		return nil
	}
	return os.WriteFile(out, enc, 0o644)
}

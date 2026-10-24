// Original to blis-vllm.

// Command metricsmanifest lists every vllm:* metric family a vLLM release defines, from the
// release's own source, and writes it as docs/metrics-manifest.json with each family's class:
// exact (blis-vllm computes it from the decisions it makes), derived (it needs inputs the
// simulator supplies, such as time or transfer outcomes, and is reported given them) or
// unavailable (it needs real hardware, a process or a feature outside the simulation).
//
// A family is defined where a "vllm:" name meets a metric type:
//
//   - a Gauge, Counter or Histogram constructor (prometheus_client's, or a logger's
//     _gauge_cls, _counter_cls, _histogram_cls) whose name= argument, or first positional
//     argument, is the literal or a name constant;
//   - an Offloading{Counter,Gauge,Histogram}Metadata entry keyed by a name constant (the KV
//     offload modules' metric tables);
//   - spec_decode's counter_specs table, and the info gauge log_metrics_info builds.
//
// The scanner reads every "vllm:" string literal under vllm/ and accounts for each: it is a
// family definition or a name constant one refers to, or it is listed in classes.json's
// not_families with why (a reader's lookup, a benchmark's prefix filter, an ORCA mapping).
// Any other literal fails the run, as does a family with no classification or a
// classification for a family the release does not define. So a family a release adds or
// drops fails the CI job until classes.json says what it is.
//
// Usage: go run ./cmd/metricsmanifest -vllm <checkout> [-classes path] [-out path] [-check]
//
// With -check it writes nothing and fails if the manifest on disk differs from a fresh one.
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
	Name string `json:"name"`
	// Type is gauge, counter or histogram, as the definition constructs it.
	Type string `json:"type"`
	// Labels are the label names the definition states. A logger family's per-engine
	// labels (model_name, engine) are written as "labelnames" when the definition passes
	// the logger's list through.
	Labels []string `json:"labels"`
	// Defined is where the name meets its type: path:line in the release.
	Defined string `json:"defined"`
	// Module is the subsystem of the file that defines it.
	Module string `json:"module"`
	// Class is exact, derived or unavailable; Reason says why.
	Class  string `json:"class"`
	Reason string `json:"reason"`
	// EnabledBy names the configuration that registers the family, when it is not always.
	EnabledBy string `json:"enabled_by,omitempty"`
}

// Manifest is the generated document.
type Manifest struct {
	Generator string   `json:"generator"`
	VLLM      string   `json:"vllm"`
	Families  []Family `json:"families"`
	Counts    Counts   `json:"counts"`
}

// Counts summarizes the classification.
type Counts struct {
	Total       int `json:"total"`
	Exact       int `json:"exact"`
	Derived     int `json:"derived"`
	Unavailable int `json:"unavailable"`
}

// classification is one family's entry in classes.json: the judgement the source cannot
// make (what the simulation can compute) and the configuration that enables it.
type classification struct {
	Class     string `json:"class"`
	Reason    string `json:"reason"`
	EnabledBy string `json:"enabled_by,omitempty"`
}

type classesFile struct {
	VLLM string `json:"vllm"`
	// NotFamilies are "vllm:" literals that define no family, keyed "<path>:<literal>",
	// each with why.
	NotFamilies map[string]string         `json:"not_families"`
	Families    map[string]classification `json:"families"`
}

var (
	literalRe = regexp.MustCompile(`"(vllm:[a-z][a-z0-9_]*)"`)
	classRe   = regexp.MustCompile(`^class ([A-Za-z_][A-Za-z0-9_]*)\b`)
	// NAME = "vllm:..." at class or module level, optionally typed, optionally parenthesized
	// with the value on the next line.
	constRe     = regexp.MustCompile(`^(\s*)([A-Z_][A-Z0-9_]*)\s*(?::\s*[A-Za-z_\[\]]+\s*)?=\s*"(vllm:[a-z][a-z0-9_]*)"`)
	constOpenRe = regexp.MustCompile(`^(\s*)([A-Z_][A-Z0-9_]*)\s*(?::\s*[A-Za-z_\[\]]+\s*)?=\s*\(\s*$`)
	ctorRe      = regexp.MustCompile(`(_gauge_cls|_counter_cls|_histogram_cls|\bGauge|\bCounter|\bHistogram)\(`)
	metaRe      = regexp.MustCompile(`Offloading(Counter|Gauge|Histogram)Metadata\(`)
	// A constant reference: Class.NAME or NAME.
	refRe       = regexp.MustCompile(`(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Z_][A-Z0-9_]{2,})\b`)
	labelsArgRe = regexp.MustCompile(`labelnames\s*=\s*(.*)`)
	quotedRe    = regexp.MustCompile(`"([a-z_][a-z0-9_]*)"`)
)

type nameConst struct {
	class, name, value, path string
	line                     int
}

func main() {
	src := flag.String("vllm", "", "a vLLM checkout at the release to describe")
	classesPath := flag.String("classes", "cmd/metricsmanifest/classes.json", "the classification")
	out := flag.String("out", "docs/metrics-manifest.json", "where to write the manifest")
	check := flag.Bool("check", false, "fail if the manifest on disk is not current")
	flag.Parse()
	if *src == "" {
		fail("-vllm is required")
	}
	raw, err := os.ReadFile(*classesPath)
	if err != nil {
		fail("%v", err)
	}
	var classes classesFile
	if err := json.Unmarshal(raw, &classes); err != nil {
		fail("%s: %v", *classesPath, err)
	}
	files, err := readTree(*src)
	if err != nil {
		fail("%v", err)
	}
	m, problems := Build(files, classes)
	if len(problems) > 0 {
		for _, p := range problems {
			fmt.Fprintln(os.Stderr, "metricsmanifest:", p)
		}
		os.Exit(1)
	}
	body, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		fail("%v", err)
	}
	body = append(body, '\n')
	summary := fmt.Sprintf("%d families (%d exact, %d derived, %d unavailable)",
		m.Counts.Total, m.Counts.Exact, m.Counts.Derived, m.Counts.Unavailable)
	if *check {
		have, err := os.ReadFile(*out)
		if err != nil {
			fail("%v", err)
		}
		if !bytes.Equal(have, body) {
			fail("%s is not current: regenerate it with go run ./cmd/metricsmanifest -vllm <v%s checkout> and commit it",
				*out, classes.VLLM)
		}
		fmt.Printf("%s is current: %s\n", *out, summary)
		return
	}
	if err := os.WriteFile(*out, body, 0o644); err != nil {
		fail("%v", err)
	}
	fmt.Printf("wrote %s: %s\n", *out, summary)
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "metricsmanifest: "+format+"\n", a...)
	os.Exit(1)
}

// readTree returns every Python file under root/vllm, by slash path relative to root.
func readTree(root string) (map[string]string, error) {
	files := map[string]string{}
	err := filepath.WalkDir(filepath.Join(root, "vllm"), func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".py") {
			return err
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		files[filepath.ToSlash(rel)] = string(b)
		return nil
	})
	return files, err
}

// Build scans the release's files and assembles the manifest. It returns every problem
// rather than the first, so one run names everything a release changed.
func Build(files map[string]string, classes classesFile) (Manifest, []string) {
	var problems []string
	paths := make([]string, 0, len(files))
	for p := range files {
		paths = append(paths, p)
	}
	sort.Strings(paths)

	// Pass 1: name constants, by Class.NAME and by NAME within their own file.
	byQualified := map[string]nameConst{} // "Class.NAME" -> const
	byLocal := map[string]map[string]nameConst{}
	constLiteral := map[string]bool{} // "<path>:<line>" of a constant's literal
	for _, p := range paths {
		lines := strings.Split(files[p], "\n")
		class := ""
		byLocal[p] = map[string]nameConst{}
		for i, l := range lines {
			if m := classRe.FindStringSubmatch(l); m != nil {
				class = m[1]
			} else if l != "" && !strings.HasPrefix(l, " ") && !strings.HasPrefix(l, "\t") && !strings.HasPrefix(l, "#") && !strings.HasPrefix(l, ")") {
				class = ""
			}
			var c *nameConst
			if m := constRe.FindStringSubmatch(l); m != nil {
				c = &nameConst{name: m[2], value: m[3], line: i + 1}
				if m[1] == "" {
					c.class = ""
				} else {
					c.class = class
				}
				constLiteral[fmt.Sprintf("%s:%d", p, i+1)] = true
			} else if m := constOpenRe.FindStringSubmatch(l); m != nil && i+1 < len(lines) {
				if v := literalRe.FindStringSubmatch(strings.TrimSpace(lines[i+1])); v != nil && strings.TrimSpace(lines[i+1]) == `"`+v[1]+`"` {
					c = &nameConst{name: m[2], value: v[1], line: i + 2}
					if m[1] != "" {
						c.class = class
					}
					constLiteral[fmt.Sprintf("%s:%d", p, i+2)] = true
				}
			}
			if c == nil {
				continue
			}
			c.path = p
			byLocal[p][c.name] = *c
			if c.class != "" {
				byQualified[c.class+"."+c.name] = *c
			}
		}
	}
	resolve := func(path, qualifier, name string) (string, bool) {
		if qualifier != "" {
			c, ok := byQualified[qualifier+"."+name]
			return c.value, ok
		}
		c, ok := byLocal[path][name]
		return c.value, ok
	}

	// Pass 2: definitions.
	families := map[string]*Family{}
	define := func(name, typ, path string, line int, labels []string) {
		if f, ok := families[name]; ok {
			if f.Type != typ {
				problems = append(problems, fmt.Sprintf("%s is defined as a %s at %s and a %s at %s:%d",
					name, f.Type, f.Defined, typ, path, line))
			}
			return
		}
		if labels == nil {
			labels = []string{}
		}
		families[name] = &Family{Name: name, Type: typ, Labels: labels,
			Defined: fmt.Sprintf("%s:%d", path, line), Module: moduleOf(path)}
	}
	accounted := map[string]bool{} // "<path>:<line>" of a literal a definition used
	for _, p := range paths {
		lines := strings.Split(files[p], "\n")
		for i, l := range lines {
			// Constructors: the call opens on this line; its name is on this line or the next.
			if m := ctorRe.FindStringSubmatchIndex(l); m != nil {
				typ := typeOf(l[m[2]:m[3]])
				args := l[m[1]:]
				at := i
				if strings.TrimSpace(args) == "" && i+1 < len(lines) {
					args, at = lines[i+1], i+1
				}
				if name, ok := ctorName(p, args, resolve); ok {
					define(name, typ, p, at+1, labelsOf(lines, i))
					accounted[fmt.Sprintf("%s:%d", p, at+1)] = true
				}
			}
			// Metadata tables: KEY: Offloading*Metadata(  or  metrics[KEY] = (Offloading*Metadata(
			if m := metaRe.FindStringSubmatch(l); m != nil {
				typ := strings.ToLower(m[1])
				key := l[:strings.Index(l, m[0])]
				if strings.TrimSpace(key) == "" || strings.TrimSpace(key) == "(" {
					// The metadata opens a parenthesized value; its key is on the line above.
					if i > 0 {
						key = lines[i-1]
					}
				}
				if name, line, ok := metaKey(p, key, resolve, lines, i); ok {
					define(name, typ, p, line, labelsOf(lines, i))
				}
			}
			// spec_decode's counter_specs: ("vllm:...", "documentation"), passed to _counter_cls.
			if strings.Contains(l, "counter_specs = [") {
				for j := i + 1; j < len(lines) && !strings.HasPrefix(strings.TrimSpace(lines[j]), "]"); j++ {
					if v := literalRe.FindStringSubmatch(lines[j]); v != nil {
						define(v[1], "counter", p, j+1, []string{"labelnames"})
						accounted[fmt.Sprintf("%s:%d", p, j+1)] = true
					}
				}
			}
			// log_metrics_info: an Info emulated as a gauge set to 1, labelled by the config's keys.
			if strings.Contains(l, `name = "vllm:`) {
				if v := literalRe.FindStringSubmatch(l); v != nil {
					define(v[1], "gauge", p, i+1, []string{"(the config's metrics_info() keys)"})
					accounted[fmt.Sprintf("%s:%d", p, i+1)] = true
				}
			}
		}
	}

	// Every literal is accounted for.
	seenNotFamily := map[string]bool{}
	for _, p := range paths {
		for i, l := range strings.Split(files[p], "\n") {
			for _, v := range literalRe.FindAllStringSubmatch(l, -1) {
				loc := fmt.Sprintf("%s:%d", p, i+1)
				key := p + ":" + v[1]
				switch {
				case accounted[loc], constLiteral[loc]:
				case classes.NotFamilies[key] != "":
					seenNotFamily[key] = true
				default:
					problems = append(problems, fmt.Sprintf("%s: %s defines no family the scanner recognizes and is not in not_families", loc, v[1]))
				}
			}
		}
	}
	for key := range classes.NotFamilies {
		if !seenNotFamily[key] {
			problems = append(problems, fmt.Sprintf("not_families lists %s, which the release no longer has", key))
		}
	}
	// Every name constant is defined as a family somewhere.
	for _, p := range paths {
		for _, c := range byLocal[p] {
			if _, ok := families[c.value]; !ok {
				problems = append(problems, fmt.Sprintf("%s:%d: the name constant %s = %s is never bound to a metric type",
					c.path, c.line, c.name, c.value))
			}
		}
	}

	var m Manifest
	m.Generator = "blis-vllm cmd/metricsmanifest"
	m.VLLM = classes.VLLM
	names := make([]string, 0, len(families))
	for n := range families {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		f := families[n]
		c, ok := classes.Families[n]
		switch {
		case !ok:
			problems = append(problems, fmt.Sprintf("%s (%s, %s) has no classification in classes.json", n, f.Type, f.Defined))
			continue
		case c.Class != "exact" && c.Class != "derived" && c.Class != "unavailable":
			problems = append(problems, fmt.Sprintf("%s: class %q is not exact, derived or unavailable", n, c.Class))
		case strings.TrimSpace(c.Reason) == "":
			problems = append(problems, fmt.Sprintf("%s: classified %s with no reason", n, c.Class))
		}
		f.Class, f.Reason, f.EnabledBy = c.Class, c.Reason, c.EnabledBy
		m.Families = append(m.Families, *f)
		switch c.Class {
		case "exact":
			m.Counts.Exact++
		case "derived":
			m.Counts.Derived++
		case "unavailable":
			m.Counts.Unavailable++
		}
	}
	for n := range classes.Families {
		if _, ok := families[n]; !ok {
			problems = append(problems, fmt.Sprintf("classes.json classifies %s, which the release does not define", n))
		}
	}
	m.Counts.Total = len(m.Families)
	sort.Strings(problems)
	return m, problems
}

// ctorName reads the family name from a constructor's arguments: name="vllm:...",
// name=CONST or name=Class.CONST, or a first positional "vllm:..." literal.
func ctorName(path, args string, resolve func(path, qualifier, name string) (string, bool)) (string, bool) {
	args = strings.TrimSpace(args)
	if strings.HasPrefix(args, "name") {
		v := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(strings.TrimPrefix(args, "name")), "="))
		if lit := literalRe.FindStringSubmatch(v); lit != nil && strings.HasPrefix(v, `"vllm:`) {
			return lit[1], true
		}
		if m := refRe.FindStringSubmatch(v); m != nil && strings.HasPrefix(v, m[0]) {
			return resolve(path, m[1], m[2])
		}
		return "", false
	}
	if lit := literalRe.FindStringSubmatch(args); lit != nil && strings.HasPrefix(args, `"vllm:`) {
		return lit[1], true
	}
	return "", false
}

// metaKey reads a metadata table's key, which is a name constant (Class.NAME or NAME), a
// subscript metrics[Class.NAME], or _vllm_metric_name(EXTERNAL), whose value lives outside
// the release (kvcr) and is recorded as such.
func metaKey(path, key string, resolve func(path, qualifier, name string) (string, bool), lines []string, i int) (string, int, bool) {
	key = strings.TrimSpace(key)
	key = strings.TrimSuffix(key, "(")
	key = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(key), "="))
	key = strings.TrimSpace(strings.TrimSuffix(key, ":"))
	if strings.HasPrefix(key, "metrics[") || strings.HasPrefix(key, "definitions[") {
		key = strings.TrimSuffix(key[strings.Index(key, "[")+1:], "]")
	}
	if strings.HasPrefix(key, "_vllm_metric_name(") {
		ext := strings.TrimSuffix(strings.TrimPrefix(key, "_vllm_metric_name("), ")")
		return "vllm:<kvcr " + ext + ">", i + 1, true
	}
	m := refRe.FindStringSubmatch(key)
	if m == nil || m[0] != key {
		return "", 0, false
	}
	v, ok := resolve(path, m[1], m[2])
	return v, i + 1, ok
}

func typeOf(ctor string) string {
	switch {
	case strings.Contains(ctor, "gauge"), ctor == "Gauge":
		return "gauge"
	case strings.Contains(ctor, "counter"), ctor == "Counter":
		return "counter"
	default:
		return "histogram"
	}
}

// labelsOf reads the labelnames argument of the call that opens on line at: the quoted
// names it lists, plus "labelnames" when it extends the logger's per-engine list.
func labelsOf(lines []string, at int) []string {
	depth := 0
	for j := at; j < len(lines) && j < at+16; j++ {
		l := lines[j]
		if m := labelsArgRe.FindStringSubmatch(l); m != nil {
			arg := m[1]
			// A label list bound to a local variable just before the call
			// (pos_labelnames = labelnames + ["position"]) is read from its definition.
			if v := regexp.MustCompile(`^\s*([a-z_]+)\s*,?\s*$`).FindStringSubmatch(arg); v != nil && v[1] != "labelnames" {
				for k := at; k >= 0 && k > at-6; k-- {
					if d := regexp.MustCompile(`^\s*` + v[1] + `\s*=\s*(.*)$`).FindStringSubmatch(lines[k]); d != nil {
						arg = d[1]
						break
					}
				}
			}
			out := []string{}
			if strings.HasPrefix(strings.TrimSpace(arg), "labelnames") {
				out = append(out, "labelnames")
			}
			for _, q := range quotedRe.FindAllStringSubmatch(arg, -1) {
				out = append(out, q[1])
			}
			if len(out) == 0 {
				out = append(out, "("+strings.TrimSuffix(strings.TrimSpace(arg), ",")+")")
			}
			return out
		}
		depth += strings.Count(l, "(") - strings.Count(l, ")")
		if j > at && depth <= 0 {
			break
		}
	}
	return []string{}
}

// moduleOf names the subsystem of a defining file.
func moduleOf(rel string) string {
	switch {
	case strings.HasPrefix(rel, "vllm/v1/metrics/"):
		return "engine"
	case strings.HasPrefix(rel, "vllm/v1/spec_decode/"):
		return "spec_decode"
	case strings.HasPrefix(rel, "vllm/v1/kv_offload/"), strings.HasPrefix(rel, "vllm/v1/simple_kv_offload/"):
		return "kv_offload"
	case strings.HasPrefix(rel, "vllm/distributed/kv_transfer/"):
		return "kv_connector"
	case strings.HasPrefix(rel, "vllm/entrypoints/"), strings.HasPrefix(rel, "vllm/parser/"):
		return "api_server"
	default:
		return "other"
	}
}

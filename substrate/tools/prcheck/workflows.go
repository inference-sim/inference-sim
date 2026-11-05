package main

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// The workflow lint reads each workflow file under .github/workflows and each composite
// action under .github/actions, and reports:
//
//   - X21: a "uses:" of an action that is not pinned by a 40-hex commit (a local action,
//     ./path, needs no pin; a docker:// image must be pinned by sha256 digest);
//   - X21: a pip install that does not pass --require-hashes;
//   - X22: a step or job with continue-on-error, and a run script that ignores an exit
//     code: "|| true", "|| :", "|| exit 0", "set +e", or a run script in a shell other
//     than bash's default -e mode without "set -e";
//   - X16: go test run with -short or a -skip flag in a job, or a skip marker written into
//     a required job's environment (SKIP_*, *_SKIP);
//   - X16: a gate (a workflow whose file starts with "gate", or that carries a
//     "# gate-paths:" comment) whose pull_request.paths do not equal the declared paths.
//
// A line may be exempted only by a "# lint-ok: <invariant> <reason>" comment on the same
// line; the reason is shown, and the exemption itself is a reviewed change (design §13.7).

var (
	pinned      = regexp.MustCompile(`^[^@\s]+@[0-9a-f]{40}$`)
	dockerPin   = regexp.MustCompile(`^docker://[^@\s]+@sha256:[0-9a-f]{64}$`)
	ignoreExit  = regexp.MustCompile(`\|\|\s*(true|:|exit\s+0)\b|^\s*set\s+\+e\b|\|\|\s*echo\b`)
	pipInstall  = regexp.MustCompile(`\bpip3?\s+install\b|\bpython3?\s+-m\s+pip\s+install\b|\buv\s+pip\s+install\b`)
	goTestShort = regexp.MustCompile(`\bgo\s+test\b[^\n]*\s(-short|-test\.short|-skip(=|\s))`)
	skipEnv     = regexp.MustCompile(`^(SKIP_[A-Z0-9_]+|[A-Z0-9_]+_SKIP)$`)
	lintOK      = regexp.MustCompile(`#\s*lint-ok:\s*([A-Z]+-?\d+)\s+\S`)
	gatePaths   = regexp.MustCompile(`(?m)^#\s*gate-paths:\s*(.+)$`)
)

type finding struct {
	file string
	line int
	msg  string
}

func (f finding) String() string { return fmt.Sprintf("%s:%d: %s", f.file, f.line, f.msg) }

// LintWorkflow returns the findings for one workflow or action file.
func LintWorkflow(name string, src []byte) ([]string, error) {
	var root yaml.Node
	if err := yaml.Unmarshal(src, &root); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	lines := strings.Split(string(src), "\n")
	exempt := func(line int) bool {
		return line >= 1 && line <= len(lines) && lintOK.MatchString(lines[line-1])
	}
	var out []finding
	add := func(line int, format string, args ...any) {
		if !exempt(line) {
			out = append(out, finding{name, line, fmt.Sprintf(format, args...)})
		}
	}
	walk(&root, func(key, val *yaml.Node) {
		switch key.Value {
		case "uses":
			u := strings.TrimSpace(val.Value)
			switch {
			case strings.HasPrefix(u, "./"):
			case strings.HasPrefix(u, "docker://"):
				if !dockerPin.MatchString(u) {
					add(val.Line, "%s is not pinned by sha256 digest (X21)", u)
				}
			case !pinned.MatchString(u):
				add(val.Line, "%s is not pinned by a full commit hash (X21)", u)
			}
		case "continue-on-error":
			if val.Value != "false" {
				add(val.Line, "continue-on-error lets a failing step pass; no step in a required job ignores an exit code (X22)")
			}
		case "run":
			for i, l := range strings.Split(val.Value, "\n") {
				ln := val.Line + i
				if val.Style == yaml.LiteralStyle || val.Style == yaml.FoldedStyle {
					ln = val.Line + 1 + i
				}
				code := stripShellComment(l)
				if ignoreExit.MatchString(code) {
					add(ln, "%q ignores an exit code (X22)", strings.TrimSpace(l))
				}
				if pipInstall.MatchString(code) && !strings.Contains(code, "--require-hashes") {
					add(ln, "%q installs without --require-hashes (X21)", strings.TrimSpace(l))
				}
				if goTestShort.MatchString(code) {
					add(ln, "%q skips tests; a skipped test in a required job is a failure (X16)", strings.TrimSpace(l))
				}
			}
		case "env":
			if val.Kind == yaml.MappingNode {
				for i := 0; i+1 < len(val.Content); i += 2 {
					k := val.Content[i]
					if skipEnv.MatchString(k.Value) {
						add(k.Line, "%s turns a check off; a skipped check in a required job is a failure (X16)", k.Value)
					}
				}
			}
		case "shell":
			if v := strings.TrimSpace(val.Value); v != "bash" && !strings.Contains(v, "-e") && v != "python" && v != "pwsh" {
				add(val.Line, "shell %q does not stop on the first failing command; use bash (X22)", v)
			}
		}
	})
	out = append(out, gateFindings(name, src, &root)...)
	sort.Slice(out, func(i, j int) bool { return out[i].line < out[j].line })
	var s []string
	for _, f := range out {
		s = append(s, f.String())
	}
	return s, nil
}

// stripShellComment drops a trailing shell comment, so "# || true" in prose is not code.
func stripShellComment(l string) string {
	t := strings.TrimSpace(l)
	if strings.HasPrefix(t, "#") {
		return ""
	}
	if i := strings.Index(l, " #"); i >= 0 {
		return l[:i]
	}
	return l
}

func walk(n *yaml.Node, f func(k, v *yaml.Node)) {
	if n.Kind == yaml.MappingNode {
		for i := 0; i+1 < len(n.Content); i += 2 {
			f(n.Content[i], n.Content[i+1])
		}
	}
	for _, c := range n.Content {
		walk(c, f)
	}
}

// gateFindings checks X16 for a gate workflow: it declares its trigger paths in a
// "# gate-paths: a, b" comment, and its pull_request paths equal them. Removing a trigger
// then needs a change to the declaration, which is a reviewed change to the check's paths.
func gateFindings(name string, src []byte, root *yaml.Node) []finding {
	base := name
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	m := gatePaths.FindSubmatch(src)
	if !strings.HasPrefix(base, "gate") && m == nil {
		return nil
	}
	if m == nil {
		return []finding{{name, 1, "a gate workflow declares its trigger paths in a \"# gate-paths:\" comment (X16)"}}
	}
	var declared []string
	for _, p := range strings.Split(string(m[1]), ",") {
		if p = strings.TrimSpace(p); p != "" {
			declared = append(declared, p)
		}
	}
	var got []string
	line := 1
	if len(root.Content) == 1 {
		walk(root.Content[0], func(k, v *yaml.Node) {
			if k.Value == "on" || k.Value == "true" { // yaml.v3 keeps "on" as a string key
				walk(v, func(k2, v2 *yaml.Node) {
					if k2.Value == "pull_request" || k2.Value == "pull_request_target" {
						walk(v2, func(k3, v3 *yaml.Node) {
							if k3.Value == "paths" {
								line = v3.Line
								for _, c := range v3.Content {
									got = append(got, c.Value)
								}
							}
						})
					}
				})
			}
		})
	}
	sort.Strings(declared)
	sort.Strings(got)
	if strings.Join(declared, ",") != strings.Join(got, ",") {
		return []finding{{name, line, fmt.Sprintf("pull_request paths %v differ from the declared gate paths %v (X16)", got, declared)}}
	}
	return nil
}

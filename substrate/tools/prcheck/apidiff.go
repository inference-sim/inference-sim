package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// The API-compatibility check of the contracts/ packages against the last substrate tag
// (SUB-1, design §12.8). Between contract-change review sets the packages change only
// compatibly, so every incompatible change apidiff reports is a finding. A contract-change
// review set names its issue with -accept-break; the findings are then printed and accepted.
//
// apidiff runs at a pinned version through go run, as the router fork's check does. It is a
// step of this check and not a dependency: no package under contracts/ imports it and go.mod
// does not require it.

// apidiffRef is the pinned apidiff. It requires a newer Go than some developer machines have;
// every CI job installs the Go that contracts/toolchain names, which is newer, so go run needs
// no other toolchain there (LIF-2).
const apidiffRef = "golang.org/x/exp/cmd/apidiff@v0.0.0-20260908205506-85c1c2202aba"

var (
	releaseTag = regexp.MustCompile(`^v(\d+)\.(\d+)\.(\d+)$`)
	// A review set is named by its issue: owner/repo#n.
	reviewSetRef = regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$`)
)

// apidiffTool runs apidiff and the go list that finds the packages. dir is where the command
// runs. The real one shells out; a test supplies its own.
type apidiffTool interface {
	// Packages lists the importable, non-internal, non-main packages under contracts/ in dir.
	Packages(dir string) ([]string, error)
	// Export writes the export data of package pkg of the module in dir to file.
	Export(dir, pkg, file string) error
	// Incompatible returns the incompatible changes from old to new export data.
	Incompatible(old, new string) (string, error)
}

// LastReleaseTag returns the newest vX.Y.Z tag (no pre-release suffix) of tags, or "".
func LastReleaseTag(tags []string) string {
	type ver struct{ maj, min, pat int }
	var best ver
	found := ""
	for _, t := range tags {
		m := releaseTag.FindStringSubmatch(t)
		if m == nil {
			continue
		}
		a, _ := strconv.Atoi(m[1])
		b, _ := strconv.Atoi(m[2])
		c, _ := strconv.Atoi(m[3])
		v := ver{a, b, c}
		if found == "" || v.maj > best.maj || v.maj == best.maj && (v.min > best.min || v.min == best.min && v.pat > best.pat) {
			best, found = v, t
		}
	}
	return found
}

// CheckContractsAPI compares the packages of the module in newDir with those of the module
// in oldDir, which holds the last tag. It returns one finding per incompatible change and per
// removed package.
func CheckContractsAPI(tool apidiffTool, oldDir, newDir, scratch string) ([]string, error) {
	oldPkgs, err := tool.Packages(oldDir)
	if err != nil {
		return nil, err
	}
	newPkgs, err := tool.Packages(newDir)
	if err != nil {
		return nil, err
	}
	if len(oldPkgs) == 0 {
		return nil, fmt.Errorf("the last tag has no package under contracts/; the check processes at least one (X22)")
	}
	have := map[string]bool{}
	for _, p := range newPkgs {
		have[p] = true
	}
	var bad []string
	for i, p := range oldPkgs {
		if !have[p] {
			bad = append(bad, fmt.Sprintf("%s: the package was removed", p))
			continue
		}
		oldFile := filepath.Join(scratch, fmt.Sprintf("old-%d", i))
		newFile := filepath.Join(scratch, fmt.Sprintf("new-%d", i))
		if err := tool.Export(oldDir, p, oldFile); err != nil {
			return nil, err
		}
		if err := tool.Export(newDir, p, newFile); err != nil {
			return nil, err
		}
		out, err := tool.Incompatible(oldFile, newFile)
		if err != nil {
			return nil, err
		}
		for _, l := range strings.Split(strings.TrimSpace(out), "\n") {
			if l = strings.TrimSpace(l); l != "" {
				bad = append(bad, p+": "+l)
			}
		}
	}
	sort.Strings(bad)
	return bad, nil
}

// JudgeContractsAPI turns the findings into the check's verdict. With a review set named, an
// incompatible change is accepted and reported; without one it is a failure (SUB-1).
func JudgeContractsAPI(findings []string, prev, acceptBreak string) ([]string, string, error) {
	if acceptBreak != "" && !reviewSetRef.MatchString(acceptBreak) {
		return nil, "", fmt.Errorf("-accept-break %q is not an issue reference owner/repo#n; a contract-change review set names its issue (design §12.8)", acceptBreak)
	}
	if len(findings) == 0 {
		return nil, fmt.Sprintf("contracts-api: no incompatible change since %s", prev), nil
	}
	if acceptBreak != "" {
		return nil, fmt.Sprintf("contracts-api: %d incompatible change(s) since %s, accepted for the contract-change review set %s:\n%s", len(findings), prev, acceptBreak, strings.Join(findings, "\n")), nil
	}
	msg := make([]string, 0, len(findings))
	for _, f := range findings {
		msg = append(msg, fmt.Sprintf("%s (incompatible since %s; only a contract-change review set may break the contract, SUB-1)", f, prev))
	}
	return msg, "", nil
}

// goTool is apidiffTool over the go command.
type goTool struct{}

func (goTool) run(dir string, args ...string) (string, error) {
	cmd := exec.Command("go", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=mod", "GOTOOLCHAIN=local")
	out, err := cmd.Output()
	if err != nil {
		var stderr string
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = strings.TrimSpace(string(ee.Stderr))
		}
		return "", fmt.Errorf("go %s: %v: %s", strings.Join(args, " "), err, stderr)
	}
	return string(out), nil
}

func (g goTool) Packages(dir string) ([]string, error) {
	out, err := g.run(dir, "list", "-f", "{{.ImportPath}} {{.Name}}", "./contracts/...")
	if err != nil {
		return nil, err
	}
	var pkgs []string
	for _, l := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Fields(l)
		if len(f) != 2 || f[1] == "main" || strings.Contains(f[0]+"/", "/internal/") {
			continue
		}
		pkgs = append(pkgs, f[0])
	}
	sort.Strings(pkgs)
	return pkgs, nil
}

func (g goTool) Export(dir, pkg, file string) error {
	_, err := g.run(dir, "run", apidiffRef, "-w", file, pkg)
	return err
}

func (g goTool) Incompatible(old, new string) (string, error) {
	return g.run(".", "run", apidiffRef, "-incompatible", old, new)
}

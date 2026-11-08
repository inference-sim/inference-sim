package main

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/mod/semver"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// Kind is what a repository releases, which decides its tag scheme (design §14.1).
type Kind string

const (
	KindSubstrate Kind = "substrate" // vX.Y.Z
	KindEngine    Kind = "engine"    // v<upstream>-blis.N (LIF-5)
	KindRouter    Kind = "router"    // v<upstream>-blis.N (LIF-5)
	KindSimulator Kind = "simulator" // vX.Y.Z
	KindApp       Kind = "app"       // vX.Y.Z
)

// Repos maps each BLIS repository to its kind, its module path where it has one, and its
// layer, lowest first (design Figure 1; X10: released bottom-up).
var Repos = map[string]struct {
	Kind   Kind
	Module string
	Layer  int
}{
	"blis-substrate":     {KindSubstrate, "github.com/inference-sim/blis-substrate", 0},
	"blis-vllm":          {KindEngine, "github.com/inference-sim/blis-vllm", 1},
	"blis-sglang":        {KindEngine, "github.com/inference-sim/blis-sglang", 1},
	"llm-d-router":       {KindRouter, "github.com/llm-d/llm-d-router", 1},
	"inference-sim":      {KindSimulator, "github.com/inference-sim/inference-sim", 2},
	"accuracy-dashboard": {KindApp, "", 3},
	"blis-config-search": {KindApp, "", 3},
}

var (
	engineTag = regexp.MustCompile(`^(v[0-9]+\.[0-9]+\.[0-9]+)-((?:post|rev)\.[1-9][0-9]*\.)?blis(?:\.([1-9][0-9]*))?$`)
	plainTag  = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+$`)
)

// upstreamOf returns the upstream release an engine tag models, as stack.lock writes it
// ("v0.32.0", "v0.5.6-post.1", "v0.10.1-rev.1"), and its N.
func upstreamOf(tag string) (string, int, bool) {
	m := engineTag.FindStringSubmatch(tag)
	if m == nil {
		return "", 0, false
	}
	// The router fork's bare v1.0.0-blis has no number; it sorts below -blis.1 and counts as 0.
	n, _ := strconv.Atoi(m[3])
	up := m[1]
	if m[2] != "" {
		up += "-" + strings.TrimSuffix(m[2], ".")
	}
	return up, n, true
}

// engineTagFor returns the tag for release N of an upstream release.
func engineTagFor(upstream string, n int) string {
	if strings.Contains(upstream, "-") {
		return fmt.Sprintf("%s.blis.%d", upstream, n)
	}
	return fmt.Sprintf("%s-blis.%d", upstream, n)
}

// CheckEngineTag applies LIF-5's tag rule: the form v<upstream>-blis.N (or the post and
// rev forms), N one above the highest N on the same upstream release (1 when there is
// none), and never a plain v<upstream>.
func CheckEngineTag(tag string, existing []string) []string {
	if plainTag.MatchString(tag) {
		return []string{fmt.Sprintf("%s is a plain tag; an engine or router tag is v<upstream>-blis.N (LIF-5)", tag)}
	}
	up, n, ok := upstreamOf(tag)
	if !ok {
		return []string{fmt.Sprintf("%s is not v<upstream>-blis.N, -post.K.blis.N or -rev.W.blis.N with N, K, W from 1 (LIF-5)", tag)}
	}
	high := 0
	for _, t := range existing {
		if t == tag {
			return []string{fmt.Sprintf("%s already exists; a tag names one commit forever (LIF-1)", tag)}
		}
		if u, m, ok := upstreamOf(t); ok && u == up && m > high {
			high = m
		}
	}
	if n != high+1 {
		return []string{fmt.Sprintf("%s: N must be %d, one above the highest -blis.N on %s (LIF-5)", tag, high+1, up)}
	}
	return nil
}

// NextSemver returns the next vX.Y.Z after the highest release tag in existing.
func NextSemver(existing []string, bump string) (string, error) {
	high := "v0.0.0"
	for _, t := range existing {
		if semver.IsValid(t) && semver.Prerelease(t) == "" && semver.Build(t) == "" && semver.Compare(t, high) > 0 {
			high = t
		}
	}
	parts := strings.Split(strings.TrimPrefix(semver.Canonical(high), "v"), ".")
	x, _ := strconv.Atoi(parts[0])
	y, _ := strconv.Atoi(parts[1])
	z, _ := strconv.Atoi(parts[2])
	switch bump {
	case "major":
		x, y, z = x+1, 0, 0
	case "minor":
		y, z = y+1, 0
	case "patch":
		z++
	default:
		return "", fmt.Errorf("bump %q is not major, minor or patch", bump)
	}
	return fmt.Sprintf("v%d.%d.%d", x, y, z), nil
}

// breaksCompatibility reports whether a semver release may break compatibility (LIF-7): a
// MAJOR release, or a MINOR release at v0.
func breaksCompatibility(prev, next string) bool {
	if semver.Major(prev) != semver.Major(next) {
		return true
	}
	return semver.Major(next) == "v0" && semver.MajorMinor(prev) != semver.MajorMinor(next)
}

// Deprecation is one deprecated feature found in a tree.
type Deprecation struct {
	File string
	Line int
	Text string
}

var deprecatedMarker = regexp.MustCompile(`(?:^\s*//\s*Deprecated:|\bDeprecated:\s*"|MarkDeprecated\(|\bDEPRECATED\b)`)

// FindDeprecations returns the deprecation markers in Go source: a // Deprecated: comment,
// a cobra command's Deprecated field, a pflag MarkDeprecated call, or a flag usage that
// says DEPRECATED (LIF-7 says each deprecation is recorded where the feature is defined).
func FindDeprecations(file string, src []byte) []Deprecation {
	if !strings.HasSuffix(file, ".go") || strings.HasSuffix(file, "_test.go") {
		return nil
	}
	var out []Deprecation
	for i, l := range strings.Split(string(src), "\n") {
		if deprecatedMarker.MatchString(l) {
			out = append(out, Deprecation{file, i + 1, strings.TrimSpace(l)})
		}
	}
	return out
}

// Relaxation is one feature the release plan lets stay past a compatibility-breaking release,
// until the milestone named (plan §3.3, "Relaxing the deprecation rule until M4").
type Relaxation struct {
	Repo      string
	Path      string // a path prefix in that repository
	Match     string // text the marker line must contain; empty matches any marker under Path
	Milestone int
	Reason    string
}

// CheckDeprecations refuses a compatibility-breaking release that still contains a
// deprecated feature (LIF-7), except those the plan's relaxation list covers while the
// plan's current milestone is before the relaxation's end.
func CheckDeprecations(repo string, found []Deprecation, relax []Relaxation, milestone int) []string {
	var bad []string
	for _, d := range found {
		covered := false
		for _, r := range relax {
			if r.Repo == repo && strings.HasPrefix(d.File, r.Path) && (r.Match == "" || strings.Contains(d.Text, r.Match)) && milestone < r.Milestone {
				covered = true
			}
		}
		if !covered {
			bad = append(bad, fmt.Sprintf("%s:%d: a deprecated feature remains (%s); a MAJOR release, or a MINOR release at v0, removes it (LIF-7)", d.File, d.Line, d.Text))
		}
	}
	return bad
}

// CheckInvariants refuses a tag unless every invariant check the repository owns with
// status ci passed at the commit (LIF-3, design §13.3), and, for each invariant whose target
// milestone has been reached, unless the invariant has its target status. A milestone is
// reached when its milestone release is cut (plan §3.1): the milestones before the current
// one have been, and the current one is reached by its own milestone release. Releases
// within a milestone are not held to that milestone's targets, as the plan schedules.
// passed maps a check's test name to whether it passed at the commit.
func CheckInvariants(reg *invariants.Registry, repo string, passed map[string]bool, milestone int, milestoneRelease bool) []string {
	var bad []string
	for _, inv := range reg.Invariants {
		owns := false
		for _, c := range inv.Checks {
			if c.Repo != repo {
				continue
			}
			owns = true
			if c.Status == invariants.StatusCI && c.Runs != invariants.RunsPractice {
				ok, seen := passed[c.Test]
				switch {
				case !seen:
					bad = append(bad, fmt.Sprintf("%s: check %q has no recorded result at this commit (LIF-3)", inv.ID, c.Test))
				case !ok:
					bad = append(bad, fmt.Sprintf("%s: check %q failed at this commit (LIF-3)", inv.ID, c.Test))
				}
			}
		}
		// The gate reads automated checks only: a maintainer practice has no check run to wait
		// for (design §13.1), so an invariant is held to its target by the status its automated
		// checks give, and one with none is not held.
		auto, autoCI := 0, 0
		for _, c := range inv.Checks {
			if c.Runs == invariants.RunsPractice {
				continue
			}
			auto++
			if c.Status == invariants.StatusCI {
				autoCI++
			}
		}
		got := invariants.StatusOf(autoCI, auto)
		reached := inv.Target.Milestone != nil && (*inv.Target.Milestone < milestone || milestoneRelease && *inv.Target.Milestone == milestone)
		if owns && auto > 0 && reached && got != inv.Target.Status {
			bad = append(bad, fmt.Sprintf("%s: milestone M%d is reached and its automated checks are %s, not the target %s (design §13.3)", inv.ID, *inv.Target.Milestone, got, inv.Target.Status))
		}
	}
	sort.Strings(bad)
	return dedupe(bad)
}

func dedupe(s []string) []string {
	var out []string
	for i, x := range s {
		if i == 0 || x != s[i-1] {
			out = append(out, x)
		}
	}
	return out
}

// Bug is an open issue labelled as breaking an invariant (X27).
type Bug struct {
	Repo   string
	Number int
	Title  string
}

// CheckBugs refuses a release while an open invariant bug lives in the repository or in a
// repository whose code it ships, unless the release fixes only that bug (fixes lists the
// issues this release closes, and it closes nothing else) or a release owner records a
// waiver (waived lists issues waived in the release notes).
func CheckBugs(bugs []Bug, fixes []int, waived []int, security bool) []string {
	if security {
		return nil
	}
	in := func(n int, l []int) bool {
		for _, x := range l {
			if x == n {
				return true
			}
		}
		return false
	}
	var bad []string
	for _, b := range bugs {
		if in(b.Number, waived) {
			continue
		}
		if len(fixes) == 1 && fixes[0] == b.Number {
			continue
		}
		bad = append(bad, fmt.Sprintf("open invariant bug %s#%d (%s) blocks this release; fix it, release only its fix, or record a release owner's waiver (X27)", b.Repo, b.Number, b.Title))
	}
	return bad
}

// CheckSubstrateBound refuses an engine or router release whose substrate requirement is
// newer than inference-sim's (design §3.2): such a release would move the simulator's data.
func CheckSubstrateBound(engineReq, simReq string) []string {
	if engineReq == "" {
		return nil
	}
	if simReq == "" {
		return []string{"inference-sim's substrate requirement is unknown, so the bound of design §3.2 cannot be checked"}
	}
	if semver.Compare(engineReq, simReq) > 0 {
		return []string{fmt.Sprintf("this release requires blis-substrate %s, newer than inference-sim's %s; move inference-sim first, in the same review set (design §3.2)", engineReq, simReq)}
	}
	return nil
}

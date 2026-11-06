package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
	"github.com/inference-sim/blis-substrate/tools/internal/bot"
)

// Env is what the bot needs that is not in git: the registry, the invariant check results
// at each commit, the open invariant bugs, and the plan's current milestone. A test supplies
// it directly; the command reads it from files and the GitHub API.
type Env struct {
	Registry *invariants.Registry
	// Passed returns, for a repository at a commit, each ci check's test name and whether
	// it passed there.
	Passed func(repo, commit string) (map[string]bool, error)
	// Bugs returns the open issues labelled invariant-bug in a repository and in each
	// repository whose code it ships.
	Bugs func(repo string) ([]Bug, error)
}

// Step is one repository's planned release.
type Step struct {
	Repo     string
	Dir      string // the local checkout
	Base     string // the branch the release merges into
	Head     string // the review set's head commit
	Prev     string // the previous release tag, "" for none
	Tag      string
	Pins     map[string]string // module path -> the tag this repository must pin, written once
	Notes    string
	Refusals []string
}

// Plan works out each repository's release from a review set, bottom-up, without changing
// anything. dirs maps each repository to its local checkout. A review set of one repository
// is a single-repository release.
func Plan(env Env, lock *StackLock, dirs map[string]string, base string) ([]Step, error) {
	var steps []Step
	tags := map[string]string{} // repo -> the tag this set gives it
	for _, r := range lock.Repos {
		dir, ok := dirs[r.Repo]
		if !ok {
			return nil, fmt.Errorf("%s: no local checkout given", r.Repo)
		}
		info := Repos[r.Repo]
		st := Step{Repo: r.Repo, Dir: dir, Base: base, Head: r.Head, Pins: map[string]string{}}
		if _, err := gitIn(dir, "cat-file", "-e", r.Head+"^{commit}"); err != nil {
			return nil, fmt.Errorf("%s: head %s is not in %s", r.Repo, r.Head, dir)
		}
		existing, err := tagList(dir)
		if err != nil {
			return nil, err
		}
		st.Prev = previousTag(existing, info.Kind, r.Upstream)
		// A head that already carries a release tag was released; planning it again would give
		// the same commit a second release tag.
		if at, err := gitIn(dir, "tag", "--points-at", r.Head); err == nil {
			for _, t := range strings.Fields(at) {
				if plainTag.MatchString(t) || engineTag.MatchString(t) {
					st.Refusals = append(st.Refusals, fmt.Sprintf("the head %s already carries the release tag %s; a commit is released once (LIF-1)", short(r.Head), t))
				}
			}
		}
		switch info.Kind {
		case KindEngine, KindRouter:
			_, n := highestN(existing, r.Upstream)
			st.Tag = engineTagFor(r.Upstream, n+1)
			st.Refusals = append(st.Refusals, CheckEngineTag(st.Tag, existing)...)
		default:
			if st.Tag, err = NextSemver(existing, r.Bump); err != nil {
				return nil, err
			}
			if st.Prev != "" && breaksCompatibility(st.Prev, st.Tag) {
				found, err := deprecationsAt(dir, r.Head)
				if err != nil {
					return nil, err
				}
				st.Refusals = append(st.Refusals, CheckDeprecations(r.Repo, found, relaxations, lock.Milestone)...)
			}
		}
		// LIF-3 and design §13.3: every ci check passed at the head.
		passed, err := env.Passed(r.Repo, r.Head)
		if err != nil {
			return nil, err
		}
		st.Refusals = append(st.Refusals, CheckInvariants(env.Registry, r.Repo, passed, lock.Milestone, lock.MilestoneRelease)...)
		// X27.
		bugs, err := env.Bugs(r.Repo)
		if err != nil {
			return nil, err
		}
		st.Refusals = append(st.Refusals, CheckBugs(bugs, r.Fixes, r.Waivers, r.Security)...)
		// X10: pin each lower repository in this set at the tag the set gives it. Each pin
		// is written once, by the bot, after the lower tag exists.
		gomod, err := gitIn(dir, "show", r.Head+":go.mod")
		if err == nil {
			mf, err := modfile.Parse("go.mod", []byte(gomod), nil)
			if err != nil {
				return nil, fmt.Errorf("%s: go.mod: %w", r.Repo, err)
			}
			for _, req := range mf.Require {
				for lower, tag := range tags {
					if Repos[lower].Module == req.Mod.Path {
						st.Pins[req.Mod.Path] = tag
					}
				}
			}
			// Every BLIS module a release requires is at a release tag. A pin the bot writes
			// replaces a requirement in the set; any other requirement at a pseudo-version
			// would ship in a permanent tag (X11, LIF-2).
			for _, req := range mf.Require {
				if _, pinned := st.Pins[req.Mod.Path]; !pinned && strings.HasPrefix(req.Mod.Path, "github.com/inference-sim/") && module.IsPseudoVersion(req.Mod.Version) {
					st.Refusals = append(st.Refusals, fmt.Sprintf("go.mod requires %s at the pseudo-version %s; a release requires BLIS modules at release tags (X11, LIF-2)", req.Mod.Path, req.Mod.Version))
				}
			}
			for _, rep := range mf.Replace {
				if rep.Old.Path == Repos["llm-d-router"].Module {
					if tag, ok := tags["llm-d-router"]; ok {
						st.Pins["replace:"+rep.Old.Path] = tag
					}
				}
			}
			// Design §3.2: an engine or router never requires a newer substrate than
			// inference-sim does.
			if info.Kind == KindEngine || info.Kind == KindRouter {
				st.Refusals = append(st.Refusals, CheckSubstrateBound(requirement(mf, Repos["blis-substrate"].Module, tags["blis-substrate"]), simSubstrate(lock, dirs, tags))...)
			}
		}
		notes, err := Notes(env.Registry, st, dir)
		if err != nil {
			return nil, err
		}
		st.Notes = notes
		tags[r.Repo] = st.Tag
		steps = append(steps, st)
	}
	return steps, nil
}

// requirement returns the version a go.mod requires of a module, or the tag this set gives
// it when the module is in the set (the pin the bot will write).
func requirement(mf *modfile.File, path, setTag string) string {
	for _, r := range mf.Require {
		if r.Mod.Path == path {
			if setTag != "" {
				return setTag
			}
			return r.Mod.Version
		}
	}
	return ""
}

// simSubstrate returns the substrate version inference-sim will require: the set's
// substrate tag if inference-sim is in the set, otherwise its current requirement.
func simSubstrate(lock *StackLock, dirs map[string]string, tags map[string]string) string {
	sub := Repos["blis-substrate"].Module
	for _, r := range lock.Repos {
		if r.Repo == "inference-sim" {
			if t, ok := tags["blis-substrate"]; ok {
				return t
			}
			gm, err := gitIn(dirs["inference-sim"], "show", r.Head+":go.mod")
			if err != nil {
				return ""
			}
			mf, err := modfile.Parse("go.mod", []byte(gm), nil)
			if err != nil {
				return ""
			}
			return requirement(mf, sub, "")
		}
	}
	dir, ok := dirs["inference-sim"]
	if !ok {
		return ""
	}
	gm, err := os.ReadFile(filepath.Join(dir, "go.mod"))
	if err != nil {
		return ""
	}
	mf, err := modfile.Parse("go.mod", gm, nil)
	if err != nil {
		return ""
	}
	return requirement(mf, sub, "")
}

func tagList(dir string) ([]string, error) {
	out, err := gitIn(dir, "tag", "--list")
	if err != nil {
		return nil, err
	}
	return strings.Fields(out), nil
}

func highestN(tags []string, upstream string) (string, int) {
	best, high := "", 0
	for _, t := range tags {
		if u, n, ok := upstreamOf(t); ok && u == upstream {
			if best == "" || n > high {
				best, high = t, n
			}
		}
	}
	return best, high
}

// latestEngineTag is the newest engine or router tag of any upstream release: the highest
// upstream by semver, then the highest N. It is the "previous release" of a first release on
// a new upstream release.
func latestEngineTag(tags []string) string {
	best, bestUp, bestN := "", "", -1
	for _, t := range tags {
		u, n, ok := upstreamOf(t)
		if !ok {
			continue
		}
		if c := semver.Compare(u, bestUp); best == "" || c > 0 || (c == 0 && n > bestN) {
			best, bestUp, bestN = t, u, n
		}
	}
	return best
}

func previousTag(tags []string, kind Kind, upstream string) string {
	if kind == KindEngine || kind == KindRouter {
		if t, _ := highestN(tags, upstream); t != "" {
			return t
		}
		return latestEngineTag(tags)
	}
	var rel []string
	for _, t := range tags {
		if plainTag.MatchString(t) {
			rel = append(rel, t)
		}
	}
	if len(rel) == 0 {
		return ""
	}
	sort.Slice(rel, func(i, j int) bool { return semver.Compare(rel[i], rel[j]) < 0 })
	return rel[len(rel)-1]
}

func deprecationsAt(dir, commit string) ([]Deprecation, error) {
	// One git grep over the commit's tree finds every candidate line; FindDeprecations
	// applies the exact rule to each. git grep exits 1 when nothing matches.
	cmd := exec.Command("git", "grep", "-n", "-I", "-E", `Deprecated:|MarkDeprecated\(|DEPRECATED`, commit, "--", "*.go", ":!*_test.go")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == 1 {
			return nil, nil
		}
		return nil, fmt.Errorf("git grep: %w", err)
	}
	var all []Deprecation
	for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		// <commit>:<file>:<line>:<text>
		parts := strings.SplitN(strings.TrimPrefix(l, commit+":"), ":", 3)
		if len(parts) != 3 {
			continue
		}
		for _, d := range FindDeprecations(parts[0], []byte(parts[2])) {
			fmt.Sscan(parts[1], &d.Line)
			all = append(all, d)
		}
	}
	return all, nil
}

// Branch is the branch Stage builds a repository's release commit on: the base branch plus
// the head's merge and the pin commit, with no tag (LIF-3).
func Branch(st Step) string { return "release/" + st.Tag }

// Stage builds each repository's final release commit, in the local checkouts, and tags
// nothing. It merges the head into the base with a merge commit and writes the pins the plan
// names in one commit, both on the branch Branch names; the base branch does not move. The
// commit it leaves is not the head Plan checked, so it has no recorded results yet. Tag, a
// later step, tags the branch tip only after every required result is recorded for that
// exact commit (LIF-3: a tag only on a commit where every ci invariant passed). Stage
// refuses to start if any step has a refusal. It never pushes. The loop stages every layer
// locally; the push of each layer's branch, the CI run on it and the next layer's checks
// against the new tags are a manual step until the release job exists (design §13.6).
func Stage(steps []Step, sign bool) error {
	for _, st := range steps {
		if len(st.Refusals) > 0 {
			return fmt.Errorf("%s: %d refusal(s); nothing was changed", st.Repo, len(st.Refusals))
		}
	}
	ident := []string{"-c", "user.name=" + bot.Name, "-c", "user.email=" + bot.Email}
	for _, st := range steps {
		if _, err := gitIn(st.Dir, "checkout", "-q", "-B", Branch(st), st.Base); err != nil {
			return err
		}
		if _, err := gitIn(st.Dir, append(ident, "merge", "--no-ff", "-q", "-m", fmt.Sprintf("Merge %s for %s", short(st.Head), st.Tag), st.Head)...); err != nil {
			return fmt.Errorf("%s: merge: %w", st.Repo, err)
		}
		if len(st.Pins) > 0 {
			if err := writePins(st.Dir, st.Pins); err != nil {
				return fmt.Errorf("%s: %w", st.Repo, err)
			}
			if _, err := gitIn(st.Dir, "add", "go.mod"); err != nil {
				return err
			}
			args := append(ident, "commit", "-q", "-m", "Pin "+pinSummary(st.Pins)+" for "+st.Tag)
			if sign {
				args = append(args, "-S")
			}
			if _, err := gitIn(st.Dir, args...); err != nil {
				return fmt.Errorf("%s: pin commit: %w", st.Repo, err)
			}
		}
	}
	return nil
}

// Tag makes the annotated tag, whose message is the release notes, on the tip of each step's
// staged branch, and on nothing else. Before it tags a step it checks that the tip still
// contains the planned head and that the environment records a passing result for every ci
// check at the tip's own SHA (LIF-3, design §13.3). A tip that moved since it was checked
// has no results and is refused. Nothing is tagged unless every step passes.
func Tag(env Env, lock *StackLock, steps []Step, sign bool) error {
	tips := make([]string, len(steps))
	for i, st := range steps {
		tip, err := gitIn(st.Dir, "rev-parse", "--verify", "refs/heads/"+Branch(st)+"^{commit}")
		if err != nil {
			return fmt.Errorf("%s: no staged release branch %s: %w", st.Repo, Branch(st), err)
		}
		if _, err := gitIn(st.Dir, "merge-base", "--is-ancestor", st.Head, tip); err != nil {
			return fmt.Errorf("%s: staged tip %s does not contain the planned head %s", st.Repo, short(tip), short(st.Head))
		}
		passed, err := env.Passed(st.Repo, tip)
		if err != nil {
			return err
		}
		if bad := CheckInvariants(env.Registry, st.Repo, passed, lock.Milestone, lock.MilestoneRelease); len(bad) > 0 {
			return fmt.Errorf("%s: staged tip %s is not checked: %s", st.Repo, short(tip), strings.Join(bad, "; "))
		}
		tips[i] = tip
	}
	ident := []string{"-c", "user.name=" + bot.Name, "-c", "user.email=" + bot.Email}
	for i, st := range steps {
		args := append(ident, "tag", "-a", st.Tag, "-F", "-", tips[i])
		if sign {
			args = append(ident, "tag", "-s", st.Tag, "-F", "-", tips[i])
		}
		cmd := exec.Command("git", args...)
		cmd.Dir = st.Dir
		cmd.Stdin = strings.NewReader(st.Notes)
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("%s: tag %s: %v: %s", st.Repo, st.Tag, err, out)
		}
	}
	return nil
}

func pinSummary(p map[string]string) string {
	var s []string
	for k, v := range p {
		s = append(s, strings.TrimPrefix(k, "replace:")+" "+v)
	}
	sort.Strings(s)
	return strings.Join(s, ", ")
}

// writePins rewrites go.mod's requirements (and the router replace) to the set's tags. The
// go.sum entries are added by go mod tidy in the next layer's checks, which run against the
// new tags once they are pushed.
func writePins(dir string, pins map[string]string) error {
	path := filepath.Join(dir, "go.mod")
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	mf, err := modfile.Parse(path, raw, nil)
	if err != nil {
		return err
	}
	for k, v := range pins {
		if old, ok := strings.CutPrefix(k, "replace:"); ok {
			if err := mf.AddReplace(old, "", Repos["llm-d-router"].Module, ""); err != nil {
				return err
			}
			if err := mf.AddReplace(old, "", "github.com/inference-sim/llm-d-router", v); err != nil {
				return err
			}
			continue
		}
		if err := mf.AddRequire(k, v); err != nil {
			return err
		}
	}
	mf.Cleanup()
	out, err := mf.Format()
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0o644)
}

func gitIn(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return strings.TrimSpace(string(out)), nil
}

func short(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	return h
}

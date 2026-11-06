// Command release is the BLIS release bot (design §13.6). It releases one repository, or a
// review set that spans repositories, described by a stack.lock file.
//
//	go run ./tools/release -lock stack.lock -dir blis-substrate=../blis-substrate \
//	    -dir blis-vllm=../blis-vllm -results results.json [-bugs bugs.json] [-base main]
//	    [-apply | -tag] [-layer <repo>] [-sign]
//
// Without -apply or -tag it is a dry run: it prints, for each repository bottom-up (X10), the
// tag it would make, the pins it would write, the refusals that stop it, and the release
// notes (LIF-4), and changes nothing. A repository whose tag is already on its staged tip is
// printed as done and skipped.
//
// A set is released one layer at a time, and every run re-plans the whole lock. -layer names
// the repository; without it, the first one in lock order that is not yet tagged. With
// -apply, and only when nothing is refused, it stages that layer's final release commit on
// the branch release/<tag>: the head merged into the base and the pins in one commit, with no
// tag. Every layer below must already be tagged and its tag published on its origin (git
// ls-remote), because the pins are those real tags. With -tag it tags that layer's staged
// tip, and only where -results records every ci check at the tip's own SHA (LIF-3) and
// where the tip no longer holds the pin-by-commit deferral file and passes the selected-module
// lint (-modlint, no deferral: the whole module graph, not only the direct requirements, so a
// transitive BLIS pseudo-version or a BLIS replace is refused). -sign signs the pin commits
// and tags. It never pushes. A run interrupted at any point is resumed by running the same
// command again: the version numbers do not advance, a staged branch is continued, and a
// tagged layer is skipped.
//
// The release job does not exist yet (it arrives in M2). After each command the tool prints
// what to do by hand: push the branch, run the layer's checks on it and record the results,
// run -tag, push the tag, run the layer's checks against the tag, then run -apply for the
// next layer. The push needs a bypass of the branch and tag protection for the bot or the
// owners.
//
// It refuses a tag when:
//   - a ci check of an invariant the repository owns has no passing result at the head
//     (LIF-3), or a reached target milestone has not been met (design §13.3); -results is a
//     JSON object {"<repo>": {"<commit>": {"<test>": true|false}}} written by CI;
//   - an engine or router tag breaks LIF-5: not v<upstream>-blis.N, N not one above the
//     previous on the same upstream release, or a plain tag;
//   - an engine or router release requires a newer substrate than inference-sim (§3.2);
//   - a MAJOR release, or a MINOR release at v0, still contains a deprecated feature, apart
//     from the relaxations of plan §3.3 (LIF-7);
//   - an open issue labelled invariant-bug blocks the repository (X27); -bugs is a JSON
//     object {"<repo>": [{"Repo":..., "Number":..., "Title":...}]}.
//
// Exit status: 0 the release can go ahead (or was applied), 1 refused, 2 usage or I/O error.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

type dirFlag map[string]string

func (d dirFlag) String() string { return "" }
func (d dirFlag) Set(v string) error {
	k, p, ok := strings.Cut(v, "=")
	if !ok || k == "" || p == "" {
		return fmt.Errorf("-dir wants <repo>=<path>, got %q", v)
	}
	d[k] = p
	return nil
}

func main() {
	code, err := run(os.Args[1:], os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, "release:", err)
	}
	os.Exit(code)
}

func run(args []string, w io.Writer) (int, error) {
	fs := flag.NewFlagSet("release", flag.ContinueOnError)
	lockPath := fs.String("lock", "", "the review set's stack.lock")
	results := fs.String("results", "", "JSON of each repository's check results at each commit")
	bugsPath := fs.String("bugs", "", "JSON of open invariant bugs per repository")
	base := fs.String("base", "main", "the branch each head merges into")
	apply := fs.Bool("apply", false, "stage the release commit on release/<tag> in the local checkouts: merge and pin, no tag (never pushes)")
	tag := fs.Bool("tag", false, "tag the staged release/<tag> tips, only where -results records every ci check at the tip's own SHA")
	modlint := fs.String("modlint", "", "the tools/modlint binary, built from the substrate version being released; -tag runs it, with no deferral, on each staged tip")
	layer := fs.String("layer", "", "the repository -apply or -tag works on; default: the first layer in lock order that is not yet tagged")
	sign := fs.Bool("sign", false, "sign the pin commits and tags")
	dirs := dirFlag{}
	fs.Var(dirs, "dir", "<repo>=<local checkout>, once per repository in the set")
	if err := fs.Parse(args); err != nil {
		return 2, err
	}
	if *lockPath == "" || *results == "" {
		return 2, fmt.Errorf("-lock and -results are required")
	}
	raw, err := os.ReadFile(*lockPath)
	if err != nil {
		return 2, err
	}
	lock, err := ParseStackLock(raw)
	if err != nil {
		return 2, err
	}
	reg, err := invariants.Embedded()
	if err != nil {
		return 2, err
	}
	var res map[string]map[string]map[string]bool
	if err := readJSON(*results, &res); err != nil {
		return 2, err
	}
	bugs := map[string][]Bug{}
	if *bugsPath != "" {
		if err := readJSON(*bugsPath, &bugs); err != nil {
			return 2, err
		}
	}
	env := Env{
		Registry: reg,
		Passed: func(repo, commit string) (map[string]bool, error) {
			return res[repo][commit], nil
		},
		Bugs: func(repo string) ([]Bug, error) { return bugs[repo], nil },
	}
	if *modlint != "" {
		env.Lint = func(dir string) ([]string, error) { return runModlint(*modlint, dir) }
	}
	env.TagPublished = func(repo, tag string) bool { return tagPublished(dirs[repo], tag) }
	steps, err := Plan(env, lock, dirs, *base)
	if err != nil {
		return 2, err
	}
	refused := false
	for _, st := range steps {
		state := ""
		switch {
		case st.Done:
			state = " [done: already tagged]"
		case st.Staged:
			state = " [staged]"
		}
		_, _ = fmt.Fprintf(w, "== %s: %s (previous %s)%s\n", st.Repo, st.Tag, orNone(st.Prev), state)
		_, _ = fmt.Fprintf(w, "pins: %s\n", orNone(pinSummary(st.Pins)))
		for _, r := range st.Refusals {
			_, _ = fmt.Fprintf(w, "REFUSED: %s\n", r)
			refused = true
		}
		_, _ = fmt.Fprintf(w, "--- notes\n%s--- end notes\n", st.Notes)
	}
	if refused {
		return 1, nil
	}
	switch {
	case *apply && *tag:
		return 2, fmt.Errorf("-apply and -tag are separate steps: stage a layer, run its CI, then tag it")
	case *apply:
		if err := Stage(env, steps, *layer, *sign); err != nil {
			return 1, err
		}
		i, _ := pickLayer(steps, *layer)
		st := steps[i]
		tip, _ := gitIn(st.Dir, "rev-parse", "refs/heads/"+Branch(st))
		_, _ = fmt.Fprintf(w, "staged %s %s at %s; no tag yet, nothing pushed\n", st.Repo, Branch(st), tip)
		_, _ = fmt.Fprintf(w, "next: push %s, run %s's checks on it and record the results for %s, then run -tag -layer %s\n", Branch(st), st.Repo, short(tip), st.Repo)
		return 0, nil
	case *tag:
		if err := Tag(env, lock, steps, *layer, *sign); err != nil {
			return 1, err
		}
		i, _ := pickLayer(steps, *layer)
		st := steps[i]
		_, _ = fmt.Fprintf(w, "tagged %s %s; nothing pushed\n", st.Repo, st.Tag)
		if next := nextLayer(steps, i); next != "" {
			_, _ = fmt.Fprintf(w, "next: push %s, run %s's checks against it, then run -apply -layer %s\n", st.Tag, st.Repo, next)
		} else {
			_, _ = fmt.Fprintf(w, "next: push %s and run %s's checks against it; the set is complete\n", st.Tag, st.Repo)
		}
		return 0, nil
	}
	_, _ = fmt.Fprintln(w, "dry run: nothing changed")
	return 0, nil
}

// nextLayer names the repository after step i that is not yet tagged, or "".
func nextLayer(steps []Step, i int) string {
	for _, st := range steps[i+1:] {
		if !st.Done {
			return st.Repo
		}
	}
	return ""
}

// tagPublished reports whether the tag is on the origin of the checkout.
func tagPublished(dir, tag string) bool {
	cmd := exec.Command("git", "ls-remote", "--exit-code", "--tags", "origin", tag)
	cmd.Dir = dir
	return cmd.Run() == nil
}

// runModlint runs the modlint binary on a checkout, without -deferral. Exit 0 is clean, 1
// lists findings on standard output, and anything else means the lint could not run.
func runModlint(bin, dir string) ([]string, error) {
	cmd := exec.Command(bin, "-C", dir)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	var ee *exec.ExitError
	switch {
	case err == nil:
		return nil, nil
	case errors.As(err, &ee) && ee.ExitCode() == 1:
		return strings.Split(strings.TrimSpace(string(out)), "\n"), nil
	}
	return nil, fmt.Errorf("modlint: %v: %s", err, strings.TrimSpace(stderr.String()))
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

func readJSON(path string, v any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// Command release is the BLIS release bot (design §13.6). It releases one repository, or a
// review set that spans repositories, described by a stack.lock file.
//
//	go run ./tools/release -lock stack.lock -dir blis-substrate=../blis-substrate \
//	    -dir blis-vllm=../blis-vllm -results results.json [-bugs bugs.json] [-base main]
//	    [-apply | -tag] [-sign]
//
// Without -apply or -tag it is a dry run: it prints, for each repository bottom-up (X10), the
// tag it would make, the pins it would write, the refusals that stop it, and the release
// notes (LIF-4), and changes nothing. With -apply, and only when nothing is refused, it
// stages each repository's final release commit on the branch release/<tag>: the head merged
// into the base and the pins in one commit, with no tag. With -tag, it tags the tip of each
// staged branch, and only where -results records every ci check at that tip's own SHA
// (LIF-3); -sign signs the pin commits and tags. It never pushes.
//
// The release job does not exist yet. The loop stages every layer locally; pushing each
// layer's branch, running CI on it, running the next layer's checks against the new tags and
// pushing the tags are manual steps for the release owners until that job exists (design
// §13.6). The push needs a bypass of the branch and tag protection for the bot or the owners.
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
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
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
	steps, err := Plan(env, lock, dirs, *base)
	if err != nil {
		return 2, err
	}
	refused := false
	for _, st := range steps {
		fmt.Fprintf(w, "== %s: %s (previous %s)\n", st.Repo, st.Tag, orNone(st.Prev))
		fmt.Fprintf(w, "pins: %s\n", orNone(pinSummary(st.Pins)))
		for _, r := range st.Refusals {
			fmt.Fprintf(w, "REFUSED: %s\n", r)
			refused = true
		}
		fmt.Fprintf(w, "--- notes\n%s--- end notes\n", st.Notes)
	}
	if refused {
		return 1, nil
	}
	switch {
	case *apply && *tag:
		return 2, fmt.Errorf("-apply and -tag are separate steps: stage, run CI on the staged commits, then tag")
	case *apply:
		if err := Stage(steps, *sign); err != nil {
			return 2, err
		}
		for _, st := range steps {
			tip, _ := gitIn(st.Dir, "rev-parse", "refs/heads/"+Branch(st))
			_, _ = fmt.Fprintf(w, "staged %s %s at %s; no tag yet\n", st.Repo, Branch(st), tip)
		}
		_, _ = fmt.Fprintln(w, "staged in the local checkouts; nothing tagged, nothing pushed")
		return 0, nil
	case *tag:
		if err := Tag(env, lock, steps, *sign); err != nil {
			return 1, err
		}
		_, _ = fmt.Fprintln(w, "tagged the checked tips in the local checkouts; nothing pushed")
		return 0, nil
	}
	_, _ = fmt.Fprintln(w, "dry run: nothing changed")
	return 0, nil
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

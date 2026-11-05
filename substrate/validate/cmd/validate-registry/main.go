// Command validate-registry loads every committed coefficient set in a blis-registry
// checkout, and every acceptance-profile set under acceptance/ when the registry hosts any,
// and validates each against the schema.
//
// blis-registry carries its own Python validator, but every Go consumer loads these
// sets through contracts.LoadCoefficientSet and the field checks in
// (*coefficient.Set).Validate — among them the intra-set (name, scope) duplicate rule a
// resolver depends on. This command runs exactly those checks over a registry root, so
// the registry's own CI can reject a bad set on the pull request that introduces it,
// rather than leaving it to surface later in a consumer. It is the coefficient-set
// counterpart to validate-catalog.
//
// Discovery is registry.SetPaths, which matches blis-registry's own validator
// (validator/validate.py): a recursive walk under coefficients/ matching .yaml and .yml
// case-insensitively, de-duplicated by resolved path. Sharing that predicate is what keeps
// the two gates from diverging on the sets that exist — with one qualification: like any
// plain directory walk it does not descend into symlinked directories, so a set reachable
// only through a linked directory is out of scope (the registry stores regular files, so
// this does not arise today).
//
// It validates each set with (*coefficient.Set).Validate — the field layer, the same
// choice validate-catalog makes with graph.Validate. That is the whole of coefficient
// validation today: contracts.Validate would add only version-scoped rules, which run
// solely against a Scenario and cannot read a coefficient (rules.Input carries none). The
// canary test TestCoefficientOnlyBundleRunsNoRules fails if that ever changes — the signal
// to route this command through contracts.Validate(Bundle{Coefficients: ...}) so it
// enforces coefficient-level rules.
//
// Unlike validate-catalog it does not check that a file's name matches the set's name. By
// convention the two are equal in every committed set, and a scenario refers to a set by its
// file name (ValidateAgainstCatalog resolves coefficients/<name>.yaml), but no gate enforces
// the convention today.
package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/validate/registry"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// run validates every coefficient set discovered under the registry root given as the
// sole argument (registry.SetPaths — recursive over coefficients/, .yaml and .yml),
// writing one summary line per set to stdout — the coefficient count when it validates,
// FAILED when it does not — and every finding to stderr. It returns a process exit code:
// 0 when all sets load and validate, 1 when one or more fail, and 2 on misuse (wrong
// arguments, or a root with no coefficient sets). Splitting this out of main keeps the
// walk testable without a subprocess.
func run(args []string, stdout, stderr io.Writer) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "usage: validate-registry <blis-registry root>")
		return 2
	}
	root := args[0]
	// Distinguish a mistyped root from a real registry with nothing to validate, as
	// validate-catalog does: without this, a nonexistent root and an empty coefficients/
	// both read as "no sets found", so a typo looks like an empty directory.
	if info, err := os.Stat(root); err != nil || !info.IsDir() {
		fmt.Fprintf(stderr, "%s: registry root does not exist or is not a directory\n", root)
		return 2
	}
	coeffDir := filepath.Join(root, "coefficients")
	paths, err := registry.SetPaths(root)
	if err != nil {
		fmt.Fprintf(stderr, "scanning %s: %v\n", coeffDir, err)
		return 2
	}
	if len(paths) == 0 {
		fmt.Fprintf(stderr, "%s: no coefficient sets (*.yaml/*.yml) found\n", coeffDir)
		return 2
	}

	accPaths, err := registry.AcceptancePaths(root)
	if err != nil {
		fmt.Fprintf(stderr, "scanning %s: %v\n", filepath.Join(root, "acceptance"), err)
		return 2
	}

	routePaths, err := registry.RoutingPaths(root)
	if err != nil {
		fmt.Fprintf(stderr, "scanning %s: %v\n", filepath.Join(root, "routing"), err)
		return 2
	}

	failed := 0
	for _, path := range paths {
		failed += check(stdout, stderr, coeffDir, path, func() (int, string, *validate.Problems, error) {
			s, err := contracts.LoadCoefficientSet(path)
			if err != nil {
				return 0, "", nil, err
			}
			return len(s.Coefficients), "coefficient(s)", s.Validate(), nil
		})
	}
	// Acceptance-profile sets are optional: a registry that hosts none is complete. They are
	// named relative to the root (acceptance/<file>) so they cannot be mistaken for a
	// coefficient set of the same file name.
	for _, path := range accPaths {
		failed += check(stdout, stderr, root, path, func() (int, string, *validate.Problems, error) {
			s, err := contracts.LoadAcceptanceProfileSet(path)
			if err != nil {
				return 0, "", nil, err
			}
			return len(s.Profiles), "profile(s)", s.Validate(), nil
		})
	}
	// Routing-profile sets are optional too, and named relative to the root (routing/<file>).
	for _, path := range routePaths {
		failed += check(stdout, stderr, root, path, func() (int, string, *validate.Problems, error) {
			s, err := contracts.LoadRoutingProfileSet(path)
			if err != nil {
				return 0, "", nil, err
			}
			return len(s.Profiles), "profile(s)", s.Validate(), nil
		})
	}
	total := len(paths) + len(accPaths) + len(routePaths)
	if failed > 0 {
		fmt.Fprintf(stderr, "\n%d of %d set(s) failed\n", failed, total)
		return 1
	}
	fmt.Fprintf(stdout, "\nall %d set(s) validate\n", total)
	return 0
}

// check loads and validates one set, writing its summary line to stdout and its findings
// to stderr, and returns 1 if it failed. The set is named by its path relative to base,
// so sets with one base name in different subdirectories stay distinct in the output.
func check(stdout, stderr io.Writer, base, path string,
	load func() (int, string, *validate.Problems, error)) int {
	name := path
	if rel, e := filepath.Rel(base, path); e == nil {
		name = rel
	}
	n, noun, p, err := load()
	if err != nil {
		// A strict-decoder error can span several lines; prefix each with the file so
		// every line is attributable, not just the first.
		fmt.Fprint(stderr, prefixLines(name, err.Error()))
		fmt.Fprintf(stdout, "%-46s FAILED (load error)\n", name)
		return 1
	}
	for _, problem := range p.All() {
		fmt.Fprintf(stderr, "%s: %s\n", name, problem)
	}
	if !p.OK() {
		fmt.Fprintf(stdout, "%-46s FAILED (%d problem(s))\n", name, len(p.Errors()))
		return 1
	}
	fmt.Fprintf(stdout, "%-46s %3d %s\n", name, n, noun)
	return 0
}

// prefixLines renders a possibly multi-line diagnostic with the file name on every
// line. A YAML decoder error can carry embedded newlines; prefixing only the first line
// would leave the rest unattributed in a log that interleaves many files. CRLF is
// normalized so a carriage return never trails a prefixed line.
func prefixLines(name, msg string) string {
	var b strings.Builder
	msg = strings.ReplaceAll(msg, "\r\n", "\n")
	for _, line := range strings.Split(strings.TrimRight(msg, "\n"), "\n") {
		fmt.Fprintf(&b, "%s: %s\n", name, strings.TrimRight(line, "\r"))
	}
	return b.String()
}

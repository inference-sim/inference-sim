// Command validate-catalog loads every committed artifact in a blis-catalog checkout and
// validates it against the schema.
//
// The catalog's own gate checks that each entry is structurally a catalog entry. This one
// checks what a cost model needs: that a derived graph is a graph it can price (its nodes
// carry their ops' parameters, its DAG is acyclic, its stack names kinds it declares),
// that a chip, a fabric and a storage tier carry sane datasheet figures, that a workload
// is a well-formed traffic shape, and that each model.yaml names itself what its directory
// is called and records where its vendor config came from. A deriver or an editing slip
// produces a well-formed YAML file that fails here.
//
// One combined report, grouped by artifact kind: a per-entry summary line to stdout for
// each artifact that validates, every problem to stderr under the offending name, and a
// nonzero exit if anything failed — exit 2 for a usage error or a path that is no catalog,
// exit 1 for any validation failure, so CI can read the code and a reviewer the lines.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/validate/dataset"
	"github.com/inference-sim/blis-substrate/validate/measurements"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// namespaces are the catalog subdirectories the validator walks. Their presence is also
// what tells a catalog root from a mistyped path: a directory with none of them is
// reported as "not a catalog" rather than silently passing with zero artifacts checked.
var namespaces = []string{"models", "hardware", "networks", "workloads", "devices", "measurements"}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "usage: validate-catalog <blis-catalog root>")
		return 2
	}
	root := args[0]
	if info, err := os.Stat(root); err != nil || !info.IsDir() {
		fmt.Fprintf(stderr, "%s: catalog root does not exist or is not a directory\n", root)
		return 2
	}
	anyNamespace := false
	for _, ns := range namespaces {
		if isDir(filepath.Join(root, ns)) {
			anyNamespace = true
			break
		}
	}
	if !anyNamespace {
		fmt.Fprintf(stderr, "%s: not a catalog root — none of %v found\n", root, namespaces)
		return 2
	}

	r := &reporter{stdout: stdout, stderr: stderr}
	// Enumerate the model entries once: a read error on models/ is one failure, not one
	// per model walk, and both walks see the same sorted set.
	if dirs, err := subdirs(filepath.Join(root, "models")); err != nil {
		r.loadFail("models", "%v", err)
	} else {
		validateModelGraphs(r, dirs)
		ids := validateModelIdentities(r, dirs)
		validateModelConfigs(r, dirs, ids)
		validateLicences(r, root, dirs, ids)
	}
	chips := validateHardware(r, root)
	validateNetworks(r, root)
	validateDevices(r, root)
	validateWorkloads(r, root)
	validateMeasurements(r, root, chips)

	// A directory can satisfy the namespace gate with an empty or optional namespace (an
	// empty devices/ or workloads/) and reach here having validated nothing. Reporting
	// "all 0 validate" and exiting 0 would pass a non-catalog, so a run that found no
	// artifact at all is a structural error, not a success — the same stance the
	// graph-only predecessor took when it found no graph.
	if r.total == 0 {
		fmt.Fprintf(stderr, "%s: no catalog artifacts found to validate\n", root)
		return 2
	}
	if r.failed > 0 {
		fmt.Fprintf(stderr, "\n%d of %d artifact(s) failed\n", r.failed, r.total)
		return 1
	}
	fmt.Fprintf(stdout, "\nall %d artifact(s) validate\n", r.total)
	return 0
}

// reporter accumulates one combined report across every artifact kind: a running count,
// per-entry summary lines on stdout, and every problem on stderr.
type reporter struct {
	stdout, stderr io.Writer
	total, failed  int
}

// section prints a heading to stdout before the entries of one artifact kind, so the
// combined report stays readable.
func (r *reporter) section(title string) { fmt.Fprintln(r.stdout, title) }

// pass records an artifact that validated clean and prints its summary line.
func (r *reporter) pass(summary string) {
	r.total++
	fmt.Fprintln(r.stdout, summary)
}

// record reports an artifact's problems under name and returns whether it failed. A
// clean document (no error-severity problems) returns false so the caller prints its
// summary; warnings are printed but do not fail it.
func (r *reporter) record(name string, p *validate.Problems) bool {
	for _, pr := range p.All() {
		fmt.Fprintf(r.stderr, "%s: %s\n", name, pr)
	}
	if !p.OK() {
		r.total++
		r.failed++
		return true
	}
	return false
}

// loadFail records an artifact that could not be loaded, or an entry that is missing.
func (r *reporter) loadFail(name, format string, args ...any) {
	r.total++
	r.failed++
	fmt.Fprintf(r.stderr, "%s: "+format+"\n", append([]any{name}, args...)...)
}

func validateModelGraphs(r *reporter, dirs []string) {
	// A model entry need not carry a derived graph.yaml, so collect the ones that do
	// (reporting any stat error that is not plain absence) before printing the section —
	// an entry without a graph is skipped, not an empty heading.
	var paths []string
	for _, dir := range dirs {
		path := filepath.Join(dir, "graph.yaml")
		switch info, err := os.Stat(path); {
		case errors.Is(err, fs.ErrNotExist):
			continue
		case err != nil:
			r.loadFail(filepath.Base(dir), "%v", err)
		case info.IsDir():
			r.loadFail(filepath.Base(dir), "graph.yaml is a directory, not a file")
		default:
			paths = append(paths, path)
		}
	}
	if len(paths) == 0 {
		return
	}
	r.section("# models/*/graph.yaml")
	for _, path := range paths {
		name := filepath.Base(filepath.Dir(path))
		g, p, err := checkFile(path, "models/"+name+"/graph.yaml", dataset.CheckModelGraph)
		if err != nil {
			r.loadFail(name, "%v", err)
			continue
		}
		if r.record(name, p) {
			continue
		}
		r.pass(fmt.Sprintf("%-46s %3d layers, %d kind(s), %s", name,
			g.Stack.Layers(), len(g.LayerKinds), g.Global.WeightDType))
	}
}

func validateModelConfigs(r *reporter, dirs []string, ids map[string]*model.Identity) {
	if len(dirs) == 0 {
		return
	}
	r.section("# models/*/config.json")
	for _, dir := range dirs {
		name := filepath.Base(dir)
		label := name + "/config.json"
		path := filepath.Join(dir, "config.json")
		// A model whose licence is not on the allow-list commits only the file's hash and
		// revision (X12): model.yaml's config.sha256 stands in for the file, which the
		// nightly fetch job retrieves with the access token.
		if id := ids[name]; id != nil && id.Config != nil {
			if _, err := os.Stat(path); err == nil {
				r.loadFail(label, "committed although model.yaml pins it by hash (config.sha256); commit one or the other (X12)")
			} else {
				r.pass(fmt.Sprintf("%-46s not committed; pinned by sha256 %s", label, id.Config.SHA256[:12]))
			}
			continue
		}
		// config.json is the verbatim vendor file every model entry pairs with its
		// model.yaml, and it is REQUIRED: this is the structural half of blis-catalog's
		// validate_models — present, parseable, a non-empty JSON object — reproduced so the
		// binary can stand in for that gate without a schema type reading the config's keys.
		switch info, err := os.Stat(path); {
		case errors.Is(err, fs.ErrNotExist):
			r.loadFail(label, "missing (a model entry needs config.json)")
			continue
		case err != nil:
			r.loadFail(label, "%v", err)
			continue
		case info.IsDir():
			r.loadFail(label, "config.json is a directory, not a file")
			continue
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			r.loadFail(label, "%v", err)
			continue
		}
		// Decode into any first so valid JSON of any shape parses; then require a non-empty
		// object. A non-object (array, scalar) and an empty object are both rejected with the
		// one message, as validate_models does; invalid JSON is reported as a parse error.
		var v any
		if err := json.Unmarshal(raw, &v); err != nil {
			r.loadFail(label, "invalid JSON: %v", err)
			continue
		}
		obj, ok := v.(map[string]any)
		if !ok || len(obj) == 0 {
			r.loadFail(label, "must be a non-empty JSON object")
			continue
		}
		r.pass(fmt.Sprintf("%-46s %d key(s)", label, len(obj)))
	}
}

func validateModelIdentities(r *reporter, dirs []string) map[string]*model.Identity {
	ids := map[string]*model.Identity{}
	if len(dirs) == 0 {
		return ids
	}
	r.section("# models/*/model.yaml")
	for _, dir := range dirs {
		name := filepath.Base(dir)
		label := name + "/model.yaml"
		path := filepath.Join(dir, "model.yaml")
		// A missing model.yaml is an incomplete entry; a permission or I/O error is a
		// different failure, and collapsing the two would report "missing" for a file that
		// is in fact present but unreadable, misdirecting the fix.
		switch info, err := os.Stat(path); {
		case errors.Is(err, fs.ErrNotExist):
			r.loadFail(label, "missing: a model entry needs a model.yaml identity manifest")
			continue
		case err != nil:
			r.loadFail(label, "%v", err)
			continue
		case info.IsDir():
			r.loadFail(label, "model.yaml is a directory, not a file")
			continue
		}
		id, p, err := checkFile(path, "models/"+name+"/model.yaml", dataset.CheckModelIdentityLicensed)
		if err != nil {
			r.loadFail(label, "%v", err)
			continue
		}
		ids[name] = id
		if r.record(label, p) {
			continue
		}
		r.pass(fmt.Sprintf("%-46s %s", label, summarizeSource(id.Source)))
	}
	return ids
}

// validateHardware checks hardware/ and returns the entries that loaded and validated, by
// name, for the checks that resolve a cluster's packaging against them.
func validateHardware(r *reporter, root string) map[string]*hardware.Chip {
	chips := map[string]*hardware.Chip{}
	paths, err := yamlFiles(filepath.Join(root, "hardware"))
	if err != nil {
		r.loadFail("hardware", "%v", err)
		return chips
	}
	if len(paths) == 0 {
		return chips
	}
	r.section("# hardware/*.yaml")
	for _, path := range paths {
		name := stem(path)
		c, p, err := checkFile(path, path, dataset.CheckChip)
		if err != nil {
			r.loadFail(name, "%v", err)
			continue
		}
		if r.record(name, p) {
			continue
		}
		chips[c.Name] = c
		r.pass(fmt.Sprintf("%-46s bf16 %g, fp8 %g TFLOP/s, %g GiB",
			c.Name, c.BF16Peak, c.FP8Peak, c.MemoryGiB))
	}
	return chips
}

func validateNetworks(r *reporter, root string) {
	paths, err := yamlFiles(filepath.Join(root, "networks"))
	if err != nil {
		r.loadFail("networks", "%v", err)
		return
	}
	if len(paths) == 0 {
		return
	}
	r.section("# networks/*.yaml")
	for _, path := range paths {
		name := stem(path)
		f, p, err := checkFile(path, path, dataset.CheckFabric)
		if err != nil {
			r.loadFail(name, "%v", err)
			continue
		}
		if r.record(name, p) {
			continue
		}
		r.pass(fmt.Sprintf("%-46s %g GB/s inter-node", f.Name, f.InterNodeBwGBps))
	}
}

func validateDevices(r *reporter, root string) {
	path := filepath.Join(root, "devices", "storage.yaml")
	// devices/storage.yaml is optional: the catalog's own gate skips a missing devices
	// namespace too, so an absent file is not a failure. Any other stat error (a
	// permission or I/O problem, or a directory in its place) is reported rather than
	// silently swallowed.
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return
	}
	r.section("# devices/storage.yaml")
	switch {
	case err != nil:
		r.loadFail("devices/storage.yaml", "%v", err)
		return
	case info.IsDir():
		r.loadFail("devices/storage.yaml", "storage.yaml is a directory, not a file")
		return
	}
	devices, err := contracts.LoadStorageDevices(path)
	if err != nil {
		r.loadFail("devices/storage.yaml", "%v", err)
		return
	}
	for _, d := range devices {
		if r.record("devices/storage.yaml:"+d.Name, d.Validate()) {
			continue
		}
		r.pass(fmt.Sprintf("%-46s r %g, w %g MB/s, %g us base latency",
			d.Name, d.ReadBandwidthMBs, d.WriteBandwidthMBs, d.BaseLatencyUs))
	}
}

func validateWorkloads(r *reporter, root string) {
	paths, err := yamlFiles(filepath.Join(root, "workloads"))
	if err != nil {
		r.loadFail("workloads", "%v", err)
		return
	}
	if len(paths) == 0 {
		return
	}
	r.section("# workloads/*.yaml")
	for _, path := range paths {
		w, p, err := checkFile(path, path, dataset.CheckWorkload)
		if err != nil {
			// The identity is the filename stem; the loader could not stamp it, so name
			// the file.
			r.loadFail(stem(path), "%v", err)
			continue
		}
		if r.record(w.Name(), p) {
			continue
		}
		if rp := w.Replay; rp != nil {
			r.pass(fmt.Sprintf("%-46s replay %s over %s@%s", rp.Name, rp.Scenario,
				rp.Corpus.Source.HuggingFace, rp.Corpus.Source.Revision))
			continue
		}
		r.pass(fmt.Sprintf("%-46s prompt~%d, output~%d tokens",
			w.Shape.Name, w.Shape.Prompt.Mean, w.Shape.Output.Mean))
	}
}

// validateMeasurements checks measurements/: the licence registry and every run against
// it. The namespace is optional, but once present it needs its sources.yaml, and every run
// in it must be one the licence lets the catalog redistribute.
func validateMeasurements(r *reporter, root string, chips map[string]*hardware.Chip) {
	dir := filepath.Join(root, "measurements")
	ok, err := measurements.Exists(dir)
	if err != nil {
		r.loadFail("measurements", "%v", err)
		return
	}
	if !ok {
		return
	}
	results, err := measurements.Check(dir, true, chips)
	if err != nil {
		r.loadFail("measurements", "%v", err)
		return
	}
	r.section("# measurements/")
	for _, res := range results {
		name := "measurements/" + res.Name
		if res.Err != nil {
			r.loadFail(name, "%v", res.Err)
			continue
		}
		if r.record(name, res.Problems) {
			continue
		}
		r.pass(fmt.Sprintf("%-46s %s", name, res.Summary))
	}
}

// checkFile reads path and runs on it the check package load runs on the same file
// (validate/dataset), so this command and a run judge a file alike (SUB-9, X15). label is
// the name the check reports the file by.
func checkFile[T any](path, label string, check func(string, []byte) (T, *validate.Problems, error)) (T, *validate.Problems, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		var zero T
		return zero, nil, fmt.Errorf("opening %s: %w", path, err)
	}
	return check(label, b)
}

// summarizeSource renders a model card's provenance for the one-line summary: provider,
// repo and an abbreviated revision, which is enough to read at a glance without widening
// the column to a full commit hash.
func summarizeSource(s *model.Source) string {
	if s == nil {
		return "(no source)"
	}
	rev := s.Revision
	if len(rev) > 12 {
		rev = rev[:12]
	}
	return fmt.Sprintf("%s %s@%s", s.Provider, s.Repo, rev)
}

// stem returns a path's base name without its extension — how the catalog names a chip, a
// fabric or a workload.
func stem(path string) string {
	base := filepath.Base(path)
	return base[:len(base)-len(filepath.Ext(base))]
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

// yamlFiles returns the sorted *.yaml files directly under dir. A missing directory yields
// no files and no error — the namespace is simply absent — but any other read error IS
// returned, so an unreadable directory is reported rather than silently treated as empty.
// filepath.Glob cannot do this: it surfaces only a bad pattern and swallows a directory
// read error, which would let the run exit 0 with the namespace unchecked.
func yamlFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".yaml" {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(out)
	return out, nil
}

// subdirs returns the sorted subdirectories of dir, with the same missing-vs-unreadable
// distinction as yamlFiles, so an unreadable models/ is reported rather than read as a
// catalog with no models.
func subdirs(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(out)
	return out, nil
}

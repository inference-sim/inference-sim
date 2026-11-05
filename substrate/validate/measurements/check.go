// Package measurements checks a measurements directory as a whole: the licence registry
// in sources.yaml and every run under <license_ref>/, each against the registry. It is
// shared by validate-catalog, which checks a catalog's committed measurements, and
// validate-measurements, which checks an adapter's output before it is committed.
package measurements

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
)

// SourcesFile is the licence registry's file name within a measurements directory.
const SourcesFile = "sources.yaml"

// Result is the outcome for one file.
type Result struct {
	// Name is the file's path relative to the measurements directory.
	Name string
	// Summary is a one-line description, set when the file loaded.
	Summary  string
	Problems *validate.Problems
	// Err is a load failure; Problems is nil when it is set.
	Err error
}

// OK reports whether the file loaded and validated.
func (r Result) OK() bool { return r.Err == nil && r.Problems.OK() }

// Check validates the registry and every run under dir. committed requires each run's
// licence to permit redistribution, which a file in a public repository is; an adapter
// checking local-only output passes false. It returns an error only when dir cannot be
// read at all.
func Check(dir string, committed bool) ([]Result, error) {
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("%s: not a directory", dir)
	}
	var out []Result
	reg, err := contracts.LoadMeasurementSources(filepath.Join(dir, SourcesFile))
	src := Result{Name: SourcesFile}
	if err != nil {
		src.Err = err
	} else {
		src.Problems = reg.Validate()
		src.Summary = summarizeRegistry(reg)
	}
	out = append(out, src)

	paths, stray, err := runFiles(dir)
	if err != nil {
		return nil, err
	}
	for _, s := range stray {
		out = append(out, Result{Name: s, Err: fmt.Errorf(
			"a run lives at <license_ref>/<name>.yaml; only %s sits at the top level", SourcesFile)})
	}
	type loaded struct {
		problems *validate.Problems
		run      *measurement.MeasuredRun
	}
	var runs []loaded
	names := map[string]string{}
	for _, path := range paths {
		rel, _ := filepath.Rel(dir, path)
		rel = filepath.ToSlash(rel)
		res := Result{Name: rel}
		r, err := contracts.LoadMeasuredRun(path)
		if err != nil {
			res.Err = err
			out = append(out, res)
			continue
		}
		p := r.ValidateWithSources(reg)
		if stem := strings.TrimSuffix(filepath.Base(path), ".yaml"); r.Name != stem {
			p.Field("name", "the run names itself %q but its file is %s.yaml", r.Name, stem)
		}
		if parent := filepath.Base(filepath.Dir(path)); r.Source.LicenseRef != parent {
			p.Field("source.license_ref", "%q, but the run is filed under %s/; a run lives under its licence's directory",
				r.Source.LicenseRef, parent)
		}
		if committed && reg != nil && !reg.Permits(r.Source.LicenseRef, measurement.UseRedistribute) {
			p.Field("source.license_ref", "licence %q does not permit redistribution, so the run cannot be committed; keep it local",
				r.Source.LicenseRef)
		}
		if prev, dup := names[r.Name]; dup {
			p.Field("name", "duplicate run name; %s has it too", prev)
		}
		names[r.Name] = rel
		a := measurement.Assess(r, reg.Permits(r.Source.LicenseRef, measurement.UseScore))
		res.Problems = p
		res.Summary = fmt.Sprintf("%s, %d point(s)", verdictText(a), len(r.Points))
		out = append(out, res)
		runs = append(runs, loaded{p, r})
	}
	for _, l := range runs {
		for field, ref := range map[string]string{"duplicate_of": l.run.DuplicateOf, "supersedes": l.run.Supersedes} {
			if _, ok := names[ref]; ref != "" && !ok {
				l.problems.Field(field, "names %q, which is no run in this directory", ref)
			}
		}
	}
	return out, nil
}

// runFiles returns every <dir>/<sub>/*.yaml, sorted, and every top-level YAML file other
// than the registry.
func runFiles(dir string) (runs, stray []string, err error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	for _, e := range entries {
		switch {
		case e.IsDir():
			sub, err := os.ReadDir(filepath.Join(dir, e.Name()))
			if err != nil {
				return nil, nil, err
			}
			for _, f := range sub {
				if !f.IsDir() && filepath.Ext(f.Name()) == ".yaml" {
					runs = append(runs, filepath.Join(dir, e.Name(), f.Name()))
				}
			}
		case filepath.Ext(e.Name()) == ".yaml" && e.Name() != SourcesFile:
			stray = append(stray, e.Name())
		}
	}
	sort.Strings(runs)
	sort.Strings(stray)
	return runs, stray, nil
}

func summarizeRegistry(r *measurement.Registry) string {
	n := map[measurement.LicenseStatus]int{}
	for _, l := range r.Sources {
		n[l.Status]++
	}
	return fmt.Sprintf("%d source(s): %d verified, %d unverified, %d refused", len(r.Sources),
		n[measurement.LicenseVerified], n[measurement.LicenseUnverified], n[measurement.LicenseRefused])
}

func verdictText(a measurement.Assessment) string {
	if a.Resolution != "" {
		return string(a.Verdict) + "/" + string(a.Resolution)
	}
	return string(a.Verdict)
}

// Exists reports whether dir is present, distinguishing absence from an unreadable path.
func Exists(dir string) (bool, error) {
	info, err := os.Stat(dir)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return false, nil
	case err != nil:
		return false, err
	case !info.IsDir():
		return false, fmt.Errorf("%s is not a directory", dir)
	}
	return true, nil
}

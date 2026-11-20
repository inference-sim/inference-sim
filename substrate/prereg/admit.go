package prereg

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

// Admit is what a fit calls before it reads a unit of source at role. It refuses the read unless
// the pre-registration id exists, lists the source, allows the role at the step, and was
// registered against the partition, defaults and source revision the ledger holds now. It reads
// no unit; the caller reads only after Admit returns nil.
func (r *Registry) Admit(ledgerDir, id string, step Step, source string, role ledger.Role) error {
	rec := r.Record(id)
	if rec == nil {
		return fmt.Errorf("no pre-registration %q: a refit reads nothing before its pre-registration is merged", id)
	}
	if contains(r.Defaults.ForbiddenSources, source) {
		return fmt.Errorf("%s: source %s is anchor-only and is never fit, selected or validated", id, source)
	}
	if !contains(rec.Sources, source) {
		return fmt.Errorf("%s: source %s is not covered by this pre-registration (covers %v)", id, source, rec.Sources)
	}
	var allowed []ledger.Role
	switch step {
	case StepFit:
		allowed = r.Defaults.FitReads
	case StepSelect:
		allowed = r.Defaults.SelectReads
	default:
		return fmt.Errorf("%s: step %q, want %s or %s", id, step, StepFit, StepSelect)
	}
	if !contains(allowed, role) {
		return fmt.Errorf("%s: the %s step reads %v, not %s", id, step, allowed, role)
	}
	cfg, err := ledger.LoadConfig(ledgerDir)
	if err != nil {
		return err
	}
	srcs, err := ledger.LoadSources(ledgerDir)
	if err != nil {
		return err
	}
	if errs := r.verifyPins(rec, cfg, srcs, &r.Defaults); len(errs) != 0 {
		return fmt.Errorf("%s is stale, so no source is read: %w", id, errors.Join(errs...))
	}
	return nil
}

// A fit script that reads a gated source states which record covers it with a line
// "# prereg: <id>" (or "// prereg: <id>"). It may not name a forbidden source at all.
var markerRe = regexp.MustCompile(`(?m)^\s*(?:#|//)\s*prereg:\s*([a-z0-9][a-z0-9.-]*)\s*$`)

var scanned = map[string]bool{".py": true, ".sh": true, ".go": true}

// ScanScripts finds the fit scripts under the defaults' fit_script_dirs (relative to root) that
// read a source no record covers: a script naming a gated source needs a "prereg:" line for a
// record that lists every gated source it names, and a script naming a forbidden source is
// refused. Test files are skipped; they exercise code, they do not fit.
func (r *Registry) ScanScripts(root string) error {
	var errs []error
	for _, dir := range r.Defaults.FitScriptDirs {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if d.Name() == "testdata" || d.Name() == "__pycache__" {
					return filepath.SkipDir
				}
				return nil
			}
			base := d.Name()
			if !scanned[filepath.Ext(base)] || strings.HasSuffix(base, "_test.go") || strings.HasPrefix(base, "test_") {
				return nil
			}
			raw, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(root, path)
			errs = append(errs, r.scanOne(filepath.ToSlash(rel), string(raw))...)
			return nil
		})
		if err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func mentions(text, source string) bool {
	return regexp.MustCompile(`(^|[^A-Za-z0-9_-])` + regexp.QuoteMeta(source) + `($|[^A-Za-z0-9_-])`).MatchString(text)
}

func (r *Registry) scanOne(rel, text string) []error {
	var errs []error
	var gated []string
	for _, s := range r.Defaults.ForbiddenSources {
		if mentions(text, s) {
			errs = append(errs, fmt.Errorf("%s: names %s, which is anchor-only and never fit, selected or validated", rel, s))
		}
	}
	for _, s := range r.Defaults.GatedSources {
		if mentions(text, s) {
			gated = append(gated, s)
		}
	}
	if len(gated) == 0 {
		return errs
	}
	sort.Strings(gated)
	markers := markerRe.FindAllStringSubmatch(text, -1)
	if len(markers) == 0 {
		return append(errs, fmt.Errorf("%s: reads %v and carries no \"# prereg: <id>\" line: a refit reads no new source before its pre-registration is merged", rel, gated))
	}
	var recs []*Record
	for _, m := range markers {
		if rec := r.Record(m[1]); rec != nil {
			recs = append(recs, rec)
		} else {
			errs = append(errs, fmt.Errorf("%s: names pre-registration %q, which does not exist", rel, m[1]))
		}
	}
	for _, s := range gated {
		ok := false
		for _, rec := range recs {
			ok = ok || contains(rec.Sources, s)
		}
		if !ok {
			errs = append(errs, fmt.Errorf("%s: reads %s, which none of the pre-registrations it names covers", rel, s))
		}
	}
	return errs
}

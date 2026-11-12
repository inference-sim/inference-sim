package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/load"
)

// generations is the architecture generation of each catalog chip: the die family, which is
// what the registry's by-generation writer transfers a coefficient along
// (registry/scripts/extrapolate_by_generation.py). A test requires it to equal that table.
var generations = map[string]string{
	"h100": "Hopper", "h200": "Hopper",
	"b200": "Blackwell", "b300": "Blackwell", "gb200-nvl72": "Blackwell", "gb300": "Blackwell",
	"a100-sxm": "Ampere", "a100-80": "Ampere",
	"l40s": "Ada",
}

// nearCopies groups chips whose measured values are the same parts' values under two names, so
// that leaving one out leaves its twin out too. The two A100 variants share every measured
// value, and a twin predicting its twin would report no error at all.
var nearCopies = map[string]string{"a100-80": "a100", "a100-sxm": "a100"}

var (
	chipSuffix  = regexp.MustCompile(`_(a100_80|a100_sxm|b200|b300|gb200_nvl72|gb300|h100|h200|l40s)$`)
	rankInName  = regexp.MustCompile(`_\d+rank`)
	formatSufix = regexp.MustCompile(`_(fp16|int8|bf16|fp8_block|fp8|nvfp4|mxfp4|w4a16_[a-z0-9_]+|w8a16_[a-z0-9_]+)$`)
	hardwareKey = regexp.MustCompile(`hardware=\S*\s*`)
)

// registryPoints reads every measured, positive coefficient of the named sets as one point per
// chip in its scope. The primitive family is the name without its chip, rank count and data
// type; the key keeps those, so a series is one quantity across chips.
func registryPoints(d *load.Data, sets []string) ([]ledger.Point, error) {
	var pts []ledger.Point
	for _, set := range sets {
		cs, err := d.CoefficientSet(set)
		if err != nil {
			return nil, err
		}
		for _, e := range cs.Coefficients {
			if e.Method != vocab.MethodMeasured || !(e.Value > 0) {
				continue
			}
			fam := formatSufix.ReplaceAllString(rankInName.ReplaceAllString(chipSuffix.ReplaceAllString(e.Name, ""), ""), "")
			key := hardwareKey.ReplaceAllString(set+"|"+chipSuffix.ReplaceAllString(e.Name, "")+"|"+e.Scope.Describe(), "")
			for _, h := range e.Scope.Hardware {
				pts = append(pts, ledger.Point{Family: fam, Chip: h, Key: key, Value: e.Value})
			}
		}
	}
	return pts, nil
}

// methodErrorSets are the coefficient sets the typical error is measured over: every set the
// cost model prices from.
var methodErrorSets = []string{
	"cost-model-primitives", "cost-model-collectives", "cost-model-collectives-inferencex",
	"cost-model-attention", "cost-model-recurrent", "cost-model-memory",
	"cost-model-host-overheads", "cost-model-fabric",
}

func cmdMethodError(args []string, stdout io.Writer) error {
	fs := flags("method-error")
	registry := fs.String("registry", "", "a coefficient registry directory; default the embedded one")
	out := fs.String("out", "", "the report to write (JSON)")
	table := fs.String("table", "", "a Markdown page of the figures to write")
	check := fs.String("check", "", "a committed page that must equal the page the registry produces")
	candidate := fs.String("candidate", "", "the commit the registry is at")
	if err := parse(fs, args, 0, "method-error [-registry DIR] [-candidate SHA -out FILE] [-table FILE] [-check FILE]"); err != nil {
		return err
	}
	if (*out == "") != (*candidate == "") || (*out == "" && *table == "" && *check == "") {
		return usageError{"usage: ledger method-error [-registry DIR] [-candidate SHA -out FILE] [-table FILE] [-check FILE]"}
	}
	d, err := load.Open(load.Options{Registry: *registry})
	if err != nil {
		return err
	}
	var sets []string
	for _, s := range methodErrorSets {
		if _, err := d.CoefficientSet(s); err == nil {
			sets = append(sets, s)
		}
	}
	pts, err := registryPoints(d, sets)
	if err != nil {
		return err
	}
	if len(pts) == 0 {
		return fmt.Errorf("the registry carries no measured coefficient in %v; the job processes at least one point (X22)", methodErrorSets)
	}
	rows, err := ledger.MethodError(pts, []ledger.Estimator{
		ledger.SameGeneration("same_generation", generations),
		ledger.MedianOfOthers("global_median"),
	}, nearCopies)
	if err != nil {
		return err
	}
	page := methodErrorPage(rows, len(pts))
	if *check != "" {
		have, err := os.ReadFile(*check)
		if err != nil {
			return err
		}
		if string(have) != page {
			return fmt.Errorf("%s is not the page the registry produces; regenerate it with ledger method-error -table %s", *check, *check)
		}
	}
	n := 0
	if *out != "" {
		rep := ledger.MethodErrorReport(*candidate, rows)
		raw, err := json.MarshalIndent(rep, "", " ")
		if err != nil {
			return err
		}
		if err := ledger.CheckReport(raw, nil); err != nil {
			return err
		}
		if err := os.WriteFile(*out, append(raw, '\n'), 0o644); err != nil {
			return err
		}
		n = len(rep.Aggregates)
	}
	if *table != "" {
		if err := os.WriteFile(*table, []byte(page), 0o644); err != nil {
			return err
		}
	}
	_, _ = fmt.Fprintf(stdout, "%d points, %d figures, each over at least %d points\n", len(pts), n, ledger.MinAggregateUnits)
	return nil
}

// methodErrorPage renders the rows as the registry's method-error reference page, methods side
// by side per family. It is generated and committed; a test requires the committed page to
// equal it, so a figure here cannot disagree with the registry.
func methodErrorPage(rows []ledger.MethodErrorRow, points int) string {
	fams := map[string]map[string]ledger.MethodErrorRow{}
	var names []string
	for _, r := range rows {
		if fams[r.Family] == nil {
			fams[r.Family] = map[string]ledger.MethodErrorRow{}
			names = append(names, r.Family)
		}
		fams[r.Family][r.Method] = r
	}
	sort.Strings(names)
	var b strings.Builder
	b.WriteString("<!-- Generated by go run ./ledger/cmd/ledger method-error -table. Do not edit. -->\n")
	b.WriteString("# Typical error of each method\n\n")
	b.WriteString("Design 9.3 and VAL-5: each method's typical error is measured by leaving one chip out. For every chip that has\n")
	b.WriteString("a measured value, the method estimates it as if the measurement did not exist, and the estimate is compared\n")
	b.WriteString("with the measurement. The two parts of the A100 family carry the same measured values, so they are left out\n")
	b.WriteString("together; otherwise each would predict itself from its twin.\n\n")
	fmt.Fprintf(&b, "The figures below cover %d measured points of the committed coefficient registry. A ratio is\n", points)
	b.WriteString("`max(estimate/truth, truth/estimate)`: 1.00 is exact and 2.00 is a factor of two either way.\n\n")
	b.WriteString("- **Same generation** is the median of the same quantity on the other chips of the held-out chip's architecture\n")
	b.WriteString("  generation, and gives no estimate for a chip with no sibling. It is the predictor `extrapolate_by_generation.py` writes.\n")
	b.WriteString("- **Global median** is the median over every other chip. It is the baseline a method must beat.\n\n")
	b.WriteString("A family with fewer than 10 held-out points is merged into `other`, and `other` is left out if it too has fewer.\n")
	b.WriteString("What this does not yet measure: the roofline (rank 6) and vendor-specification (rank 5) methods. Each estimates\n")
	b.WriteString("a value from a datasheet figure, so measuring its error needs an estimator that reads the chip's datasheet; the\n")
	b.WriteString("two above read only other chips' measurements. Until such an estimator is written, the page states no error for\n")
	b.WriteString("those methods and says so rather than reusing the figures above for them. The whole-run error is the dashboard's.\n\n")
	b.WriteString("| Family | Points | Same generation, median | Same generation, worst | Global median, median | Global median, worst |\n|---|---|---|---|---|---|\n")
	cell := func(r ledger.MethodErrorRow, ok bool, f func(ledger.MethodErrorRow) float64) string {
		if !ok {
			return "no estimate"
		}
		return fmt.Sprintf("%.2f", f(r))
	}
	for _, n := range names {
		g, gok := fams[n]["same_generation"]
		m, mok := fams[n]["global_median"]
		pts := g.Points
		if !gok {
			pts = m.Points
		}
		fmt.Fprintf(&b, "| `%s` | %d | %s | %s | %s | %s |\n", n, pts,
			cell(g, gok, func(r ledger.MethodErrorRow) float64 { return r.MedianRatio }),
			cell(g, gok, func(r ledger.MethodErrorRow) float64 { return r.WorstRatio }),
			cell(m, mok, func(r ledger.MethodErrorRow) float64 { return r.MedianRatio }),
			cell(m, mok, func(r ledger.MethodErrorRow) float64 { return r.WorstRatio }))
	}
	return b.String()
}

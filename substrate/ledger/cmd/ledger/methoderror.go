package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
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
	"l40s":              "Ada",
	"rtx6000pro-server": "RTXBlackwell",
}

// nearCopies groups chips whose measured values are the same parts' values under two names, so
// that leaving one out leaves its twin out too. The two A100 variants share every measured
// value, and a twin predicting its twin would report no error at all.
var nearCopies = map[string]string{"a100-80": "a100", "a100-sxm": "a100"}

var (
	chipSuffix  = regexp.MustCompile(`_(a100_80|a100_sxm|b200|b300|gb200_nvl72|gb300|h100|h200|l40s|rtx6000pro_server)$`)
	rankInName  = regexp.MustCompile(`_\d+rank`)
	formatSufix = regexp.MustCompile(`_(fp16|int8|bf16|fp8_block|fp8|nvfp4|mxfp4|w4a16_[a-z0-9_]+|w8a16_[a-z0-9_]+)$`)
	// The hardware term of a scope key (coefficient.Scope.Key: "label=v1,v2;" terms), and only it:
	// the model and the other terms stay in the series key.
	hardwareKey = regexp.MustCompile(`hardware=[^;]*;?`)
)

// seriesKey is the series one coefficient's points belong to: its set, its name without the chip,
// and its scope without the hardware, so the same quantity on different chips is one series and
// the same name scoped to different models is two.
func seriesKey(set string, e coefficient.Entry) string {
	return set + "|" + chipSuffix.ReplaceAllString(e.Name, "") + "|" + hardwareKey.ReplaceAllString(e.Scope.Describe(), "")
}

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
			key := seriesKey(set, e)
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
	candidate := fs.String("candidate", "", "the commit the registry is at")
	if err := parse(fs, args, 0, "method-error [-registry DIR] -candidate SHA -out FILE"); err != nil {
		return err
	}
	if *out == "" || *candidate == "" {
		return usageError{"usage: ledger method-error [-registry DIR] -candidate SHA -out FILE"}
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
	n := 0
	{
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
	_, _ = fmt.Fprintf(stdout, "%d points, %d figures, each over at least %d points\n", len(pts), n, ledger.MinAggregateUnits)
	return nil
}

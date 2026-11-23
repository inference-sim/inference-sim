package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/load"
)

const (
	axisMemoryBandwidth    = "memory_bandwidth"
	axisIntraNodeBandwidth = "intra_node_bandwidth"
)

var (
	memoryBoundFamily = regexp.MustCompile(`^attention_decode_rate(_mla|_swa)?$`)
	linkBoundFamily   = regexp.MustCompile(`^collective_(peak|transition)_rate_`)
)

// transferAxis is the datasheet quantity that limits a primitive family, or false for a family
// no datasheet quantity limits (a floor, a count, a fitted shape parameter).
func transferAxis(family string) (string, bool) {
	switch {
	case memoryBoundFamily.MatchString(family):
		return axisMemoryBandwidth, true
	case linkBoundFamily.MatchString(family):
		return axisIntraNodeBandwidth, true
	}
	return "", false
}

// datasheet reads each axis's figure for every catalog chip.
func datasheet(d *load.Data) (map[string]map[string]float64, error) {
	out := map[string]map[string]float64{axisMemoryBandwidth: {}, axisIntraNodeBandwidth: {}}
	for _, name := range d.Chips() {
		c, err := d.Chip(name)
		if err != nil {
			return nil, err
		}
		out[axisMemoryBandwidth][name] = c.MemoryBandwidthTBs
		out[axisIntraNodeBandwidth][name] = c.IntraNodeBwGBps
	}
	return out, nil
}

type transferValidation struct {
	Kind        string               `json:"kind"`
	Description string               `json:"description"`
	MinPoints   int                  `json:"min_points"`
	Rows        []ledger.TransferRow `json:"rows"`
}

const transferDescription = "VAL-5, leaving one generation out. The rule carries a measured value from one generation to another by keeping its efficiency (value over the datasheet figure of the class's axis) and multiplying by the target's figure, reading the same series on the source generation. Roofline is the target's figure times the family's median efficiency on every chip outside the target's generation. Both are scored on the same held-out points by max(estimate/truth, truth/estimate); use_rule is true only if the rule's median is strictly below roofline's, and otherwise a cross-generation transfer ranks as roofline."

func transferReport(d *load.Data) ([]byte, error) {
	var sets []string
	for _, s := range methodErrorSets {
		if _, err := d.CoefficientSet(s); err == nil {
			sets = append(sets, s)
		}
	}
	pts, err := registryPoints(d, sets)
	if err != nil {
		return nil, err
	}
	sheet, err := datasheet(d)
	if err != nil {
		return nil, err
	}
	rows, err := ledger.TransferError(pts, ledger.TransferSpec{Axis: transferAxis, Datasheet: sheet, Generation: generations})
	if err != nil {
		return nil, err
	}
	raw, err := json.MarshalIndent(transferValidation{
		Kind: "TransferRuleValidation", Description: transferDescription, MinPoints: ledger.MinAggregateUnits, Rows: rows,
	}, "", " ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

func cmdTransferError(args []string, stdout io.Writer) error {
	const usage = "transfer-error [-registry DIR] (-out FILE | -check FILE)"
	fs := flags("transfer-error")
	registry := fs.String("registry", "", "a coefficient registry directory; default the embedded one")
	out := fs.String("out", "", "the validation to write (JSON)")
	check := fs.String("check", "", "require this file to equal the validation computed now")
	if err := parse(fs, args, 0, usage); err != nil {
		return err
	}
	if (*out == "") == (*check == "") {
		return usageError{"usage: ledger " + usage}
	}
	d, err := load.Open(load.Options{Registry: *registry})
	if err != nil {
		return err
	}
	raw, err := transferReport(d)
	if err != nil {
		return err
	}
	if *check != "" {
		have, err := os.ReadFile(*check)
		if err != nil {
			return err
		}
		if !bytes.Equal(have, raw) {
			return fmt.Errorf("%s is stale: regenerate it with `ledger transfer-error -out %s`", *check, *check)
		}
		_, _ = fmt.Fprintf(stdout, "%s is current\n", *check)
		return nil
	}
	if err := os.WriteFile(*out, raw, 0o644); err != nil {
		return err
	}
	var v transferValidation
	if err := json.Unmarshal(raw, &v); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "%d rows, each over at least %d points\n", len(v.Rows), ledger.MinAggregateUnits)
	return nil
}

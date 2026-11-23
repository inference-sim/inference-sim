package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

// cmdPartitionPrep is M3.2 row 26's D2 join report. It reads no seed and assigns no role: it
// shows, from the consumers' records alone, which units of an anchor-only source were never
// read under a training role.
func cmdPartitionPrep(args []string, stdout io.Writer) error {
	fs := flags("partition-prep")
	source := fs.String("source", "configiq", "the anchor-only source to report on")
	require := fs.String("require", "", "comma-separated consumers whose record must be among READS")
	out := fs.String("out", "", "the report to write (job-local; never committed)")
	propose := fs.String("propose", "", "also write the proposed next ledger.yaml here, for a maintainer's pull request")
	if err := fs.Parse(args); err != nil || fs.NArg() < 2 || *out == "" {
		return usageError{"usage: ledger partition-prep [-source S] [-require A,B] -out REPORT [-propose LEDGER_YAML] LEDGER READS..."}
	}
	if !contains(ledger.AnchorOnlySources, *source) {
		return fmt.Errorf("-source %q: only %v become anchor-only", *source, ledger.AnchorOnlySources)
	}
	l, err := ledger.Load(fs.Arg(0), roleKeys)
	if err != nil {
		return err
	}
	if *propose != "" && l.Config.PartitionVersion >= ledger.RoleHoldersFrom {
		return fmt.Errorf("ledger.yaml is at partition version %d; -propose prepares the move to version %d", l.Config.PartitionVersion, ledger.RoleHoldersFrom)
	}
	var records []*ledger.Reads
	for _, p := range fs.Args()[1:] {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		r, err := ledger.DecodeReads(bytes.NewReader(raw))
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		records = append(records, r)
	}
	var req []string
	if *require != "" {
		req = strings.Split(*require, ",")
	}
	rep, err := ledger.PrepareAnchorSource(*source, l.Units[*source], records, req)
	if err != nil {
		return err
	}
	b, err := yamlBytes(rep, "# Job-local D2 join report (M3.2 row 26). Not committed: it lists the units the next ledger.yaml leaves out.\n")
	if err != nil {
		return err
	}
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		return err
	}
	if *propose != "" {
		next, err := ledger.ProposeNextPartition(l.Config, map[string]*ledger.PrepReport{*source: rep})
		if err != nil {
			return err
		}
		pb, err := yamlBytes(next, "# PROPOSAL, written by `ledger partition-prep`. It keeps the seed record and the InferenceX rule. A maintainer reviews it\n# and commits it with the pull request that deletes the old published lists (README, \"Moving to partition version 2\").\n")
		if err != nil {
			return err
		}
		if err := os.WriteFile(*propose, pb, 0o644); err != nil {
			return err
		}
	}
	_, _ = fmt.Fprintf(stdout, "%s: %d units, %d read under a training role (left out), %d anchor; %d records, %d units read under two roles\n",
		*source, rep.Units, len(rep.LeftOut), rep.Anchor, len(rep.Consumers), rep.Conflicts)
	return nil
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if x == y {
			return true
		}
	}
	return false
}

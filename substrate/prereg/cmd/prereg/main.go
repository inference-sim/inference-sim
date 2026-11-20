// Command prereg checks the calibration pre-registrations under registry/gates/ (plan Table W row 18).
//
//	prereg check               Verify the defaults and every record against the ledger, the
//	                           coefficient sets and the engines' code IDs, and scan the fit scripts.
//	prereg pins SOURCE...      Print the pins block of a record that covers the sources.
//	prereg admit ID STEP SOURCE ROLE
//	                           Exit zero only if a fit of record ID may read SOURCE at ROLE.
//
// It runs from the substrate directory. No command reads a unit or runs a fit.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/prereg"
)

const (
	gatesDir = "registry/gates"
	root     = "."
)

var env = prereg.Env{LedgerDir: "catalog/ledger", CoefficientsDir: "registry/coefficients", EngineCodeIDs: "registry/inputs/engine_code_ids.json"}

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		_, _ = fmt.Fprintln(stderr, "usage: prereg <check|pins|admit> ...")
		return 2
	}
	r, err := prereg.Load(gatesDir)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 1
	}
	switch args[0] {
	case "check":
		if err := r.Verify(env); err != nil {
			_, _ = fmt.Fprintln(stderr, err)
			return 1
		}
		if err := r.ScanScripts(root); err != nil {
			_, _ = fmt.Fprintln(stderr, err)
			return 1
		}
		_, _ = fmt.Fprintf(stdout, "ok: %d pre-registrations\n", len(r.Records))
	case "pins":
		p, err := prereg.ComputePins(env.LedgerDir, r.Defaults, args[1:])
		if err != nil {
			_, _ = fmt.Fprintln(stderr, err)
			return 1
		}
		_, _ = fmt.Fprintf(stdout, "pins:\n  partition_version: %d\n  seed_sha256: %q\n  defaults_sha256: %q\n", p.PartitionVersion, p.SeedSHA256, p.DefaultsSHA256)
		for _, name := range []struct {
			title string
			m     map[string]string
		}{{"sources", p.Sources}, {"cut_points", p.CutPoints}} {
			_, _ = fmt.Fprintf(stdout, "  %s:\n", name.title)
			for _, s := range args[1:] {
				_, _ = fmt.Fprintf(stdout, "    %s: %q\n", s, name.m[s])
			}
		}
	case "admit":
		if len(args) != 5 {
			_, _ = fmt.Fprintln(stderr, "usage: prereg admit ID STEP SOURCE ROLE")
			return 2
		}
		if err := r.Admit(env.LedgerDir, args[1], prereg.Step(args[2]), args[3], ledger.Role(args[4])); err != nil {
			_, _ = fmt.Fprintln(stderr, err)
			return 1
		}
	default:
		_, _ = fmt.Fprintf(stderr, "unknown command %q\n", args[0])
		return 2
	}
	return 0
}

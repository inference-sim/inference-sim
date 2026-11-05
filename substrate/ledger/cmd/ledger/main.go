// Command ledger generates and checks the data ledger under catalog/ledger/ (design §10.2).
//
//	ledger ingest-inferencex -rows R -partition P [-revision REV] DIR
//	    Generate units/inferencex.jsonl.gz, reports/inferencex.yaml, legacy-map.yaml and
//	    legacy-points.jsonl.gz in DIR from the rows export.sql printed.
//	ledger check DIR
//	    Load and check every committed ledger file (D3, D4, D6).
//	ledger check-generated -rows R -partition P DIR
//	    Regenerate into a scratch directory and require each generated file to equal the
//	    committed one byte for byte (D4).
//	ledger check-base BASE DIR
//	    D4's check against the ledger at the pull request's merge base, including that each
//	    added alias table equals the generated one.
//	ledger generate-aliases -source S -reason TEXT -out FILE BASE DIR
//	    Generate the alias table that pairs BASE's and DIR's identities for the same records.
//	ledger scan-roles ROOT
//	    Refuse any tracked file under ROOT, other than the published lists, that assigns a role.
//	ledger lint-workflows WORKFLOWS
//	    D7's lint of the seed-holding workflows.
//	ledger release -out DIR LEDGER          (seed-holding: reads LEDGER_SEED)
//	ledger join -assign LEDGER READS...     (seed-holding with -assign: reads LEDGER_SEED)
//	ledger check-report REPORT LEDGER
//	    Refuse a seed-holding job's output that names a unit or aggregates fewer than 10 (D7).
//	ledger candidate-units -out FILE LEDGER
//	    List the units a candidate job predicts: every unit not on the published lists.
//	ledger measured-inferencex -rows R -out FILE LEDGER
//	    Write each InferenceX unit's measured values and lane, for the seed-holding jobs.
//	ledger aggregate -job J -role R -predictions P -measured M -out FILE LEDGER  (seed-holding)
//	    Reduce a candidate's predictions on the units of role R to aggregates (D7).
//
// Exit 0 when everything passes, 1 when a check fails, 2 on misuse.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/ledger/inferencex"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

var roleKeys = map[string]ledger.RoleKeyFunc{
	inferencex.SourceID: inferencex.RoleKey,
}

type usageError struct{ msg string }

func (u usageError) Error() string { return u.msg }

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		_, _ = fmt.Fprintln(stderr, "usage: ledger <ingest-inferencex|check|check-generated|check-base|generate-aliases|scan-roles|lint-workflows|release|join|check-report|candidate-units|measured-inferencex|aggregate> ...")
		return 2
	}
	cmds := map[string]func([]string, io.Writer) error{
		"ingest-inferencex":   cmdIngest,
		"check":               cmdCheck,
		"check-generated":     cmdCheckGenerated,
		"check-base":          cmdCheckBase,
		"generate-aliases":    cmdGenerateAliases,
		"scan-roles":          cmdScanRoles,
		"lint-workflows":      cmdLintWorkflows,
		"release":             cmdRelease,
		"join":                cmdJoin,
		"check-report":        cmdCheckReport,
		"candidate-units":     cmdCandidateUnits,
		"measured-inferencex": cmdMeasuredInferenceX,
		"aggregate":           cmdAggregate,
	}
	f, ok := cmds[args[0]]
	if !ok {
		_, _ = fmt.Fprintf(stderr, "ledger: unknown command %q\n", args[0])
		return 2
	}
	if err := f(args[1:], stdout); err != nil {
		_, _ = fmt.Fprintf(stderr, "ledger %s: %v\n", args[0], err)
		var u usageError
		if errors.As(err, &u) {
			return 2
		}
		return 1
	}
	return 0
}

func flags(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	return fs
}

func parse(fs *flag.FlagSet, args []string, nargs int, usage string) error {
	if err := fs.Parse(args); err != nil || fs.NArg() != nargs {
		return usageError{"usage: ledger " + usage}
	}
	return nil
}

// generate runs the InferenceX ingest and returns each generated file by its path under the
// ledger directory.
func generate(rowsPath, partitionPath, dir string) (map[string][]byte, *inferencex.Report, error) {
	raw, err := os.ReadFile(rowsPath)
	if err != nil {
		return nil, nil, err
	}
	sum := sha256.Sum256(raw)
	sources, err := ledger.LoadSources(dir)
	if err != nil {
		return nil, nil, err
	}
	src := sources.Lookup(inferencex.SourceID)
	if src == nil {
		return nil, nil, fmt.Errorf("%s names no %q source", ledger.FileSources, inferencex.SourceID)
	}
	rows, err := inferencex.ReadRows(bytes.NewReader(raw))
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", rowsPath, err)
	}
	units, rep, err := inferencex.Ingest(rows, src.Revision.Ref, hex.EncodeToString(sum[:]))
	if err != nil {
		return nil, nil, err
	}
	out := map[string][]byte{}
	var gz bytes.Buffer
	if err := ledger.EncodeUnits(&gz, units); err != nil {
		return nil, nil, err
	}
	out[src.Units] = gz.Bytes()
	if out["reports/"+inferencex.SourceID+".yaml"], err = yamlBytes(rep, "# Generated by `go run ./ledger/cmd/ledger ingest-inferencex`. Do not edit.\n"); err != nil {
		return nil, nil, err
	}

	praw, err := os.ReadFile(partitionPath)
	if err != nil {
		return nil, nil, err
	}
	labels, seed, err := inferencex.ReadPartition(bytes.NewReader(praw))
	if err != nil {
		return nil, nil, err
	}
	psum := sha256.Sum256(praw)
	part := ledger.LegacyPartition{URL: legacyPartitionURL, SHA256: hex.EncodeToString(psum[:]), Seed: seed}
	if part.SHA256 != legacyPartitionSHA256 {
		return nil, nil, fmt.Errorf("%s: sha256 %s, want the pinned partition %s", partitionPath, part.SHA256, legacyPartitionSHA256)
	}
	lm, points := ledger.BuildLegacy(inferencex.SourceID, part, labels, inferencex.PointUnits(rows, units, labels), units, legacyPointsFile)
	if out[ledger.FileLegacyMap], err = yamlBytes(lm, "# The legacy table (design §10.2): each legacy unit's side, FIT or HELD, from the partition\n# drawn before the ledger existed. Sides, not roles. Generated by\n# `go run ./ledger/cmd/ledger ingest-inferencex`. Do not edit.\n"); err != nil {
		return nil, nil, err
	}
	var pts bytes.Buffer
	if err := encodeLines(&pts, points); err != nil {
		return nil, nil, err
	}
	out[legacyPointsFile] = pts.Bytes()
	return out, rep, nil
}

const (
	legacyPartitionURL    = "https://github.com/inference-sim/inference-sim/blob/8e344596c35272e6f505df65c0935aeb5e0000ec/tools/validation/inferencex/partition/partition.v5.json.gz"
	legacyPartitionSHA256 = "90f1df60f5aa9decff7094d035ea44f4b63055e61616d90bcc0dcc30f4891d7e"
	legacyPointsFile      = "legacy-points.jsonl.gz"
)

func encodeLines(w io.Writer, points []ledger.LegacyPoint) error {
	var lines bytes.Buffer
	for _, p := range points {
		b, err := ledger.CanonicalStruct(p)
		if err != nil {
			return err
		}
		lines.Write(b)
		lines.WriteByte('\n')
	}
	return ledger.Gzip(w, lines.Bytes())
}

func yamlBytes(v any, header string) ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(header)
	enc := yaml.NewEncoder(&b)
	enc.SetIndent(2)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	if err := enc.Close(); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func cmdIngest(args []string, stdout io.Writer) error {
	fs := flags("ingest-inferencex")
	rows := fs.String("rows", "", "rows printed by ledger/inferencex/export.sql")
	partition := fs.String("partition", "", "the legacy partition, partition.v5.json.gz")
	if err := parse(fs, args, 1, "ingest-inferencex -rows R -partition P DIR"); err != nil {
		return err
	}
	if *rows == "" || *partition == "" {
		return usageError{"-rows and -partition are required"}
	}
	dir := fs.Arg(0)
	files, rep, err := generate(*rows, *partition, dir)
	if err != nil {
		return err
	}
	for _, rel := range sortedKeys(files) {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(p, files[rel], 0o644); err != nil {
			return err
		}
	}
	_, _ = fmt.Fprintf(stdout, "inferencex: %d records, %d units, %d role keys, %d duplicates, %d merged, %d joined units\n",
		rep.Records, rep.Units, rep.RoleKeys, rep.Duplicates, rep.Merged, rep.JoinedUnits)
	return nil
}

func cmdCheck(args []string, stdout io.Writer) error {
	fs := flags("check")
	if err := parse(fs, args, 1, "check DIR"); err != nil {
		return err
	}
	l, err := ledger.Load(fs.Arg(0), roleKeys)
	if err != nil {
		return err
	}
	n := 0
	for _, us := range l.Units {
		n += len(us)
	}
	if n == 0 {
		return errors.New("the ledger holds no unit; a check over nothing passes nothing (X22)")
	}
	_, _ = fmt.Fprintf(stdout, "ledger: %d sources, %d units, partition version %d, seed recorded: %v\n",
		len(l.Sources.Sources), n, l.Config.PartitionVersion, l.Config.Seed.Created())
	return nil
}

func cmdCheckGenerated(args []string, stdout io.Writer) error {
	fs := flags("check-generated")
	rows := fs.String("rows", "", "rows printed by ledger/inferencex/export.sql")
	partition := fs.String("partition", "", "the legacy partition")
	if err := parse(fs, args, 1, "check-generated -rows R -partition P DIR"); err != nil {
		return err
	}
	dir := fs.Arg(0)
	files, _, err := generate(*rows, *partition, dir)
	if err != nil {
		return err
	}
	var errs []error
	for _, rel := range sortedKeys(files) {
		want := files[rel]
		got, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if strings.HasSuffix(rel, ".gz") {
			// Compare the content, not the compressor's output.
			if want, err = ledger.Gunzip(want); err == nil {
				got, err = ledger.Gunzip(got)
			}
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", rel, err))
				continue
			}
		}
		if !bytes.Equal(got, want) {
			errs = append(errs, fmt.Errorf("%s differs from the ingest tool's output; regenerate it with ledger ingest-inferencex", rel))
		}
	}
	if err := errors.Join(errs...); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "every generated ledger file equals the committed one (%d files)\n", len(files))
	return nil
}

func cmdCheckBase(args []string, stdout io.Writer) error {
	fs := flags("check-base")
	if err := parse(fs, args, 2, "check-base BASE DIR"); err != nil {
		return err
	}
	base, err := ledger.Load(fs.Arg(0), roleKeys)
	if err != nil {
		return fmt.Errorf("the merge base's ledger: %w", err)
	}
	cand, err := ledger.Load(fs.Arg(1), roleKeys)
	if err != nil {
		return err
	}
	if err := errors.Join(ledger.CheckAgainstBase(base, cand), ledger.CheckAliasesAgainstBase(base, cand)); err != nil {
		return err
	}
	_, _ = fmt.Fprintln(stdout, "every unit at the merge base keeps its unit_id and role key, or is aliased")
	return nil
}

func cmdScanRoles(args []string, stdout io.Writer) error {
	fs := flags("scan-roles")
	if err := parse(fs, args, 1, "scan-roles ROOT"); err != nil {
		return err
	}
	root := fs.Arg(0)
	out, err := exec.Command("git", "-C", root, "ls-files", "-z").Output()
	if err != nil {
		return fmt.Errorf("git ls-files: %w", err)
	}
	var files []string
	for _, f := range strings.Split(string(out), "\x00") {
		if f != "" && scannable(f) {
			files = append(files, f)
		}
	}
	if len(files) == 0 {
		return errors.New("no files to scan (X22)")
	}
	if err := ledger.CheckNoRoles(root, files, isRoleFixture); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "%d files scanned; none assigns a role outside the published lists\n", len(files))
	return nil
}

// scannable selects data files: the role scan reads data, not code that names the roles.
func scannable(f string) bool {
	for _, ext := range []string{".yaml", ".yml", ".json", ".jsonl", ".jsonl.gz", ".csv", ".tsv"} {
		if strings.HasSuffix(f, ext) && !strings.HasPrefix(f, ".github/") {
			return true
		}
	}
	return false
}

// isRoleFixture names the test fixtures that hold roles on purpose, as known-bad inputs.
func isRoleFixture(f string) bool { return strings.HasPrefix(f, "ledger/testdata/roles/") }

func cmdLintWorkflows(args []string, stdout io.Writer) error {
	fs := flags("lint-workflows")
	if err := parse(fs, args, 1, "lint-workflows WORKFLOWS_DIR"); err != nil {
		return err
	}
	n, err := ledger.LintWorkflows(fs.Arg(0))
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "%d seed-holding jobs: each runs only from main in environment %q, and no candidate job holds the seed\n", n, ledger.SeedEnvironment)
	return nil
}

func seedFromEnv() (string, error) {
	s := os.Getenv("LEDGER_SEED")
	if s == "" {
		return "", errors.New("LEDGER_SEED is not set; only a seed-holding job in the ledger environment has it")
	}
	return s, nil
}

func cmdRelease(args []string, stdout io.Writer) error {
	fs := flags("release")
	out := fs.String("out", "", "directory to write fit.yaml, anchor.yaml and report.yaml into")
	if err := parse(fs, args, 1, "release -out DIR LEDGER"); err != nil {
		return err
	}
	l, err := ledger.Load(fs.Arg(0), roleKeys)
	if err != nil {
		return err
	}
	seed, err := seedFromEnv()
	if err != nil {
		return err
	}
	a, err := ledger.NewAssigner(l.Config, l.Legacy, roleKeys, seed)
	if err != nil {
		return err
	}
	lists, err := a.Release(l.Config, l.Units)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		return err
	}
	for r, p := range lists {
		b, err := yamlBytes(p, "# Written by the release job (design §10.2). Do not edit.\n")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(*out, string(r)+".yaml"), b, 0o644); err != nil {
			return err
		}
	}
	_, _ = fmt.Fprintf(stdout, "published %d fit, %d anchor and %d report units\n",
		len(lists[ledger.RoleFit].Units), len(lists[ledger.RoleAnchor].Units), len(lists[ledger.RoleReport].Units))
	return nil
}

func cmdJoin(args []string, stdout io.Writer) error {
	fs := flags("join")
	assignDir := fs.String("assign", "", "ledger directory; with it, also check each read against the role the seed assigns")
	if err := fs.Parse(args); err != nil || fs.NArg() == 0 {
		return usageError{"usage: ledger join [-assign LEDGER] READS..."}
	}
	var records []*ledger.Reads
	for _, p := range fs.Args() {
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
	var assign func(string, string) (ledger.Role, error)
	if *assignDir != "" {
		l, err := ledger.Load(*assignDir, roleKeys)
		if err != nil {
			return err
		}
		seed, err := seedFromEnv()
		if err != nil {
			return err
		}
		a, err := ledger.NewAssigner(l.Config, l.Legacy, roleKeys, seed)
		if err != nil {
			return err
		}
		index := map[string]*ledger.Unit{}
		for src, us := range l.Units {
			for i := range us {
				index[src+"/"+us[i].UnitID] = &us[i]
			}
		}
		assign = func(src, id string) (ledger.Role, error) {
			u, ok := index[src+"/"+id]
			if !ok {
				return "", fmt.Errorf("a record reads %s unit %s, which the ledger does not hold", src, id)
			}
			return a.Role(src, u)
		}
	}
	conflicts, err := ledger.Join(records, assign)
	if err != nil {
		return err
	}
	if len(conflicts) > 0 {
		// The count only: naming the units or their roles would publish held members.
		return fmt.Errorf("%d units read under two roles, or under a role the seed does not give them (D2)", len(conflicts))
	}
	n := 0
	for _, r := range records {
		n += len(r.Reads)
	}
	if n == 0 {
		return errors.New("the records hold no read; a join over nothing passes nothing (X22)")
	}
	_, _ = fmt.Fprintf(stdout, "%d records, %d reads: no unit read under two roles\n", len(records), n)
	return nil
}

func cmdCheckReport(args []string, stdout io.Writer) error {
	fs := flags("check-report")
	if err := parse(fs, args, 2, "check-report REPORT LEDGER"); err != nil {
		return err
	}
	raw, err := os.ReadFile(fs.Arg(0))
	if err != nil {
		return err
	}
	l, err := ledger.Load(fs.Arg(1), roleKeys)
	if err != nil {
		return err
	}
	var ids []string
	for _, us := range l.Units {
		for _, u := range us {
			ids = append(ids, u.UnitID)
		}
	}
	if err := ledger.CheckReport(raw, ids); err != nil {
		return err
	}
	var r ledger.Report
	_ = json.Unmarshal(raw, &r)
	_, _ = fmt.Fprintf(stdout, "%d aggregates, each over at least %d units; no unit named\n", len(r.Aggregates), ledger.MinAggregateUnits)
	return nil
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// published reads the three published lists, if the release job has written them.
func published(dir string) (map[string]bool, error) {
	out := map[string]bool{}
	for _, r := range []ledger.Role{ledger.RoleFit, ledger.RoleAnchor, ledger.RoleReport} {
		p := filepath.Join(dir, ledger.DirPublished, string(r)+".yaml")
		raw, err := os.ReadFile(p)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		var l ledger.Published
		dec := yaml.NewDecoder(bytes.NewReader(raw))
		dec.KnownFields(true)
		if err := dec.Decode(&l); err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		for _, id := range l.Units {
			out[id] = true
		}
	}
	return out, nil
}

// CandidateUnit is one unit a candidate job is handed.
type CandidateUnit struct {
	Source   string         `json:"source"`
	UnitID   string         `json:"unit_id"`
	Identity map[string]any `json:"identity"`
}

func cmdCandidateUnits(args []string, stdout io.Writer) error {
	fs := flags("candidate-units")
	out := fs.String("out", "", "file to write")
	if err := parse(fs, args, 1, "candidate-units -out FILE LEDGER"); err != nil {
		return err
	}
	l, err := ledger.Load(fs.Arg(0), roleKeys)
	if err != nil {
		return err
	}
	pub, err := published(fs.Arg(0))
	if err != nil {
		return err
	}
	var list []CandidateUnit
	for _, src := range sortedKeys(l.Units) {
		for _, u := range l.Units[src] {
			if !pub[u.UnitID] {
				list = append(list, CandidateUnit{Source: src, UnitID: u.UnitID, Identity: u.Identity})
			}
		}
	}
	if len(list) == 0 {
		return errors.New("no unit to predict (X22)")
	}
	b, err := json.Marshal(list)
	if err != nil {
		return err
	}
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "%d units outside the published lists\n", len(list))
	return nil
}

func cmdMeasuredInferenceX(args []string, stdout io.Writer) error {
	fs := flags("measured-inferencex")
	rowsPath := fs.String("rows", "", "rows printed by ledger/inferencex/export.sql")
	out := fs.String("out", "", "file to write")
	if err := parse(fs, args, 1, "measured-inferencex -rows R -out FILE LEDGER"); err != nil {
		return err
	}
	l, err := ledger.Load(fs.Arg(0), roleKeys)
	if err != nil {
		return err
	}
	rawRows, err := os.ReadFile(*rowsPath)
	if err != nil {
		return err
	}
	rows, err := inferencex.ReadRows(bytes.NewReader(rawRows))
	if err != nil {
		return err
	}
	m, err := inferencex.MeasuredValues(rows, l.Units[inferencex.SourceID])
	if err != nil {
		return err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if err := os.WriteFile(*out, b, 0o600); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "measured values for %d units\n", len(m.Values))
	return nil
}

func cmdAggregate(args []string, stdout io.Writer) error {
	fs := flags("aggregate")
	job := fs.String("job", "", "select, error or gate")
	role := fs.String("role", "", "select, validate or held")
	predPath := fs.String("predictions", "", "the candidate job's predictions file (untrusted)")
	unitsPath := fs.String("units", "", "the units the candidate was handed (candidate-units, run from main)")
	measPath := fs.String("measured", "", "measured values (measured-inferencex)")
	out := fs.String("out", "", "report to write")
	if err := parse(fs, args, 1, "aggregate -job J -role R -predictions P -units U -measured M -out FILE LEDGER"); err != nil {
		return err
	}
	pairs := map[string]string{"select": "select", "error": "validate", "gate": "held"}
	if pairs[*job] == "" || pairs[*job] != *role {
		return usageError{fmt.Sprintf("-job %q -role %q: the select job reads select, the error job validate, the gate job held", *job, *role)}
	}
	l, err := ledger.Load(fs.Arg(0), roleKeys)
	if err != nil {
		return err
	}
	seed, err := seedFromEnv()
	if err != nil {
		return err
	}
	a, err := ledger.NewAssigner(l.Config, l.Legacy, roleKeys, seed)
	if err != nil {
		return err
	}
	rawUnits, err := os.ReadFile(*unitsPath)
	if err != nil {
		return err
	}
	var handed []CandidateUnit
	if err := json.Unmarshal(rawUnits, &handed); err != nil {
		return fmt.Errorf("%s: %w", *unitsPath, err)
	}
	want := map[string]bool{}
	for _, u := range handed {
		want[u.Source+"/"+u.UnitID] = true
	}
	pf, err := os.Open(*predPath)
	if err != nil {
		return err
	}
	pred, err := ledger.ReadPredictions(pf, want)
	if cerr := pf.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	rawM, err := os.ReadFile(*measPath)
	if err != nil {
		return err
	}
	var meas ledger.Measured
	if err := json.Unmarshal(rawM, &meas); err != nil {
		return fmt.Errorf("%s: %w", *measPath, err)
	}
	index := map[string]*ledger.Unit{}
	var ids []string
	for src, us := range l.Units {
		for i := range us {
			index[src+"/"+us[i].UnitID] = &us[i]
			ids = append(ids, us[i].UnitID)
		}
	}
	rep, err := ledger.AggregatePredictions(*job, *role, pred, &meas, func(src, id string) (ledger.Role, error) {
		u, ok := index[src+"/"+id]
		if !ok {
			return "", fmt.Errorf("predictions name %s unit %s, which the ledger does not hold", src, id)
		}
		return a.Role(src, u)
	}, func(src, id string) bool {
		_, on := a.LegacySide(src, index[src+"/"+id])
		return on
	})
	if err != nil {
		return err
	}
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	if err := ledger.CheckReport(b, ids); err != nil {
		return err
	}
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "%d aggregates; evaluated: %v\n", len(rep.Aggregates), rep.Evaluated)
	return nil
}

func cmdGenerateAliases(args []string, stdout io.Writer) error {
	fs := flags("generate-aliases")
	source := fs.String("source", "", "source id")
	reason := fs.String("reason", "", "why identities changed")
	out := fs.String("out", "", "alias table to write, under DIR/aliases/")
	if err := parse(fs, args, 2, "generate-aliases -source S -reason TEXT -out FILE BASE DIR"); err != nil {
		return err
	}
	if *source == "" || *reason == "" || *out == "" {
		return usageError{"-source, -reason and -out are required"}
	}
	base, err := ledger.Load(fs.Arg(0), roleKeys)
	if err != nil {
		return fmt.Errorf("the merge base's ledger: %w", err)
	}
	cand, err := ledger.Load(fs.Arg(1), roleKeys)
	if err != nil {
		return err
	}
	t, err := ledger.GenerateAliases(*source, *reason, base.Units[*source], cand.Units[*source])
	if err != nil {
		return err
	}
	if len(t.Alias) == 0 {
		_, _ = fmt.Fprintln(stdout, "every unit keeps its unit_id and role key; no alias table is needed")
		return nil
	}
	b, err := yamlBytes(t, "# Generated by `go run ./ledger/cmd/ledger generate-aliases`. Do not edit.\n")
	if err != nil {
		return err
	}
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "%d aliases written to %s\n", len(t.Alias), *out)
	return nil
}

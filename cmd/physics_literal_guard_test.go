package cmd

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

// #1819: two STATIC guards, both here because both catch the same failure mode — a physics
// number that is authored in this repository instead of being read from its source of truth,
// and that no behavioural test can see because it looks like ordinary Go.
//
//	BC-G1  no `cmd/` flag default prices physics with a numeric literal. #1819 converted
//	       --kv-transfer-bandwidth from a 100.0 blocks/tick default into a derivation from
//	       the catalog cpu_dram device; without a guard, the next edit that "restores a
//	       sensible default" silently undoes it, and every test above would still pass
//	       (they exercise the derivation, not the flag table).
//	BC-G2  the live defaults.yaml `lora:` block does not diverge from blis-registry's
//	       coefficients/lora-adapter-costs.yaml, which is a FROZEN SNAPSHOT of it. Those 9
//	       values are not a `cmd/` flag default, so BC-G1 does not reach them, and nothing
//	       in the registry repository catches upstream drift — so it is caught source-side
//	       here (R2G3b / blis-registry PR #14 review thread).

// ---------------------------------------------------------------------------
// BC-G1: the detector
// ---------------------------------------------------------------------------

// physicsPricedFlagPattern matches the flag names whose numeric value PRICES A PHYSICAL
// QUANTITY — a bus rate, a per-transfer latency, a byte size, a FLOP rate. Those belong to
// the catalog (device / hardware facts) or the registry (fitted and modelling coefficients),
// never to a Go flag default: a default is invisible to both, carries no provenance, and
// cannot be argued about.
//
// It deliberately does NOT match every numeric flag. --max-num-seqs, --rate,
// --gpu-memory-utilization and friends are POLICY and deployment knobs — operator inputs
// with no physical referent — and a default is the right home for them.
var physicsPricedFlagPattern = regexp.MustCompile(
	`(?:^|-)(bandwidth|base-latency|bytes|bytes-us|bytes-per-rank|flops|tflops)(?:$|-)`)

// physicsPricedFlagExceptions are the flags that match the pattern, still carry a non-zero
// literal default, and are NOT yet converted. Each entry must name why and what would retire
// it. The list is deliberately awkward to extend: adding to it is a visible admission, which
// is the point.
//
// It is NOT a way to keep a number here — it is a record of what R2 has not reached yet.
var physicsPricedFlagExceptions = map[string]string{
	"pd-transfer-bandwidth": "PD fabric rate; transcribed in blis-registry " +
		"coefficients/pd-transfer-estimates.yaml as pd_transfer_bandwidth but not yet derived " +
		"from a catalog fabric class in cmd/. Retired by the PD half of R2, not by #1819.",
	"pd-transfer-base-latency": "PD modelling per-transfer latency; transcribed in " +
		"blis-registry coefficients/pd-transfer-estimates.yaml as pd_transfer_base_latency " +
		"but still authored here. Retired by the PD half of R2, not by #1819.",
}

// flagDefaultFinding is one `cmd.Flags().<T>Var(&v, "name", <literal>, ...)` whose default is
// a non-zero numeric literal and whose name prices physics.
type flagDefaultFinding struct {
	file string
	line int
	flag string
	lit  string
}

// physicsPricedFlagDefaults walks a parsed Go file for pflag registrations and reports the
// physics-priced ones carrying a non-zero numeric literal default.
//
// A pure function over an AST rather than a regexp over text, so it sees the actual third
// argument of the call (the default) and cannot be fooled by the flag name appearing in help
// text, a comment, or an unrelated string. Kept exported-in-package and separately testable
// so TestPhysicsPricedFlagDefaults_DetectorRejectsANewLiteral can prove it FIRES — a guard
// that silently matches nothing is worse than no guard.
func physicsPricedFlagDefaults(fset *token.FileSet, file *ast.File, name string) []flagDefaultFinding {
	var findings []flagDefaultFinding
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) < 3 {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || !strings.HasSuffix(sel.Sel.Name, "Var") {
			return true
		}
		// The receiver must be a Flags()/PersistentFlags() call, so this matches flag
		// registration and not any other three-argument ...Var method.
		recv, ok := sel.X.(*ast.CallExpr)
		if !ok {
			return true
		}
		recvSel, ok := recv.Fun.(*ast.SelectorExpr)
		if !ok || (recvSel.Sel.Name != "Flags" && recvSel.Sel.Name != "PersistentFlags") {
			return true
		}
		flagName, ok := stringLiteral(call.Args[1])
		if !ok || !physicsPricedFlagPattern.MatchString(flagName) {
			return true
		}
		lit, isNumeric := nonZeroNumericLiteral(call.Args[2])
		if !isNumeric {
			return true
		}
		findings = append(findings, flagDefaultFinding{
			file: name, line: fset.Position(call.Args[2].Pos()).Line, flag: flagName, lit: lit,
		})
		return true
	})
	return findings
}

// flagVarCallCounts counts the `<something>.<T>Var("name", ...)` calls in a file two ways:
// `viaAccessor` applies the same Flags()/PersistentFlags() receiver filter the detector does,
// `total` does not. They must be equal for the detector's coverage to be complete — see
// TestPhysicsPricedFlagDefaults_DetectorSeesEveryRegistration.
func flagVarCallCounts(file *ast.File) (viaAccessor, total int) {
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) < 3 {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || !strings.HasSuffix(sel.Sel.Name, "Var") {
			return true
		}
		if _, isString := stringLiteral(call.Args[1]); !isString {
			return true
		}
		total++
		recv, ok := sel.X.(*ast.CallExpr)
		if !ok {
			return true
		}
		recvSel, ok := recv.Fun.(*ast.SelectorExpr)
		if !ok || (recvSel.Sel.Name != "Flags" && recvSel.Sel.Name != "PersistentFlags") {
			return true
		}
		viaAccessor++
		return true
	})
	return viaAccessor, total
}

// stringLiteral unwraps a plain string literal argument.
func stringLiteral(e ast.Expr) (string, bool) {
	basic, ok := e.(*ast.BasicLit)
	if !ok || basic.Kind != token.STRING {
		return "", false
	}
	s, err := strconv.Unquote(basic.Value)
	if err != nil {
		return "", false
	}
	return s, true
}

// nonZeroNumericLiteral reports whether an expression is a numeric literal (optionally
// negated) that is not zero. A ZERO default is exactly how a converted flag signals
// "absent ⇒ derive / not charged", so zero is the shape the guard wants and must not flag.
// A named constant or a function call is not a literal here: it has a declaration site that
// can carry provenance, which is the distinction the guard is drawing.
func nonZeroNumericLiteral(e ast.Expr) (string, bool) {
	switch v := e.(type) {
	case *ast.UnaryExpr:
		if v.Op != token.SUB && v.Op != token.ADD {
			return "", false
		}
		inner, ok := nonZeroNumericLiteral(v.X)
		if !ok {
			return "", false
		}
		return v.Op.String() + inner, true
	case *ast.BasicLit:
		if v.Kind != token.INT && v.Kind != token.FLOAT {
			return "", false
		}
		f, err := strconv.ParseFloat(v.Value, 64)
		if err != nil || f == 0 {
			return "", false
		}
		return v.Value, true
	}
	return "", false
}

// ---------------------------------------------------------------------------
// BC-G1: the guard over this package's real sources
// ---------------------------------------------------------------------------

// TestPhysicsPricedFlagDefaults_NoneInProductionSources is BC-G1. It fails when a flag whose
// name prices physics is given a non-zero literal default anywhere in cmd/'s production
// sources — which is precisely the edit that would undo #1819.
func TestPhysicsPricedFlagDefaults_NoneInProductionSources(t *testing.T) {
	files, scanned := parseCmdProductionSources(t)
	if scanned == 0 {
		t.Fatal("non-vacuity: no production sources were scanned")
	}

	var offenders []flagDefaultFinding
	for name, parsed := range files {
		offenders = append(offenders, physicsPricedFlagDefaults(parsed.fset, parsed.file, name)...)
	}
	sort.Slice(offenders, func(i, j int) bool {
		if offenders[i].file != offenders[j].file {
			return offenders[i].file < offenders[j].file
		}
		return offenders[i].line < offenders[j].line
	})

	seen := map[string]bool{}
	for _, o := range offenders {
		seen[o.flag] = true
		if _, allowed := physicsPricedFlagExceptions[o.flag]; allowed {
			continue
		}
		t.Errorf("%s:%d: --%s is given the physics literal %s as a flag default.\n"+
			"  A number that prices a physical quantity belongs to the CATALOG (a device or "+
			"hardware fact) or to blis-registry (a fitted/modelling coefficient), and must be "+
			"DERIVED here — see cmd/kv_transfer_derive.go for the shape (#1819, R2G3b).\n"+
			"  If it genuinely cannot be converted yet, add it to physicsPricedFlagExceptions "+
			"with the reason and what retires it.",
			o.file, o.line, o.flag, o.lit)
	}

	// A stale exception is as bad as a missing guard: it silently pre-authorizes a flag
	// nobody is watching any more.
	for flagName := range physicsPricedFlagExceptions {
		if !seen[flagName] {
			t.Errorf("physicsPricedFlagExceptions lists --%s, which no longer carries a non-zero "+
				"literal flag default in cmd/ — delete the exception", flagName)
		}
	}

	// The flag #1819 converted must never be excused.
	if _, excused := physicsPricedFlagExceptions["kv-transfer-bandwidth"]; excused {
		t.Error("--kv-transfer-bandwidth must not be in physicsPricedFlagExceptions: #1819 " +
			"derives it from the catalog cpu_dram device, so an exception would mean the " +
			"conversion was reverted")
	}
}

// TestPhysicsPricedFlagDefaults_ConvertedFlagsAreZero is the positive statement of the same
// law for the flags #1819 and R2G3b touched: both legacy KV-transfer flags must register a
// ZERO default. Zero is load-bearing — it is what makes "unset" distinguishable from a rate
// (for --kv-transfer-bandwidth, unset ⇒ derive; for --kv-transfer-base-latency, the
// registry's `method: not_charged`).
// Both commands are checked: the flags live in the shared registerSimConfigFlags, so a
// divergence would also be an INV-13 defect.
func TestPhysicsPricedFlagDefaults_ConvertedFlagsAreZero(t *testing.T) {
	for _, flagName := range []string{"kv-transfer-bandwidth", "kv-transfer-base-latency"} {
		for cmdName, flags := range map[string]*pflag.FlagSet{
			"run": runCmd.Flags(), "replay": replayCmd.Flags(),
		} {
			f := flags.Lookup(flagName)
			if f == nil {
				t.Fatalf("%s must register --%s (registerSimConfigFlags)", cmdName, flagName)
			}
			got, err := strconv.ParseFloat(f.DefValue, 64)
			if err != nil {
				t.Fatalf("%s --%s default %q is not numeric: %v", cmdName, flagName, f.DefValue, err)
			}
			if got != 0 {
				t.Errorf("%s --%s default is %v, want 0 — a non-zero default re-authors physics in "+
					"a flag table (#1819)", cmdName, flagName, got)
			}
		}
	}
}

// TestPhysicsPricedFlagDefaults_DetectorSeesEveryRegistration closes the guard's one
// structural blind spot. The detector requires the receiver to be a literal
// `Flags()`/`PersistentFlags()` call, so a refactor to `fs := cmd.Flags(); fs.Float64Var(...)`
// would make it silently stop seeing that registration — the guard would still PASS while no
// longer guarding anything. This fails the moment such a style appears, naming the file.
func TestPhysicsPricedFlagDefaults_DetectorSeesEveryRegistration(t *testing.T) {
	files, scanned := parseCmdProductionSources(t)
	if scanned == 0 {
		t.Fatal("non-vacuity: no production sources were scanned")
	}
	totalSeen := 0
	for _, name := range sortedSourceNames(files) {
		viaAccessor, total := flagVarCallCounts(files[name].file)
		totalSeen += viaAccessor
		if viaAccessor != total {
			t.Errorf("%s: %d of %d flag registrations do not go through Flags()/PersistentFlags() "+
				"directly, so physicsPricedFlagDefaults cannot see them — either restore the "+
				"`cmd.Flags().<T>Var(...)` form or widen the detector's receiver check (#1819)",
				name, total-viaAccessor, total)
		}
	}
	// Non-vacuity: this package really does register flags, so the equality above is not
	// 0 == 0 in every file.
	if totalSeen == 0 {
		t.Fatal("non-vacuity: the detector saw no flag registrations at all in cmd/")
	}
}

// sortedSourceNames keeps per-file diagnostics deterministic (INV-6).
func sortedSourceNames(files map[string]parsedSource) []string {
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// parsedSource pairs a parsed file with its FileSet so positions stay resolvable.
type parsedSource struct {
	fset *token.FileSet
	file *ast.File
}

// parseCmdProductionSources parses every non-test Go file in cmd/. Production sources only:
// the test files here deliberately contain synthetic physics literals.
func parseCmdProductionSources(t *testing.T) (map[string]parsedSource, int) {
	t.Helper()
	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob cmd/*.go: %v", err)
	}
	out := make(map[string]parsedSource, len(names))
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		fset := token.NewFileSet()
		parsed, parseErr := parser.ParseFile(fset, name, nil, parser.ParseComments)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", name, parseErr)
		}
		out[name] = parsedSource{fset: fset, file: parsed}
	}
	return out, len(out)
}

// TestPhysicsPricedFlagDefaults_DetectorRejectsANewLiteral is the non-vacuity proof the
// acceptance criteria ask for: the guard REJECTS a new numeric physics literal added to a
// cmd/ flag default. Run against synthetic sources so it tests the detector rather than the
// current state of the tree.
func TestPhysicsPricedFlagDefaults_DetectorRejectsANewLiteral(t *testing.T) {
	cases := []struct {
		name     string
		src      string
		wantFlag string
	}{
		{
			name: "reintroduced_kv_transfer_bandwidth",
			src: `package cmd
func f(cmd *cobra.Command) {
	cmd.Flags().Float64Var(&kvTransferBandwidth, "kv-transfer-bandwidth", 100.0, "rate")
}`,
			wantFlag: "kv-transfer-bandwidth",
		},
		{
			name: "brand_new_physics_flag",
			src: `package cmd
func f(cmd *cobra.Command) {
	cmd.Flags().Float64Var(&x, "host-dram-bandwidth", 2.0e4, "bytes/us")
}`,
			wantFlag: "host-dram-bandwidth",
		},
		{
			name: "negative_literal_still_counts",
			src: `package cmd
func f(cmd *cobra.Command) {
	cmd.Flags().Float64Var(&x, "fabric-base-latency", -0.05, "ms")
}`,
			wantFlag: "fabric-base-latency",
		},
		{
			name: "persistent_flags_too",
			src: `package cmd
func f(cmd *cobra.Command) {
	cmd.PersistentFlags().Float64Var(&x, "weight-bytes", 2.0, "bytes")
}`,
			wantFlag: "weight-bytes",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := findingsFor(t, tc.src)
			if len(got) != 1 {
				t.Fatalf("detector must report exactly the one physics literal, got %d: %+v", len(got), got)
			}
			if got[0].flag != tc.wantFlag {
				t.Errorf("reported --%s, want --%s", got[0].flag, tc.wantFlag)
			}
		})
	}
}

// TestPhysicsPricedFlagDefaults_DetectorIgnoresLegitimateDefaults is the other half of
// non-vacuity: a guard that fires on everything would be turned off within a week. Policy
// knobs, zero sentinels and named constants must all pass.
func TestPhysicsPricedFlagDefaults_DetectorIgnoresLegitimateDefaults(t *testing.T) {
	src := `package cmd
func f(cmd *cobra.Command) {
	// Policy / deployment knobs: no physical referent.
	cmd.Flags().IntVar(&x, "max-num-seqs", 256, "policy")
	cmd.Flags().Float64Var(&y, "rate", 10.0, "arrivals/s")
	cmd.Flags().Float64Var(&z, "gpu-memory-utilization", 0.9, "fraction")
	cmd.Flags().Int64Var(&w, "snapshot-refresh-interval", 50000, "us")
	// A converted physics flag: zero means "absent => derive".
	cmd.Flags().Float64Var(&a, "kv-transfer-bandwidth", 0, "derived")
	cmd.Flags().Int64Var(&b, "kv-transfer-base-latency", 0, "not charged")
	// A named constant has a declaration site that can carry provenance.
	cmd.Flags().Float64Var(&c, "host-dram-bandwidth", hostDRAMBandwidth, "bytes/us")
	// The flag name in help text must not trip the AST detector.
	cmd.Flags().IntVar(&d, "num-requests", 100, "see --pd-transfer-bandwidth 25.0")
}`
	if got := findingsFor(t, src); len(got) != 0 {
		t.Errorf("detector must not flag policy knobs, zero sentinels or named constants, got: %+v", got)
	}
}

// findingsFor parses a synthetic source and runs the detector over it.
func findingsFor(t *testing.T, src string) []flagDefaultFinding {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, "synthetic.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse synthetic source: %v", err)
	}
	return physicsPricedFlagDefaults(fset, parsed, "synthetic.go")
}

// ---------------------------------------------------------------------------
// BC-G2: the LoRA defaults must not drift from the registry snapshot
// ---------------------------------------------------------------------------

// registryLoRACoefficients is the LoRA set as blis-registry froze it in
// coefficients/lora-adapter-costs.yaml (R2G3b, family #1) — nine values, TRANSCRIBED from
// this repository's defaults.yaml `lora:` block, never regenerated.
//
// It is a FROZEN GOLDEN in both directions. The registry's own transcription test proves the
// registry matches a snapshot taken at transcription time; nothing over there watches the
// LIVE defaults.yaml, so an edit here would silently make the two disagree and the registry's
// entry would quietly describe a number BLIS no longer ships. This is that missing half.
//
// If you are changing a LoRA coefficient: change it here AND in blis-registry in the same
// change set, and argue the physics — that is exactly the conversation this guard exists to
// force (#1508 carries the provenance: the Agullo Digital Twin, arXiv:2508.08343).
func registryLoRACoefficients() map[string]float64 {
	return map[string]float64{
		"load_base_latency_us":     1500.0,
		"load_bandwidth_bytes_us":  2.0e6,
		"footprint_bytes_per_rank": 2.0e6,
		"step_overhead_k6_rank8":   0.02,
		"step_overhead_k7_rank8":   1.0,
		"step_overhead_k6_rank16":  0.035,
		"step_overhead_k7_rank16":  1.0,
		"step_overhead_k6_rank32":  0.06,
		"step_overhead_k7_rank32":  1.0,
	}
}

// TestLoRADefaults_MatchRegistrySnapshot is BC-G2. It flattens the bundled defaults.yaml
// `lora:` block into the registry's coefficient names and compares the two sets exactly —
// both directions, so a NEW rank tier (which the registry does not carry) fails just as
// loudly as a changed value.
func TestLoRADefaults_MatchRegistrySnapshot(t *testing.T) {
	cfg := loadDefaultsConfig("../defaults.yaml")
	if cfg.LoRADefaults == nil {
		t.Fatal("defaults.yaml must declare a `lora:` block — blis-registry's " +
			"coefficients/lora-adapter-costs.yaml is a snapshot of it")
	}

	live := map[string]float64{
		"load_base_latency_us":     cfg.LoRADefaults.LoadBaseLatencyUs,
		"load_bandwidth_bytes_us":  cfg.LoRADefaults.LoadBandwidthBytesUs,
		"footprint_bytes_per_rank": cfg.LoRADefaults.FootprintBytesPerRank,
	}
	for rank, tier := range cfg.LoRADefaults.StepOverheadTiers {
		live[fmt.Sprintf("step_overhead_k6_rank%d", rank)] = tier.K6
		live[fmt.Sprintf("step_overhead_k7_rank%d", rank)] = tier.K7
	}

	want := registryLoRACoefficients()
	// Non-vacuity: the guard covers exactly the nine values the registry set carries.
	if len(want) != 9 {
		t.Fatalf("the registry LoRA set is nine values; the frozen snapshot has %d", len(want))
	}

	for _, name := range sortedKeys(want) {
		got, present := live[name]
		if !present {
			t.Errorf("defaults.yaml no longer supplies %s, which blis-registry's "+
				"lora-adapter-costs.yaml transcribes as %v — the registry entry would describe a "+
				"coefficient BLIS does not ship", name, want[name])
			continue
		}
		if got != want[name] {
			t.Errorf("defaults.yaml lora %s = %v, but blis-registry's lora-adapter-costs.yaml "+
				"froze %v.\n  These must move together: update blis-registry "+
				"coefficients/lora-adapter-costs.yaml in the same change set and argue the "+
				"physics (#1508 carries the provenance).", name, got, want[name])
		}
	}
	for _, name := range sortedKeys(live) {
		if _, present := want[name]; !present {
			t.Errorf("defaults.yaml lora declares %s = %v, which blis-registry's "+
				"lora-adapter-costs.yaml does not carry — add it there (with provenance) in the "+
				"same change set, or this coefficient ships with no source behind it",
				name, live[name])
		}
	}
}

// sortedKeys keeps the diagnostics deterministic (R2/INV-6): map iteration order must not
// decide the order failures are reported in.
func sortedKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

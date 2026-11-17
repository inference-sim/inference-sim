package cmd

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
	"github.com/inference-sim/inference-sim/sim/workload"
)

var (
	otelInputPath            string
	otelTraceOutput          string
	otelContextGrowth        string
	otelMaxThinkTime         time.Duration
	otelIncludeErrors        bool
	otelMinRounds            int
	otelThinkFromArrivalGaps bool
)

var convertOtelCmd = &cobra.Command{
	Use:   "otel",
	Short: "Convert OTel agentic trace JSON to a TraceV2 file for closed-loop replay",
	Long: `Convert OpenTelemetry agentic trace JSON (one agent session per trace) into a
TraceV2 pair (<prefix>.yaml + <prefix>.csv) suitable for 'blis replay --concurrent-sessions'.

Input may be a single JSON file ({"spans":[...]}), a directory of such files, or a
.jsonl file (one trace per line). Each trace becomes one session; each LLM chat span
becomes one round. Per-round input token counts are stored as deltas so accumulate
replay reconstructs the exact growing prompt with a strictly-identical shared prefix.

Note: unlike other 'blis convert' subcommands (which emit a WorkloadSpec to stdout),
'convert otel' writes TraceV2 files, because it targets 'blis replay' (fixed per-call
token counts) rather than 'blis run' (distribution sampling).`,
	Run: func(cmd *cobra.Command, args []string) {
		if otelContextGrowth != "accumulate" && otelContextGrowth != "independent" {
			logrus.Fatalf("--context-growth must be \"accumulate\" or \"independent\", got %q", otelContextGrowth)
		}
		if otelMinRounds < 1 {
			logrus.Fatalf("--min-rounds must be >= 1, got %d", otelMinRounds)
		}
		if otelMaxThinkTime < 0 {
			logrus.Fatalf("--max-think-time must be >= 0, got %s", otelMaxThinkTime)
		}
		opts := workload.OTelConvertOptions{
			ContextGrowth:  otelContextGrowth,
			MaxThinkTimeUs: otelMaxThinkTime.Microseconds(),
			IncludeErrors:  otelIncludeErrors,
			MinRounds:      otelMinRounds,

			ThinkFromArrivalGaps: otelThinkFromArrivalGaps,
		}
		if err := runConvertOtel(otelInputPath, otelTraceOutput, opts); err != nil {
			logrus.Fatalf("OTel conversion failed: %v", err)
		}
	},
}

// traceInput pairs a raw trace payload with a human-readable source label, so
// skip/parse-error diagnostics can name the offending file (and line, for
// .jsonl inputs) instead of failing silently.
type traceInput struct {
	name string
	raw  []byte
}

// collectTraceInputs returns each trace found at path, paired with a source
// label. A directory yields one entry per *.json file (name = filename); a
// *.jsonl file yields one entry per non-empty line (name = "<path>:line<N>");
// any other file is treated as a single trace (name = path).
func collectTraceInputs(path string) ([]traceInput, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, err
		}
		var names []string
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
				names = append(names, e.Name())
			}
		}
		sort.Strings(names) // deterministic corpus order (INV-6)
		var out []traceInput
		for _, n := range names {
			b, err := os.ReadFile(filepath.Join(path, n))
			if err != nil {
				return nil, err
			}
			out = append(out, traceInput{name: n, raw: b})
		}
		return out, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if strings.HasSuffix(path, ".jsonl") {
		var out []traceInput
		for i, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				out = append(out, traceInput{name: fmt.Sprintf("%s:line%d", path, i+1), raw: []byte(line)})
			}
		}
		return out, nil
	}
	return []traceInput{{name: path, raw: b}}, nil
}

// sharePrefix gives the sessions that opened with the same system message a shared prefix group on
// round 0 (#2130), so a replay finds that message cached for every session after the first, as the
// recorded server did. A span states its input tokens for the whole call, not per message, so the
// prefix length is an estimate: the system message's share of the call's characters times the
// call's tokens, the least over the group. It is a named substitution, logged.
func sharePrefix(sessions []*workload.OTelSession) {
	groups := map[string][]*workload.OTelSession{}
	for _, s := range sessions {
		if s.SystemText != "" && s.SystemShare > 0 {
			groups[s.SystemText] = append(groups[s.SystemText], s)
		}
	}
	keys := make([]string, 0, len(groups))
	for k, g := range groups {
		if len(g) > 1 {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	for _, k := range keys {
		g := groups[k]
		length := -1
		for _, s := range g {
			n := int(s.SystemShare * float64(s.Records[0].InputTokens))
			if length < 0 || n < length {
				length = n
			}
		}
		if length < 1 {
			continue
		}
		sum := sha256.Sum256([]byte(k))
		name := fmt.Sprintf("system-%x", sum[:4])
		for _, s := range g {
			r := &s.Records[0]
			r.PrefixGroup, r.PrefixLength = name, length
			r.InputTokens -= length
		}
		logrus.Infof("substitution: %d sessions share a system message; prefix group %s is estimated at %d tokens (the message's share of the first call's characters times its tokens)", len(g), name, length)
	}
}

// runConvertOtel reads all traces at inputPath, converts each to session
// records, assigns global request IDs, and writes a TraceV2 pair to
// <outPrefix>.yaml / <outPrefix>.csv.
func runConvertOtel(inputPath, outPrefix string, opts workload.OTelConvertOptions) error {
	// Return errors (do not Fatalf): the cobra Run wrapper owns the CLI exit
	// boundary and Fatalf's on a returned error (R6 — library-style helpers
	// surface errors, the CLI layer terminates). --input/--trace-output are also
	// MarkFlagRequired'd, so these guards are defense-in-depth.
	if inputPath == "" {
		return fmt.Errorf("--input is required")
	}
	if outPrefix == "" {
		return fmt.Errorf("--trace-output is required")
	}
	inputs, err := collectTraceInputs(inputPath)
	if err != nil {
		return err
	}

	var sessions []*workload.OTelSession
	skipped := 0
	for _, in := range inputs {
		sess, err := workload.ConvertOTelSession(in.raw, opts)
		if err != nil {
			logrus.Warnf("skipping unparseable trace %s: %v", in.name, err)
			skipped++
			continue
		}
		if sess == nil {
			logrus.Debugf("skipping trace %s: below --min-rounds", in.name)
			skipped++
			continue // below MinRounds
		}
		sessions = append(sessions, sess)
	}
	// Sessions are written in the order the client started them (#2130), not the order the
	// traces are stored in, so a pool replays them in the run's order. Ties keep input order.
	sort.SliceStable(sessions, func(i, j int) bool { return sessions[i].StartUs < sessions[j].StartUs })
	sharePrefix(sessions)
	var allRecords []workload.TraceRecord
	nextID := 0
	for _, sess := range sessions {
		recs := sess.Records
		for i := range recs {
			recs[i].RequestID = nextID
			nextID++
		}
		allRecords = append(allRecords, recs...)
	}

	if len(allRecords) == 0 {
		return fmt.Errorf("no usable sessions found in %q (skipped %d)", inputPath, skipped)
	}

	growth := opts.ContextGrowth
	if growth == "independent" {
		growth = "" // empty header value = per-round-independent inputs
	}
	header := &workload.TraceHeader{
		Version:              workload.TraceVersion,
		TimeUnit:             "microseconds",
		Mode:                 workload.TraceModeConverted,
		SessionContextGrowth: trace.ContextGrowth(growth),
	}
	if err := workload.ExportTraceV2(header, allRecords, outPrefix+".yaml", outPrefix+".csv"); err != nil {
		return err
	}
	logrus.Infof("Wrote %d records from %d sessions to %s.{yaml,csv} (skipped %d)",
		len(allRecords), len(inputs)-skipped, outPrefix, skipped)
	return nil
}

func init() {
	convertOtelCmd.Flags().StringVar(&otelInputPath, "input", "", "Path to OTel trace JSON file, directory of *.json, or *.jsonl (required)")
	convertOtelCmd.Flags().StringVar(&otelTraceOutput, "trace-output", "", "Output TraceV2 prefix; writes <prefix>.yaml + <prefix>.csv (required)")
	convertOtelCmd.Flags().StringVar(&otelContextGrowth, "context-growth", "accumulate", "Prefix model: \"accumulate\" (strict growing shared prefix) or \"independent\"")
	convertOtelCmd.Flags().DurationVar(&otelMaxThinkTime, "max-think-time", 15*time.Second, "Cap on per-round inter-call gap (think time); 0 = no cap")
	convertOtelCmd.Flags().BoolVar(&otelIncludeErrors, "include-errors", false, "Include spans with error status (status.code == 2)")
	convertOtelCmd.Flags().BoolVar(&otelThinkFromArrivalGaps, "think-from-arrival-gaps", false, "State no think time, so a replay derives it from arrival gaps (reproduces the output of versions before #2130; by default the think time is the gap from the previous span's end to this span's start)")
	convertOtelCmd.Flags().IntVar(&otelMinRounds, "min-rounds", 1, "Skip sessions with fewer than N usable LLM calls")
	_ = convertOtelCmd.MarkFlagRequired("input")
	_ = convertOtelCmd.MarkFlagRequired("trace-output")

	convertCmd.AddCommand(convertOtelCmd)
}

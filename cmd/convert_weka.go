package cmd

import (
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/workload"
)

var (
	wekaInputPath     string
	wekaTraceOutput   string
	wekaContextGrowth string
	wekaMaxThinkTime  time.Duration
	wekaMinRounds     int
)

var convertWekaCmd = &cobra.Command{
	Use:   "weka",
	Short: "Convert SemiAnalysis WekaTrace agentic JSONL to a TraceV2 file for closed-loop replay",
	Long: `Convert SemiAnalysis WekaTrace agentic JSONL (one proxy session per line) into a
TraceV2 pair (<prefix>.yaml + <prefix>.csv) suitable for 'blis replay --session-mode closed-loop'
(or 'blis replay --concurrent-sessions', which auto-promotes to closed-loop).

Input is a .jsonl file (one session per line) — the native Weka shape. Each session's
requests[] is filtered to the linear main-agent stream — type:"subagent" groups are
skipped (deferred to PR-E) — and each main turn becomes one round. Per-round input token counts are stored as deltas so accumulate replay
reconstructs the exact growing prompt with a strictly-identical shared prefix. Per-round
pure client think time is recomputed as max(0, t_i − t_{i-1} − api_time_{i-1}) between
consecutive main turns and carried in the think_time_us column.

Caveats: Weka input token counts are huge (p50 ≈ 110K, p90 ≈ 395K); raise --max-model-len
(the ~41K default drops every request as unservable) and scale --total-kv-blocks at replay.
The recorded model name (claude-*) is dropped so requests inherit --model (routing safety).

Note: unlike other 'blis convert' subcommands (which emit a WorkloadSpec to stdout),
'convert weka' writes TraceV2 files, because it targets 'blis replay' (fixed per-call
token counts) rather than 'blis run' (distribution sampling).`,
	Run: func(cmd *cobra.Command, args []string) {
		if wekaContextGrowth != "accumulate" && wekaContextGrowth != "independent" {
			logrus.Fatalf("--context-growth must be \"accumulate\" or \"independent\", got %q", wekaContextGrowth)
		}
		if wekaMinRounds < 1 {
			logrus.Fatalf("--min-rounds must be >= 1, got %d", wekaMinRounds)
		}
		if wekaMaxThinkTime < 0 {
			logrus.Fatalf("--max-think-time must be >= 0, got %s", wekaMaxThinkTime)
		}
		opts := workload.WekaConvertOptions{
			ContextGrowth:  wekaContextGrowth,
			MaxThinkTimeUs: wekaMaxThinkTime.Microseconds(),
			MinRounds:      wekaMinRounds,
		}
		if err := runConvertWeka(wekaInputPath, wekaTraceOutput, opts); err != nil {
			logrus.Fatalf("Weka conversion failed: %v", err)
		}
	},
}

// runConvertWeka reads all Weka sessions at inputPath, converts each to session
// records, assigns global request IDs, and writes a TraceV2 pair to
// <outPrefix>.yaml / <outPrefix>.csv. Reuses collectTraceInputs (defined in
// convert_otel.go — format-agnostic: it splits .jsonl by line, reads *.json in a
// directory, or treats any other file as a single payload).
func runConvertWeka(inputPath, outPrefix string, opts workload.WekaConvertOptions) error {
	// Return errors (do not Fatalf): the cobra Run wrapper owns the CLI exit
	// boundary (R6 — library-style helpers surface errors, the CLI layer
	// terminates). --input/--trace-output are also MarkFlagRequired'd, so these
	// guards are defense-in-depth.
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

	var allRecords []workload.TraceRecord
	nextID := 0
	skipped := 0
	for _, in := range inputs {
		recs, err := workload.ConvertWekaSession(in.raw, opts)
		if err != nil {
			logrus.Warnf("skipping unparseable weka session %s: %v", in.name, err)
			skipped++
			continue
		}
		if recs == nil {
			logrus.Debugf("skipping weka session %s: below --min-rounds", in.name)
			skipped++
			continue // below MinRounds
		}
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
		Version:              3,
		TimeUnit:             "microseconds",
		Mode:                 "generated",
		SessionContextGrowth: growth,
	}
	if err := workload.ExportTraceV2(header, allRecords, outPrefix+".yaml", outPrefix+".csv"); err != nil {
		return err
	}
	logrus.Infof("Wrote %d records from %d sessions to %s.{yaml,csv} (skipped %d)",
		len(allRecords), len(inputs)-skipped, outPrefix, skipped)
	return nil
}

func init() {
	convertWekaCmd.Flags().StringVar(&wekaInputPath, "input", "", "Path to WekaTrace JSONL file (one session per line), directory of *.json, or single JSON session (required)")
	convertWekaCmd.Flags().StringVar(&wekaTraceOutput, "trace-output", "", "Output TraceV2 prefix; writes <prefix>.yaml + <prefix>.csv (required)")
	convertWekaCmd.Flags().StringVar(&wekaContextGrowth, "context-growth", "accumulate", "Prefix model: \"accumulate\" (strict growing shared prefix) or \"independent\"")
	convertWekaCmd.Flags().DurationVar(&wekaMaxThinkTime, "max-think-time", 0, "Cap on the recomputed per-round think gap; 0 = no cap (Weka gaps are genuine away-from-keyboard times)")
	convertWekaCmd.Flags().IntVar(&wekaMinRounds, "min-rounds", 1, "Skip sessions with fewer than N usable main-agent turns")
	_ = convertWekaCmd.MarkFlagRequired("input")
	_ = convertWekaCmd.MarkFlagRequired("trace-output")

	convertCmd.AddCommand(convertWekaCmd)
}

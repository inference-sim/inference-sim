package cmd

import (
	"fmt"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/workload"
)

var (
	convertTraceHeader string
	convertTraceData   string
	convertTraceOutput string
)

var convertTraceCmd = &cobra.Command{
	Use:   "trace",
	Short: "Convert a trace in format version 3 (or 2) to version 4, the format blis replay reads",
	Long: `Convert a trace written by v0.15.0 or earlier (trace format version 3 or 2) into trace
format version 4, the format of blis-substrate's contracts/spec/trace, which blis replay reads
and blis run, blis replay and blis convert write.

Version 4 is version 3 without what only the removed 'blis observe' command wrote or read.
These are dropped:
  header:  created_at, warm_up_requests, workload_spec, server, network, observed_kv_metrics
  columns: vllm_priority, x_request_id, server_input_tokens, send_time_us, num_chunks,
           error_message, finish_reason
A header with mode "real" (a capture by blis observe) becomes mode "converted". Every other
field keeps its name and value.

Each dropped field that carried a value is reported, with its row count. Two of them changed
what v0.15.0's replay simulated, and a warning names them when they did: server_input_tokens
was the prompt length of a request with no prefix group, and send_time_us, where it differed
from arrival_time_us, was the injection time. Version 4 replays every request with
input_tokens at arrival_time_us.

Example:
  blis convert trace --header old.yaml --data old.csv --trace-output new
  # writes new.yaml and new.csv`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := runConvertTrace(convertTraceHeader, convertTraceData, convertTraceOutput); err != nil {
			logrus.Fatalf("trace conversion failed: %v", err)
		}
	},
}

// runConvertTrace converts the version 3 trace at headerPath and dataPath and writes it as
// version 4 to <outPrefix>.yaml and <outPrefix>.csv, reporting each dropped field.
func runConvertTrace(headerPath, dataPath, outPrefix string) error {
	if outPrefix == "" {
		return fmt.Errorf("--trace-output is required")
	}
	tr, drops, err := workload.ConvertTraceV3(headerPath, dataPath)
	if err != nil {
		return err
	}
	for _, d := range drops {
		if d.ChangesReplay {
			logrus.Warnf("dropped %s (%d rows): v0.15.0's replay read it, so this trace replays differently", d.Field, d.Rows)
			continue
		}
		logrus.Infof("dropped %s (%d rows)", d.Field, d.Rows)
	}
	if err := workload.ExportTraceV2(&tr.Header, tr.Records, outPrefix+".yaml", outPrefix+".csv"); err != nil {
		return err
	}
	logrus.Infof("Wrote %d records to %s.{yaml,csv} (trace format version %d)", len(tr.Records), outPrefix, workload.TraceVersion)
	return nil
}

func init() {
	convertTraceCmd.Flags().StringVar(&convertTraceHeader, "header", "", "The version 3 trace header (YAML) (required)")
	convertTraceCmd.Flags().StringVar(&convertTraceData, "data", "", "The version 3 trace data (CSV) (required)")
	convertTraceCmd.Flags().StringVar(&convertTraceOutput, "trace-output", "", "Output prefix; writes <prefix>.yaml and <prefix>.csv (required)")
	for _, f := range []string{"header", "data", "trace-output"} {
		_ = convertTraceCmd.MarkFlagRequired(f)
	}
	convertCmd.AddCommand(convertTraceCmd)
}

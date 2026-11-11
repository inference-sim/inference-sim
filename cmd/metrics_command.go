package cmd

import (
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-substrate/contracts/metrics"
)

// `blis metrics` prints the metric registry of the blis-substrate release this build pins
// (design 8.7, DEV-8): each metric's identifier, formula, statistic, population, window and
// unit, the scopes it is defined at and the aggregates it admits. The dashboard and
// configuration search read metric definitions here and keep no table of metric names. The
// command holds no definition of its own: it prints metrics.All().
//
// With --json the output is {"metrics": [...]}, each record with the registry's own keys. Without
// it, one line per metric: identifier, statistic, unit, direction, scopes, aggregates.

var metricsCmd = &cobra.Command{
	Use:   "metrics",
	Short: "Print the metric registry with each metric's scope and aggregate",
	Long: "Print every metric of the registry embedded in blis-substrate's contracts/metrics at the " +
		"version this build pins (stack.json names it): identifier, formula, statistic, population, " +
		"window, unit, which direction is better, the scopes it is defined at (cluster, pool, instance) and the aggregates a " +
		"figure of it may take over narrower scopes (max, min). A figure is compared with " +
		"another only when its metric, scope and aggregate are all the same (DEV-8). With --json " +
		"the output is {\"metrics\": [...]}, each record with the registry's own keys.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		return printMetrics(cmd.OutOrStdout(), metrics.All(), substrateJSON)
	},
}

// printMetrics writes defs, as JSON or as one line each.
func printMetrics(w io.Writer, defs []metrics.Definition, asJSON bool) error {
	if asJSON {
		return writeJSON(w, map[string]any{"metrics": defs})
	}
	for _, d := range defs {
		if _, err := fmt.Fprintf(w, "%-48s %-12s %-14s better=%-6s scopes=%s aggregates=%s\n", d.ID, d.Statistic, d.Unit,
			d.Better, joinNames(d.Scopes), joinNames(d.Aggregates)); err != nil {
			return err
		}
	}
	return nil
}

// joinNames renders a list of scopes or aggregates as a comma-separated list, "-" when empty.
func joinNames[T ~string](l []T) string {
	if len(l) == 0 {
		return "-"
	}
	s := make([]string, len(l))
	for i, x := range l {
		s[i] = string(x)
	}
	return strings.Join(s, ",")
}

func init() {
	metricsCmd.Flags().BoolVar(&substrateJSON, "json", false, "print JSON")
	rootCmd.AddCommand(metricsCmd)
}

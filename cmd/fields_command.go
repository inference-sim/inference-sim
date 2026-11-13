package cmd

import (
	"fmt"
	"path/filepath"

	"github.com/spf13/cobra"
)

// `blis fields` prints the field list of the engine the build registers, as the engine reports
// it through engine.Engine.Fields (design section 7, X19, SUB-1). A script or generator that
// needs an engine's default reads it here and never writes one of its own, and the accuracy
// dashboard and configuration search read the list through this command and keep no copy.
//
// The output is the list in the substrate's canonical form (fields.FieldList.Marshal): one
// document, entries sorted by name and source, the engine kind and upstream release in its
// "engine" member, and "kind": "FieldList" naming the document type. The list the substrate
// defines carries no further version number, so none is invented here. The command holds no
// field and reads no file of the engine's.

var (
	fieldsScenario    string
	fieldsScenarioDir string
)

var fieldsCmd = &cobra.Command{
	Use:   "fields --scenario <file> --scenarios <dir>",
	Short: "Print the field list of the engine a scenario names, as the engine reports it",
	Long: "Build the engine the scenario names, over the catalog and registry in use, and print the " +
		"field list it reports through engine.Engine.Fields: every configuration field and " +
		"environment variable of the upstream release it models, each with its status and the " +
		"estimation rule a scoring harness applies. The list is the engine's own and the same for " +
		"every scenario of that engine and release; a scenario is the input because the engine " +
		"contract gives a list only to an engine that has been built. Output is one JSON " +
		"document in canonical form, sorted, and byte-for-byte what the engine reports.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		if fieldsScenario == "" || fieldsScenarioDir == "" {
			return fmt.Errorf("fields: --scenario and --scenarios are required: the engine is built from a scenario")
		}
		data, err := runData()
		if err != nil {
			return err
		}
		l, err := engineFieldList(filepath.Join(fieldsScenarioDir, fieldsScenario), data, 0)
		if err != nil {
			return err
		}
		b, err := l.Marshal()
		if err != nil {
			return fmt.Errorf("the engine's field list: %w", err)
		}
		_, err = cmd.OutOrStdout().Write(b)
		return err
	},
}

func init() {
	fieldsCmd.Flags().StringVar(&fieldsScenario, "scenario", "", "Scenario FILE NAME within --scenarios (REQUIRED): names the engine whose list is printed")
	fieldsCmd.Flags().StringVar(&fieldsScenarioDir, "scenarios", "", "Directory holding scenario YAML files (REQUIRED)")
	registerDataFlags(fieldsCmd)
	rootCmd.AddCommand(fieldsCmd)
}

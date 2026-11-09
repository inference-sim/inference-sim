package cmd

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/schema"
)

// The commands the applications read the document schemas through (design §5, SUB-19):
// `blis schema <document>` prints the JSON Schema this build's substrate release publishes,
// and `blis schema validate <document> <file>` runs the Go validators on a file. Neither
// holds a schema or a rule of its own.

// errDocumentInvalid marks a document the validators refuse. The command exits 1 for it,
// and 2 for any other failure, so a caller can tell a refused document from a broken run.
var errDocumentInvalid = errors.New("document refused")

// exitCode is an error that carries the process exit status Execute returns.
type exitCode int

func (e exitCode) Error() string { return fmt.Sprintf("exit status %d", int(e)) }

// documentExitCode is 1 for a refused document and 2 for any other failure.
func documentExitCode(err error) int {
	if errors.Is(err, errDocumentInvalid) {
		return 1
	}
	return 2
}

var schemaCmd = &cobra.Command{
	Use:   "schema <document>",
	Short: "Print the JSON Schema of a BLIS document",
	Long: "Print the JSON Schema of a document that crosses a repository boundary, as the " +
		"blis-substrate release this build pins publishes it. Documents: " +
		strings.Join(schema.Names(), ", ") + ".",
	Args:         cobra.ExactArgs(1),
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		b, err := schema.Published(args[0])
		if err != nil {
			return err
		}
		_, err = cmd.OutOrStdout().Write(b)
		return err
	},
}

var schemaValidateCmd = &cobra.Command{
	Use:   "validate <document> <file>",
	Short: "Validate a file as a BLIS document with the Go validators",
	Long: "Read <file> (JSON or YAML, strictly: a key no field declares is an error) as a " +
		"<document> and run the validators of the blis-substrate release this build pins. Exit " +
		"0 when valid, 1 when the document is refused (each finding is printed to standard " +
		"error), 2 when the command could not run. Documents: " + strings.Join(validatable(), ", ") + ".",
	Args:         cobra.ExactArgs(2),
	SilenceUsage: true,
	// The command prints its own findings; cobra would add a line naming the exit status.
	SilenceErrors: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		err := validateDocument(args[0], args[1])
		if err == nil {
			return nil
		}
		// A failure to write the finding must not hide the exit status.
		_, _ = fmt.Fprintln(cmd.ErrOrStderr(), err)
		return exitCode(documentExitCode(err))
	},
}

// validators maps a document name to a function that reads the file with the substrate's own
// loader, which refuses an unknown key, and runs the document's Validate. A scenario file may
// hold the scenario alone, or the scenario and the deployment applied to it (a bundle).
var validators = map[string]func(path string) (findings []string, err error){
	"scenario": func(path string) ([]string, error) {
		if sc, dep, err := contracts.LoadBundle(path); err == nil {
			return append(problemLines(sc.Validate().Errors()), problemLines(dep.Validate().Errors())...), nil
		}
		sc, err := contracts.LoadScenario(path)
		if err != nil {
			return nil, err
		}
		return problemLines(sc.Validate().Errors()), nil
	},
	"deployment": func(path string) ([]string, error) {
		d, err := contracts.LoadDeployment(path)
		if err != nil {
			return nil, err
		}
		return problemLines(d.Validate().Errors()), nil
	},
	"result": func(path string) ([]string, error) {
		r, err := contracts.LoadResult(path)
		if err != nil {
			return nil, err
		}
		return problemLines(r.Validate().Errors()), nil
	},
	"point-record": func(path string) ([]string, error) {
		r, err := contracts.LoadPointRecord(path)
		if err != nil {
			return nil, err
		}
		return problemLines(r.Validate().Errors()), nil
	},
	"workload": func(path string) ([]string, error) {
		w, err := contracts.LoadWorkloadSpec(path)
		if err != nil {
			return nil, err
		}
		return problemLines(w.Validate().Errors()), nil
	},
}

func validatable() []string {
	names := make([]string, 0, len(validators))
	for n := range validators {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// problemLines renders the validators' findings, one per line, in their own words.
func problemLines[T fmt.Stringer](ps []T) []string {
	out := make([]string, 0, len(ps))
	for _, p := range ps {
		out = append(out, p.String())
	}
	return out
}

// validateDocument returns nil when the file is a valid document, an error wrapping
// errDocumentInvalid when a validator or the strict decoder refuses it, and any other error
// when the document name or the file is unusable.
func validateDocument(document, path string) error {
	v, ok := validators[document]
	if !ok {
		return fmt.Errorf("no validator for document %q; documents are %s",
			document, strings.Join(validatable(), ", "))
	}
	if _, err := os.Stat(path); err != nil {
		return err
	}
	findings, err := v(path)
	if err != nil {
		return fmt.Errorf("%w: %s: %v", errDocumentInvalid, document, err)
	}
	if len(findings) > 0 {
		return fmt.Errorf("%w: %s:\n%s", errDocumentInvalid, document, strings.Join(findings, "\n"))
	}
	return nil
}

func init() {
	// A schema is JSON; the flag is accepted so a caller can pass --json to every command alike.
	schemaCmd.Flags().Bool("json", true, "accepted for uniformity with the other commands; a schema is always JSON")
	schemaCmd.AddCommand(schemaValidateCmd)
	rootCmd.AddCommand(schemaCmd)
}

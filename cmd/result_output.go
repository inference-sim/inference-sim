package cmd

// The Result document (#1871): what `blis run` and `blis replay` print. stdout is exactly one
// blis-schemas Result (spec/result), JSON by default and YAML with --format yaml; --output
// writes the same bytes to a file. --view summary renders the plain-text sections the
// commands used to print, from the document, to stderr. Design of record: blis-schemas#41,
// sections A and D.
//
// Every number here is read from the run's metrics; nothing is computed twice. The document
// is validated with blis-schemas' own validator before it is printed, and a document that
// fails is a logrus.Fatalf: a Result that does not validate is not a Result.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/result"
)

// The --format, --output and --view values, shared by run and replay.
var (
	resultFormat string // --format: json (default) or yaml
	resultOutput string // --output: a file that receives the same bytes as stdout
	resultView   string // --view: "" or "summary" (today's text sections, to stderr)
)

// viewSummary is the one --view value: today's plain-text sections, rendered from the Result.
const viewSummary = "summary"

// registerResultFlags adds the Result document's output flags to cmd.
func registerResultFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&resultFormat, "format", string(result.FormatJSON),
		"Encoding of the Result document on stdout and in --output: json or yaml")
	cmd.Flags().StringVar(&resultOutput, "output", "",
		"File to write the Result document to: the same bytes stdout carries")
	cmd.Flags().StringVar(&resultView, "view", "",
		"Also render the Result for a reader, to stderr: summary (the text sections run and replay printed before the Result document)")
}

// checkResultFlags refuses a --format or --view value before the run, so a typo costs no
// simulation.
func checkResultFlags() {
	switch result.Format(resultFormat) {
	case result.FormatJSON, result.FormatYAML:
	default:
		logrus.Fatalf("--format %q: the Result document is written as %s or %s", resultFormat, result.FormatJSON, result.FormatYAML)
	}
	switch resultView {
	case "", viewSummary:
	default:
		logrus.Fatalf("--view %q: the one view is %q", resultView, viewSummary)
	}
}

// warnDeprecatedFile logs that a results file is deprecated in favor of --output (#1871).
// The file keeps its content for one release.
func warnDeprecatedFile(flag, path string) {
	if path == "" {
		return
	}
	logrus.Warnf("%s is deprecated and will be removed in the next release: its file keeps its "+
		"content for this release, and the Result document on stdout (or --output) carries the same "+
		"figures under blis-schemas' names", flag)
}

// emitResult validates r, writes it to stdout and to --output in the selected format, and,
// with --view summary, renders the text summary to stderr. A document the validator refuses
// is fatal: it names every problem.
func emitResult(r *result.Result, view *summaryView) {
	if err := writeResult(os.Stdout, r); err != nil {
		logrus.Fatalf("%v", err)
	}
	if resultView == viewSummary {
		renderSummary(os.Stderr, r, view)
	}
}

// writeResult validates and encodes r, writes it to --output when set, then to stdout. It writes
// nothing at all for a document that does not validate, and nothing to stdout when --output
// cannot be written.
func writeResult(stdout io.Writer, r *result.Result) error {
	data, err := encodeResult(r, result.Format(resultFormat))
	if err != nil {
		return fmt.Errorf("the Result document: %w", err)
	}
	// --output is written first: a run that cannot write its file fails before stdout carries a
	// complete document a consumer could take for a successful run.
	if resultOutput != "" {
		if err := os.WriteFile(resultOutput, data, 0o644); err != nil {
			return fmt.Errorf("--output %s: %w", resultOutput, err)
		}
		logrus.Infof("Result written to: %s", resultOutput)
	}
	if _, err := stdout.Write(data); err != nil {
		return fmt.Errorf("writing the Result document to stdout: %w", err)
	}
	return nil
}

// encodeResult validates r and encodes it. It is emitResult without the writes, for tests.
func encodeResult(r *result.Result, f result.Format) ([]byte, error) {
	if err := validateResult(r); err != nil {
		return nil, err
	}
	return result.Marshal(r, f)
}

// problemWarning is blis-schemas' warning severity (its validate package is internal, so the
// value is named here): usable, but a reader should look.
const problemWarning = "warning"

// validateResult runs blis-schemas' validator over the document. Field problems are errors;
// a warning (none is expected) is logged.
func validateResult(r *result.Result) error {
	rep := blisschemas.Validate(blisschemas.Bundle{Result: r})
	for _, p := range rep.Problems() {
		if string(p.Severity) == problemWarning {
			logrus.Warnf("Result document: %s", p)
		}
	}
	if !rep.OK() {
		var b strings.Builder
		for _, p := range rep.Problems() {
			if string(p.Severity) != problemWarning {
				fmt.Fprintf(&b, "\n  %s", p)
			}
		}
		return fmt.Errorf("the emitted Result does not validate against blis-schemas:%s", b.String())
	}
	return nil
}

// fileDigest is the sha256 of a file the run wrote, for an artifact reference.
func fileDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// bytesDigest is the sha256 of data.
func bytesDigest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

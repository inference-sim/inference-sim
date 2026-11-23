package tools

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// SUB-12 says every loader of a BLIS document refuses the malformed set. The contract loaders
// are checked by contracts.TestEveryLoaderRefusesTheMalformedSet; the committed files that the
// ledger and the tools read (ledger.yaml, sources.yaml, the deferrals, the quarantine list,
// the gates, stack.lock) are read through contracts/check.DecodeStrict, which
// check.TestDecodeStrictRefusesTheMalformedSet holds to the set. This test keeps it that way:
// a non-test Go file under ledger/ or tools/ that builds its own yaml.Decoder or calls
// yaml.Unmarshal on a document would bypass the shared rules, so it fails here unless it is
// one of the listed exceptions, each with its reason.
func TestNoYAMLDocumentIsDecodedOutsideTheSharedStrictDecoder(t *testing.T) {
	exceptions := map[string]string{
		// A GitHub Actions workflow is not a BLIS document; the lint reads its structure to
		// check its jobs, with a lenient parse on purpose.
		"prcheck/workflows.go": "reads GitHub workflow files, not a BLIS document",
		"prcheck/pathgates.go": "reads GitHub workflow files, not a BLIS document",
		"prcheck/status.go":    "reads GitHub workflow files for their path triggers, not a BLIS document",
		"workflows.go":         "the ledger's lint of GitHub workflow files, not a BLIS document",
		// The PR-check request record is built from the API, not read from a repository file.
		"prcheck/main.go": "decodes the check-run request the workflow builds",
		// The linter of this very rule.
		"strict_decoders_test.go": "this test",
	}
	pat := regexp.MustCompile(`yaml\.NewDecoder|yaml\.Unmarshal\(`)
	var offenders []string
	for _, root := range []string{".", "../ledger"} {
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") && !strings.HasSuffix(path, "strict_decoders_test.go") {
				return err
			}
			rel := strings.TrimPrefix(filepath.ToSlash(path), "./")
			rel = strings.TrimPrefix(rel, "../")
			src, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if !pat.Match(src) {
				return nil
			}
			key := strings.TrimPrefix(rel, "ledger/")
			if _, ok := exceptions[rel]; ok {
				return nil
			}
			if _, ok := exceptions[key]; ok {
				return nil
			}
			offenders = append(offenders, rel)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(offenders) > 0 {
		t.Errorf("these files decode YAML without contracts/check.DecodeStrict: %v", offenders)
	}
}

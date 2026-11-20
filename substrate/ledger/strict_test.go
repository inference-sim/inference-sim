package ledger

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// SUB-12: the ledger's committed files are read through the contracts' strict decoder, so a
// repeated key, a non-finite number or a recursive alias in one is refused, not read.
func TestTheLedgersFilesAreReadStrictly(t *testing.T) {
	good, err := os.ReadFile("../catalog/ledger/ledger.yaml")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	write := func(body string) {
		if err := os.WriteFile(filepath.Join(dir, FileLedger), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(string(good))
	if _, err := LoadConfig(dir); err != nil {
		t.Fatalf("the committed ledger.yaml is refused: %v", err)
	}
	for name, body := range map[string]string{
		"a repeated key":      "kind: LedgerConfig\n" + string(good),
		"a recursive alias":   "alias: &a [*a]\n" + string(good),
		"a non-finite number": string(good) + "\nnan_field: .nan\n",
		"a second document":   string(good) + "\n---\nkind: Other\n",
		"an unknown key":      string(good) + "\nbogus_key: 1\n",
	} {
		write(body)
		if _, err := LoadConfig(dir); err == nil {
			t.Errorf("%s in ledger.yaml was accepted", name)
		} else if !strings.Contains(err.Error(), FileLedger) {
			t.Errorf("%s: the refusal does not name the file: %v", name, err)
		}
	}
}

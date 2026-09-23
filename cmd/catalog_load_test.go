package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// This file is the R1/C6 acceptance GATE (#1750): CI runs a load over every catalog entry,
// through the same code paths `blis run` uses, and fails on any rejection. There is no
// validate command by design — the gate is these tests.
//
// Three layers:
//
//  1. TestCatalogStrictLoad_CommittedFixtureCatalog — unconditional, over the committed
//     testdata/catalog. Proves every committed vendor config.json resolves and parses through
//     the RUN path, every committed preset loads through the production preset reader, and the
//     committed storage-device table loads — on real data, in every `go test ./cmd/...` run.
//  2. TestCatalogStrictLoad_CompleteCatalogLoadsClean and the contract tables — a fully
//     populated catalog (all four namespaces) loads with zero problems, and each strict rule
//     fires when violated.
//  3. TestCatalogStrictLoad_RealCatalog — the load over the AUTHORITATIVE catalog. It is
//     driven by scripts/catalog-load-gate.sh, which clones inference-sim/blis-catalog at a
//     pinned revision, points BLIS_CATALOG at it, runs this test and fails when it skips. This
//     test skips when the variable is unset, so a local `go test ./...` needs no catalog
//     checkout.

// fixtureCatalog is the committed clone-root-shaped test catalog.
const fixtureCatalog = "../testdata/catalog"

// validHardwareEntry is a well-formed hardware/<gpu>.yaml body (the H100 shape: every
// required calibration field, plus the optional interconnect pair and provenance comments).
const validHardwareEntry = `_comment: "fixture — mirrors the committed blis-catalog h100.yaml shape"
TFlopsPeak: 989.5
TFlopsFP8: 1979.0
BwPeakTBs: 3.35
mfuPrefill: 0.45
mfuDecode: 0.30
MemoryGiB: 80.0
_comment_interconnect: "fixture"
IntraNodeBwGBps: 450
InterNodeBwGBps: 50
`

// validWorkloadEntry is a well-formed workloads/<name>.yaml body.
const validWorkloadEntry = `prefix_tokens: 0
prompt_tokens: 256
prompt_tokens_stdev: 100
prompt_tokens_min: 2
prompt_tokens_max: 800
output_tokens: 256
output_tokens_stdev: 100
output_tokens_min: 1
output_tokens_max: 1024
`

// validDeviceTable is a well-formed devices/storage.yaml body.
const validDeviceTable = `nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}
cpu_dram: {read_bandwidth: 2.0e4, write_bandwidth: 2.0e4, base_latency: 1.0}
`

// gateModel is the fixture model every temp catalog in this file is built around. Its
// config.json is copied from the committed fixture catalog, so the model half of a temp
// catalog is real vendor data rather than a hand-written stub that could drift from what the
// parser accepts.
const gateModel = "qwen3-14b"

// modelEntryYAML renders a well-formed model.yaml body for the named model.
func modelEntryYAML(name string) string {
	return fmt.Sprintf(`name: %s
source:
  provider: huggingface
  repo: Example/%s
  revision: 0000000000000000000000000000000000000000
  retrieved: 2026-09-16
`, name, name)
}

// newCompleteCatalog builds a temporary catalog with all four namespaces populated and every
// rule satisfied. Tests mutate one file of the returned catalog to drive one contract.
func newCompleteCatalog(t *testing.T) string {
	t.Helper()
	root := t.TempDir()

	entryDir := filepath.Join(root, catalogModelsSubdir, gateModel)
	if err := os.MkdirAll(entryDir, 0o755); err != nil {
		t.Fatalf("mkdir model entry: %v", err)
	}
	config, err := os.ReadFile(filepath.Join(fixtureCatalog, catalogModelsSubdir, gateModel, hfConfigFile))
	if err != nil {
		t.Fatalf("read fixture config.json: %v", err)
	}
	writeCatalogFile(t, filepath.Join(entryDir, hfConfigFile), string(config))
	writeCatalogFile(t, filepath.Join(entryDir, catalogModelEntryFile), modelEntryYAML(gateModel))
	writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt), validHardwareEntry)
	writeCatalogFile(t, filepath.Join(root, catalogWorkloadsSubdir, "chatbot"+presetFileExt), validWorkloadEntry)
	writeCatalogFile(t, catalogStorageDevicesPath(root), validDeviceTable)
	return root
}

// writeCatalogFile writes one catalog file, creating its directory.
func writeCatalogFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// loadOrFatal runs the whole-catalog load and fails the test on a root-level error (an
// unusable root), which is distinct from the per-file problems a contract test asserts on.
func loadOrFatal(t *testing.T, root string) catalogLoadReport {
	t.Helper()
	report, err := loadCatalog(root)
	if err != nil {
		t.Fatalf("loadCatalog(%q) returned a root-level error: %v", root, err)
	}
	return report
}

// TestCatalogStrictLoad_CompleteCatalogLoadsClean is the positive control for every rule in
// this file: a catalog with all four namespaces populated and nothing wrong with it loads with
// zero problems, and reports what it loaded. Without this, a mutation test could pass because
// the loader rejects everything.
func TestCatalogStrictLoad_CompleteCatalogLoadsClean(t *testing.T) {
	report := loadOrFatal(t, newCompleteCatalog(t))
	if err := report.Err(); err != nil {
		t.Fatalf("a complete, well-formed catalog must load clean; got: %v", err)
	}
	for _, got := range []struct {
		namespace string
		count     int
	}{
		{"models", report.Models},
		{catalogHardwareSubdir, report.Hardware},
		{catalogWorkloadsSubdir, report.Presets},
		{catalogDevicesSubdir, report.DeviceClasses},
	} {
		if got.count == 0 {
			t.Errorf("%s: loaded 0 entries, so the gate would be vacuous for that namespace", got.namespace)
		}
	}
}

// TestCatalogStrictLoad_CommittedFixtureCatalog loads the committed testdata/catalog. It is
// the unconditional half of the gate, and asserts TWO things about real committed data:
//
//   - every vendor config.json resolves through resolveModelConfigInCatalog and parses through
//     latency.GetModelConfig (the RUN path), every committed preset loads through the
//     production preset reader, and the committed storage-device table loads;
//   - the ONLY problems the fixture catalog has are its missing model.yaml files.
//
// The fixture catalog deliberately carries no model.yaml and no hardware/ namespace: it exists
// so `go test ./cmd/...` can resolve models and presets offline, and three of its config.json
// files are older/trimmed copies of the authoritative ones (tracked by #1748), so giving them a
// provenance record stating an upstream revision would state something untrue. Asserting the
// incompleteness here — rather than leaving it unexplained — makes the completeness rule fire
// on real committed data, and pins that no OTHER rule false-positives on it.
func TestCatalogStrictLoad_CommittedFixtureCatalog(t *testing.T) {
	report := loadOrFatal(t, fixtureCatalog)

	if report.Presets == 0 {
		t.Error("the fixture catalog's committed workload presets must all load through the production reader")
	}
	if report.DeviceClasses == 0 {
		t.Error("the fixture catalog's committed storage-device table must load through the production reader")
	}

	modelDirs, err := os.ReadDir(filepath.Join(fixtureCatalog, catalogModelsSubdir))
	if err != nil {
		t.Fatalf("read fixture models: %v", err)
	}
	wantModelEntries := 0
	for _, e := range modelDirs {
		if e.IsDir() {
			wantModelEntries++
		}
	}
	if wantModelEntries == 0 {
		t.Fatal("the fixture catalog has no model entries, so this gate would be vacuous")
	}
	if len(report.Problems) != wantModelEntries {
		t.Fatalf("expected exactly one problem per fixture model entry (its missing %s); got %d problem(s) for %d entries:\n%v",
			catalogModelEntryFile, len(report.Problems), wantModelEntries, report.Err())
	}
	for _, problem := range report.Problems {
		if !strings.Contains(problem, catalogModelEntryFile) {
			t.Errorf("the only expected fixture-catalog problem is a missing %s, but got: %s",
				catalogModelEntryFile, problem)
		}
	}
}

// TestCatalogStrictLoad_RealCatalog is the gate against the authoritative catalog.
// scripts/catalog-load-gate.sh is what drives it: the script itself clones blis-catalog at the
// pinned revision it holds, points BLIS_CATALOG at the checkout, runs this test and fails if it
// skipped or never ran. Skipped when the variable is unset so a local `go test ./...` needs no
// checkout.
//
// Wiring that script into .github/workflows/ci.yml as a `catalog-load` job is a PENDING HUMAN
// EDIT — the delivery loop's token cannot push workflow files (docs/contributing/automated-delivery.md),
// so the job body is quoted verbatim in the script's header for a human to paste. Until it
// lands, this test runs on demand rather than on every PR; the unconditional coverage is
// TestCatalogStrictLoad_CommittedFixtureCatalog over testdata/catalog.
func TestCatalogStrictLoad_RealCatalog(t *testing.T) {
	root := os.Getenv(catalogEnvVar)
	if root == "" {
		t.Skipf("%s is not set; scripts/catalog-load-gate.sh sets it to a blis-catalog checkout", catalogEnvVar)
	}
	report, err := loadCatalog(root)
	if err != nil {
		t.Fatalf("catalog %q is unusable: %v", root, err)
	}
	if err := report.Err(); err != nil {
		t.Fatalf("catalog %q did not load clean: %v", root, err)
	}
	t.Logf("catalog %q loaded: %d model(s), %d hardware entr(ies), %d workload preset(s), %d device class(es)",
		root, report.Models, report.Hardware, report.Presets, report.DeviceClasses)
	// Non-vacuity: a catalog checkout that silently produced an empty tree (a wrong path, a
	// failed checkout) must not pass as "loaded clean".
	for _, got := range []struct {
		namespace string
		count     int
	}{
		{"models", report.Models},
		{catalogHardwareSubdir, report.Hardware},
		{catalogWorkloadsSubdir, report.Presets},
		{catalogDevicesSubdir, report.DeviceClasses},
	} {
		if got.count == 0 {
			t.Errorf("%s: the catalog at %q loaded 0 entries for this namespace", got.namespace, root)
		}
	}
}

// TestCatalogStrictLoad_StrictRules drives one contract per row: each mutation of an otherwise
// clean catalog must be REPORTED, and the report must name the offending file and what is
// wrong with it (C6: "fails with a message naming the file and the problem").
func TestCatalogStrictLoad_StrictRules(t *testing.T) {
	modelEntry := filepath.Join(catalogModelsSubdir, gateModel)
	tests := []struct {
		name string
		// mutate changes one file of a complete catalog rooted at root.
		mutate func(t *testing.T, root string)
		// wantFile is a path fragment the diagnostic must name.
		wantFile string
		// wantPhrases are fragments the diagnostic must contain (the offending key, or the
		// reason).
		wantPhrases []string
	}{
		{
			name: "unknown key in model.yaml",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, modelEntry, catalogModelEntryFile),
					modelEntryYAML(gateModel)+"licence: apache-2.0\n")
			},
			wantFile:    catalogModelEntryFile,
			wantPhrases: []string{"licence"},
		},
		{
			name: "unknown key in hardware entry",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt),
					validHardwareEntry+"MemoryGB: 80.0\n")
			},
			wantFile:    filepath.Join(catalogHardwareSubdir, "h100"+catalogYAMLExt),
			wantPhrases: []string{"MemoryGB", "unknown"},
		},
		{
			name: "case-mismatched key in hardware entry",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt),
					strings.Replace(validHardwareEntry, "IntraNodeBwGBps", "IntraNodeBwGbps", 1))
			},
			wantFile:    filepath.Join(catalogHardwareSubdir, "h100"+catalogYAMLExt),
			wantPhrases: []string{"IntraNodeBwGbps", "letter case"},
		},
		{
			name: "retired per-collective interconnect key in hardware entry",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt),
					validHardwareEntry+"InterNodeLatencyUs: 25.0\n")
			},
			wantFile:    filepath.Join(catalogHardwareSubdir, "h100"+catalogYAMLExt),
			wantPhrases: []string{"InterNodeLatencyUs", "InterNodeHopLatencyUs"},
		},
		{
			name: "hardware entry omits a required calibration field",
			mutate: func(t *testing.T, root string) {
				body := strings.Replace(validHardwareEntry, "MemoryGiB: 80.0\n", "", 1)
				writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt), body)
			},
			wantFile:    filepath.Join(catalogHardwareSubdir, "h100"+catalogYAMLExt),
			wantPhrases: []string{"MemoryGiB", "required"},
		},
		{
			name: "hardware entry states a non-positive calibration value",
			mutate: func(t *testing.T, root string) {
				body := strings.Replace(validHardwareEntry, "BwPeakTBs: 3.35", "BwPeakTBs: 0", 1)
				writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt), body)
			},
			wantFile:    filepath.Join(catalogHardwareSubdir, "h100"+catalogYAMLExt),
			wantPhrases: []string{"BwPeakTBs"},
		},
		{
			name: "hardware entry sets only one interconnect bandwidth",
			mutate: func(t *testing.T, root string) {
				body := strings.Replace(validHardwareEntry, "InterNodeBwGBps: 50\n", "", 1)
				writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt), body)
			},
			wantFile:    filepath.Join(catalogHardwareSubdir, "h100"+catalogYAMLExt),
			wantPhrases: []string{"InterNodeBwGBps"},
		},
		{
			name: "unknown key in workload preset",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, catalogWorkloadsSubdir, "chatbot"+presetFileExt),
					validWorkloadEntry+"prompt_tokens_median: 12\n")
			},
			wantFile:    filepath.Join(catalogWorkloadsSubdir, "chatbot"+presetFileExt),
			wantPhrases: []string{"prompt_tokens_median"},
		},
		{
			name: "unknown key in storage-device table",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, catalogStorageDevicesPath(root),
					validDeviceTable+"tape: {read_bandwidth: 1.0, write_bandwidth: 1.0, base_latency: 1.0, seek_latency: 9.0}\n")
			},
			wantFile:    catalogStorageDevicesRelPath,
			wantPhrases: []string{"seek_latency"},
		},
		{
			name: "model entry has no config.json",
			mutate: func(t *testing.T, root string) {
				if err := os.Remove(filepath.Join(root, modelEntry, hfConfigFile)); err != nil {
					t.Fatalf("remove config.json: %v", err)
				}
			},
			wantFile:    hfConfigFile,
			wantPhrases: []string{"not in the catalog"},
		},
		{
			name: "model entry has no model.yaml",
			mutate: func(t *testing.T, root string) {
				if err := os.Remove(filepath.Join(root, modelEntry, catalogModelEntryFile)); err != nil {
					t.Fatalf("remove model.yaml: %v", err)
				}
			},
			wantFile:    catalogModelEntryFile,
			wantPhrases: []string{"incomplete"},
		},
		{
			name: "model entry config.json is not a HuggingFace config",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, modelEntry, hfConfigFile), `{"model": "mine"}`)
			},
			wantFile:    hfConfigFile,
			wantPhrases: []string{"HuggingFace"},
		},
		{
			name: "model.yaml names a different model than its directory",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, modelEntry, catalogModelEntryFile),
					modelEntryYAML("some-other-model"))
			},
			wantFile:    catalogModelEntryFile,
			wantPhrases: []string{"some-other-model", gateModel},
		},
		{
			name: "model.yaml omits its source revision",
			mutate: func(t *testing.T, root string) {
				body := strings.Replace(modelEntryYAML(gateModel),
					"  revision: 0000000000000000000000000000000000000000\n", "", 1)
				writeCatalogFile(t, filepath.Join(root, modelEntry, catalogModelEntryFile), body)
			},
			wantFile:    catalogModelEntryFile,
			wantPhrases: []string{"source.revision", "required"},
		},
		{
			name: "model.yaml states a GPU",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, modelEntry, catalogModelEntryFile),
					modelEntryYAML(gateModel)+"gpu: H100\n")
			},
			wantFile:    catalogModelEntryFile,
			wantPhrases: []string{"deployment fact", "--hardware", `"gpu"`},
		},
		{
			name: "model.yaml states a nested tensor-parallel degree",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, modelEntry, catalogModelEntryFile),
					modelEntryYAML(gateModel)+"deployment:\n  tensor_parallel_size: 8\n")
			},
			wantFile:    catalogModelEntryFile,
			wantPhrases: []string{"deployment fact", "--tp", "deployment.tensor_parallel_size"},
		},
		{
			name: "workload preset states a tensor-parallel degree",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, catalogWorkloadsSubdir, "chatbot"+presetFileExt),
					validWorkloadEntry+"tp: 4\n")
			},
			wantFile:    filepath.Join(catalogWorkloadsSubdir, "chatbot"+presetFileExt),
			wantPhrases: []string{"deployment fact", "--tp"},
		},
		{
			name: "hardware entry states a GPU type",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt),
					validHardwareEntry+"gpu_type: H100\n")
			},
			wantFile:    filepath.Join(catalogHardwareSubdir, "h100"+catalogYAMLExt),
			wantPhrases: []string{"deployment fact", "--hardware"},
		},
		{
			name: "storage-device table states a tensor-parallel degree",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, catalogStorageDevicesPath(root),
					validDeviceTable+"deployment: {tensor_parallelism: 2}\n")
			},
			wantFile:    catalogStorageDevicesRelPath,
			wantPhrases: []string{"deployment fact", "--tp"},
		},
		{
			// A second document is read by NOTHING: every typed reader decodes one document,
			// so its keys face neither the strict-key check nor a required-field check. The
			// file is refused rather than half-read.
			name: "model.yaml is a multi-document YAML stream",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, modelEntry, catalogModelEntryFile),
					modelEntryYAML(gateModel)+"---\nlicence: apache-2.0\n")
			},
			wantFile:    catalogModelEntryFile,
			wantPhrases: []string{"multi-document", "2 documents"},
		},
		{
			// The loophole this closes: a deployment fact parked after a `---` was invisible
			// to a guard that only ever unmarshalled the first document.
			name: "a later document of a stream states a GPU",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt),
					validHardwareEntry+"---\ngpu: H100\n")
			},
			wantFile:    filepath.Join(catalogHardwareSubdir, "h100"+catalogYAMLExt),
			wantPhrases: []string{"deployment fact", "--hardware", "document[2].gpu"},
		},
		{
			// A later document that does not even parse is reported here, because no typed
			// parse reaches it: the first document decodes fine and the reader stops.
			name: "a later document of a stream does not parse",
			mutate: func(t *testing.T, root string) {
				writeCatalogFile(t, filepath.Join(root, catalogWorkloadsSubdir, "chatbot"+presetFileExt),
					validWorkloadEntry+"---\nbroken: [1\n")
			},
			wantFile:    filepath.Join(catalogWorkloadsSubdir, "chatbot"+presetFileExt),
			wantPhrases: []string{"document 2", "does not parse"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			root := newCompleteCatalog(t)
			tc.mutate(t, root)
			report := loadOrFatal(t, root)
			err := report.Err()
			if err == nil {
				t.Fatalf("the load accepted a catalog it must reject (%s)", tc.name)
			}
			message := err.Error()
			if !strings.Contains(message, tc.wantFile) {
				t.Errorf("the diagnostic must name the offending file %q; got:\n%s", tc.wantFile, message)
			}
			for _, phrase := range tc.wantPhrases {
				if !strings.Contains(message, phrase) {
					t.Errorf("the diagnostic must state the problem (missing %q); got:\n%s", phrase, message)
				}
			}
		})
	}
}

// TestReadCatalogModelEntry_ParsesIdentityAndProvenance pins what a model entry is FOR: a
// result must be auditable and the vendor fetch reproducible by hand, which takes the whole
// provenance record, not just a name. It also pins that an unquoted ISO date decodes as the
// string it looks like rather than as a YAML timestamp — the field is provenance text, and a
// silently-empty `retrieved` would leave a record that cannot answer "as of when".
func TestReadCatalogModelEntry_ParsesIdentityAndProvenance(t *testing.T) {
	root := newCompleteCatalog(t)
	entry, err := readCatalogModelEntry(filepath.Join(root, catalogModelsSubdir, gateModel), gateModel)
	if err != nil {
		t.Fatalf("readCatalogModelEntry: %v", err)
	}
	for _, got := range []struct{ field, value, want string }{
		{"name", entry.Name, gateModel},
		{"source.provider", entry.Source.Provider, "huggingface"},
		{"source.repo", entry.Source.Repo, "Example/" + gateModel},
		{"source.revision", entry.Source.Revision, "0000000000000000000000000000000000000000"},
		{"source.retrieved", entry.Source.Retrieved, "2026-09-16"},
	} {
		if got.value != got.want {
			t.Errorf("%s = %q, want %q", got.field, got.value, got.want)
		}
	}
}

// TestCatalogStrictLoad_MisnamedStorageTableIsReported covers the one devices/ failure a
// lazily-read namespace would otherwise swallow: the namespace exists and holds YAML, but not
// the storage.yaml BLIS reads. Absent-entirely stays fine (covered by
// TestCatalogStrictLoad_MissingNamespaceIsNotAProblem) — it is the half-present case that
// signals a typo, and that would fail a real run with a message blaming the operator's
// device_class.
func TestCatalogStrictLoad_MisnamedStorageTableIsReported(t *testing.T) {
	root := newCompleteCatalog(t)
	if err := os.Rename(catalogStorageDevicesPath(root),
		filepath.Join(root, catalogDevicesSubdir, "storages"+catalogYAMLExt)); err != nil {
		t.Fatalf("rename storage table: %v", err)
	}
	report := loadOrFatal(t, root)
	err := report.Err()
	if err == nil {
		t.Fatal("a devices namespace holding YAML but no storage.yaml must be reported")
	}
	for _, want := range []string{catalogStorageDevicesFile, "storages" + catalogYAMLExt} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the diagnostic must name %q; got:\n%v", want, err)
		}
	}
}

// TestCatalogStrictLoad_DevicesNamespaceWithNothingReadableIsReported covers the rest of the
// half-present devices/ namespace: the misnamed-table case above lists a sibling .yaml, but a
// namespace holding only prose, or the table filed one directory too deep, is just as broken and
// used to be indistinguishable from an absent namespace (the listing filtered to .yaml files, so
// both filtered down to nothing and the loader returned "no devices to load"). Each would then
// fail a real run with "device_class is not defined", blaming the operator's config.
func TestCatalogStrictLoad_DevicesNamespaceWithNothingReadableIsReported(t *testing.T) {
	for _, tc := range []struct {
		name     string
		populate func(t *testing.T, devices string)
		wantName string
	}{
		{
			name: "only a non-YAML file",
			populate: func(t *testing.T, devices string) {
				writeCatalogFile(t, filepath.Join(devices, "README.txt"), "device tables live here\n")
			},
			wantName: "README.txt",
		},
		{
			name: "the table filed one directory too deep",
			populate: func(t *testing.T, devices string) {
				writeCatalogFile(t, filepath.Join(devices, "storage", catalogStorageDevicesFile), validDeviceTable)
			},
			wantName: "storage",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newCompleteCatalog(t)
			devices := filepath.Join(root, catalogDevicesSubdir)
			if err := os.Remove(catalogStorageDevicesPath(root)); err != nil {
				t.Fatalf("remove storage table: %v", err)
			}
			tc.populate(t, devices)

			report := loadOrFatal(t, root)
			err := report.Err()
			if err == nil {
				t.Fatalf("a devices namespace holding %s but no %s must be reported",
					tc.wantName, catalogStorageDevicesFile)
			}
			for _, want := range []string{catalogStorageDevicesFile, tc.wantName} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("the diagnostic must name %q; got:\n%v", want, err)
				}
			}
		})
	}

	// The negative control that keeps the rule from becoming "devices/ is mandatory": an
	// EMPTY namespace (or an absent one) stays silent, because nothing reads the table until a
	// --kv-offload-config tier names a device_class.
	t.Run("empty namespace stays silent", func(t *testing.T) {
		root := newCompleteCatalog(t)
		if err := os.Remove(catalogStorageDevicesPath(root)); err != nil {
			t.Fatalf("remove storage table: %v", err)
		}
		if err := loadOrFatal(t, root).Err(); err != nil {
			t.Fatalf("an empty devices namespace must load clean; got: %v", err)
		}
	})
}

// TestCatalogStrictLoad_EmptyTrailingDocumentIsAccepted is the precision control on the
// one-document rule: a file ending in a bare `---` carries no second document (yaml.v3 decodes
// it as nil), so refusing it would reject a harmless separator rather than hidden content.
func TestCatalogStrictLoad_EmptyTrailingDocumentIsAccepted(t *testing.T) {
	root := newCompleteCatalog(t)
	writeCatalogFile(t, filepath.Join(root, catalogModelsSubdir, gateModel, catalogModelEntryFile),
		modelEntryYAML(gateModel)+"---\n")
	if err := loadOrFatal(t, root).Err(); err != nil {
		t.Fatalf("a trailing empty YAML document must load clean; got: %v", err)
	}
}

// TestCatalogHardwareKeys_ClassifyEveryCalibField is the drift guard on the hardware namespace's
// one hand-written policy. The ACCEPTED key set is derived reflectively in
// latency.ParseHardwareCalibEntries, so a field added to sim.HardwareCalib is accepted with no
// parser change — but whether a new field must be STATED is a judgement no reflection can make,
// and defaulting it to "optional" would silently reintroduce the silent-zero defect
// hardwareCalibRequiredKeys exists to prevent. So every field must appear in exactly one of the
// two lists, and this test is what forces that decision at the moment the field is added.
func TestCatalogHardwareKeys_ClassifyEveryCalibField(t *testing.T) {
	classification := make(map[string]string, len(hardwareCalibRequiredKeys)+len(hardwareCalibOptionalKeys))
	for _, key := range hardwareCalibRequiredKeys {
		classification[key] = "required"
	}
	for _, key := range hardwareCalibOptionalKeys {
		if was, dup := classification[key]; dup {
			t.Errorf("%q is classified both %s and optional; a key must be one or the other", key, was)
		}
		classification[key] = "optional"
	}

	typ := reflect.TypeOf(sim.HardwareCalib{})
	fields := make(map[string]bool, typ.NumField())
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if !f.IsExported() {
			continue // invisible to the decoder, so not a catalog key either
		}
		key := f.Name
		if tag, ok := f.Tag.Lookup("json"); ok {
			if name := strings.Split(tag, ",")[0]; name == "-" {
				continue
			} else if name != "" {
				key = name
			}
		}
		fields[key] = true
		if _, classified := classification[key]; !classified {
			t.Errorf("sim.HardwareCalib field %q is in neither hardwareCalibRequiredKeys nor "+
				"hardwareCalibOptionalKeys: decide whether a catalog hardware entry must STATE it "+
				"(an omitted non-pointer float reads 0) and add it to the right list", key)
		}
	}
	for key := range classification {
		if !fields[key] {
			t.Errorf("%q is classified but is not a sim.HardwareCalib field; it could never be "+
				"required or omitted", key)
		}
	}
	// A positivity rule on a key nothing requires would never run.
	for _, key := range hardwareCalibPositiveKeys {
		if classification[key] != "required" {
			t.Errorf("%q must be > 0 but is not required to be present; the check would never run", key)
		}
	}
}

// TestCatalogHardwareEntry_SharesTheRunPathValueValidation pins the PARITY the catalog reader
// depends on: the values a hardware entry may hold are decided by
// latency.ValidateHardwareCalibEntry, the single home for the load-boundary rules, so the
// catalog namespace and the run path (latency.GetHWConfig over hardware_config.json) accept and
// reject the same entries (R23). A rule that lands there fails a catalog entry too; a rule added
// to only one path is what this asserts against, and its row goes here.
func TestCatalogHardwareEntry_SharesTheRunPathValueValidation(t *testing.T) {
	const gpu = "h100"
	for _, tc := range []struct {
		name       string
		yamlBody   string
		wantReject bool
	}{
		{
			name:     "a complete interconnect pair is accepted by both",
			yamlBody: validHardwareEntry,
		},
		{
			name:       "a half-set interconnect pair is rejected by both",
			yamlBody:   strings.Replace(validHardwareEntry, "InterNodeBwGBps: 50\n", "", 1),
			wantReject: true,
		},
		{
			name:       "a negative per-hop latency is rejected by both",
			yamlBody:   validHardwareEntry + "InterNodeHopLatencyUs: -3.0\n",
			wantReject: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			yamlPath := filepath.Join(dir, gpu+catalogYAMLExt)
			writeCatalogFile(t, yamlPath, tc.yamlBody)
			_, catalogErr := readCatalogHardwareEntry(yamlPath)

			// The same entry as the run path sees it: one GPU of a hardware_config.json.
			calib, err := parseHardwareYAMLForTest(t, tc.yamlBody)
			if err != nil {
				t.Fatalf("convert the fixture entry to a hardware config: %v", err)
			}
			jsonPath := filepath.Join(dir, "hardware_config.json")
			writeCatalogFile(t, jsonPath, calib)
			_, runErr := latency.GetHWConfig(jsonPath, gpu)

			if (catalogErr != nil) != (runErr != nil) {
				t.Fatalf("the catalog namespace and the run path disagree about this entry — "+
					"catalog: %v; run path (GetHWConfig): %v", catalogErr, runErr)
			}
			if tc.wantReject && catalogErr == nil {
				t.Fatalf("both paths accepted an entry they must reject")
			}
			if !tc.wantReject && catalogErr != nil {
				t.Fatalf("both paths rejected a valid entry: %v", catalogErr)
			}
		})
	}
}

// parseHardwareYAMLForTest re-expresses a hardware/<gpu>.yaml body as the hardware_config.json
// payload the run path reads, so one fixture drives both sides of the parity test above.
func parseHardwareYAMLForTest(t *testing.T, body string) (string, error) {
	t.Helper()
	var fields map[string]any
	if err := yaml.Unmarshal([]byte(body), &fields); err != nil {
		return "", err
	}
	payload, err := json.Marshal(map[string]any{"h100": fields})
	return string(payload), err
}

// TestCatalogStrictLoad_VendorConfigMayStatePretrainingTP is the precision control on the
// deployment-fact rule: it is scoped to catalog-AUTHORED YAML, never the vendor config.json,
// which is committed verbatim and never edited. 10 of the authoritative catalog's vendor
// configs carry `pretraining_tp` — the TP degree the checkpoint was PRETRAINED with, an
// architectural fact — so a scan that reached into them would reject most of the catalog for
// stating something it is right to state.
func TestCatalogStrictLoad_VendorConfigMayStatePretrainingTP(t *testing.T) {
	root := newCompleteCatalog(t)
	path := filepath.Join(root, catalogModelsSubdir, gateModel, hfConfigFile)
	config, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}
	// Splice the vendor keys in without disturbing anything else the parser reads.
	withTP := strings.Replace(string(config), "{", "{\n  \"pretraining_tp\": 1,\n  \"tensor_parallel_size\": 8,", 1)
	writeCatalogFile(t, path, withTP)

	report := loadOrFatal(t, root)
	if err := report.Err(); err != nil {
		t.Fatalf("a vendor config.json stating pretraining_tp must load clean; got: %v", err)
	}
}

// TestCatalogStrictLoad_ReportsEveryProblemDeterministically pins two properties of the
// report: one broken file does not hide another (an operator fixes everything in one pass
// rather than one push per defect), and the problem list is deterministic (INV-6 — a CI
// diagnostic that reordered run to run would be unreviewable).
func TestCatalogStrictLoad_ReportsEveryProblemDeterministically(t *testing.T) {
	root := newCompleteCatalog(t)
	writeCatalogFile(t, filepath.Join(root, catalogModelsSubdir, gateModel, catalogModelEntryFile),
		modelEntryYAML(gateModel)+"gpu: H100\n")
	writeCatalogFile(t, filepath.Join(root, catalogHardwareSubdir, "h100"+catalogYAMLExt),
		validHardwareEntry+"MemoryGB: 80.0\n")
	writeCatalogFile(t, filepath.Join(root, catalogWorkloadsSubdir, "chatbot"+presetFileExt),
		validWorkloadEntry+"prompt_tokens_median: 12\n")

	first := loadOrFatal(t, root)
	if len(first.Problems) != 3 {
		t.Fatalf("three broken files must produce three problems, not %d:\n%v", len(first.Problems), first.Err())
	}
	if !sort.StringsAreSorted(first.Problems) {
		t.Errorf("problems must be sorted so the diagnostic is byte-identical across runs; got %v", first.Problems)
	}
	for i := 0; i < 3; i++ {
		again := loadOrFatal(t, root)
		if strings.Join(again.Problems, "\n") != strings.Join(first.Problems, "\n") {
			t.Fatalf("repeat load %d produced a different problem list:\n%v\nvs\n%v", i, again.Problems, first.Problems)
		}
	}
}

// TestCatalogStrictLoad_UnusableRootIsARootLevelError separates "this catalog is broken" from
// "this path is not a catalog": an absent or non-directory root is reported as a root-level
// error rather than as a per-file problem, so a mistyped --catalog is never dressed up as a
// broken model entry (the same distinction resolveCatalogRoot draws, #1776).
func TestCatalogStrictLoad_UnusableRootIsARootLevelError(t *testing.T) {
	notADir := filepath.Join(t.TempDir(), "catalog-file")
	writeCatalogFile(t, notADir, "not a catalog\n")

	for _, tc := range []struct{ name, root string }{
		{"empty root", ""},
		{"absent root", filepath.Join(t.TempDir(), "no-such-catalog")},
		{"root is a file", notADir},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := loadCatalog(tc.root); err == nil {
				t.Fatalf("loadCatalog(%q) must fail at the root level", tc.root)
			}
		})
	}
}

// TestCatalogStrictLoad_MissingNamespaceIsNotAProblem pins the deliberate asymmetry: models/
// is required (a root with no models is not a catalog, and accepting it would let this gate
// pass on a mistyped path), while hardware/, workloads/ and devices/ are loaded when present
// and skipped when absent — a catalog is allowed to predate a namespace, and the gate's
// non-vacuity assertions are what stop "absent" from hiding behind "loaded nothing".
func TestCatalogStrictLoad_MissingNamespaceIsNotAProblem(t *testing.T) {
	root := newCompleteCatalog(t)
	for _, dir := range []string{catalogHardwareSubdir, catalogWorkloadsSubdir, catalogDevicesSubdir} {
		if err := os.RemoveAll(filepath.Join(root, dir)); err != nil {
			t.Fatalf("remove %s: %v", dir, err)
		}
	}
	report := loadOrFatal(t, root)
	if err := report.Err(); err != nil {
		t.Fatalf("a catalog with only a models/ namespace must load clean; got: %v", err)
	}
	if report.Models == 0 {
		t.Error("the models namespace must still have loaded")
	}

	t.Run("models namespace is required", func(t *testing.T) {
		bare := t.TempDir()
		report, err := loadCatalog(bare)
		if err != nil {
			t.Fatalf("a directory with no models/ is still a usable root to inspect: %v", err)
		}
		if report.Err() == nil {
			t.Fatal("a catalog with no models/ namespace must be reported")
		}
	})
}

// The CI half of the gate — that the load actually runs against the authoritative catalog, at a
// pinned revision, and fails when it SKIPS or never ran — is implemented by
// scripts/catalog-load-gate.sh and pinned by scripts/catalog_load_gate_test.go. It lives there
// because the script is what holds the pinned revision and what a workflow job calls (that job
// is a pending human edit to .github/workflows/ci.yml, tracked by #1823; see the script header).
// So until #1823 lands, the leg that runs unconditionally on every `go test ./cmd/...` is
// TestCatalogStrictLoad_CommittedFixtureCatalog below, over the committed testdata/catalog.

// TestNormalizeCatalogKey pins the key folding the deployment-fact rule relies on, so a
// separator or case variant of a banned key cannot slip through.
func TestNormalizeCatalogKey(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"tensor_parallel_size", "tensorparallelsize"},
		{"tensor-parallel-size", "tensorparallelsize"},
		{"tensorParallelSize", "tensorparallelsize"},
		{"TENSOR PARALLEL SIZE", "tensorparallelsize"},
		{"GPU", "gpu"},
		{"read_bandwidth", "readbandwidth"},
	} {
		if got := normalizeCatalogKey(tc.in); got != tc.want {
			t.Errorf("normalizeCatalogKey(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	// Every declared banned key must already be in normalized form, or it could never match.
	for key := range catalogDeploymentFactKeys {
		if got := normalizeCatalogKey(key); got != key {
			t.Errorf("banned key %q is not in normalized form (%q); it would never match", key, got)
		}
	}
}

package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	"gopkg.in/yaml.v3"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// This file implements the strict WHOLE-CATALOG load behind the R1/C6 acceptance gate
// (#1750). C6 states: "There is no validate command: the simulator validates whatever it
// reads and fails with a message naming the file and the problem. CI runs a load over every
// catalog entry, which is the same code path."
//
// Two consequences shape the design:
//
//  1. NO validate command, and no bespoke validator. Every namespace a run consumes is
//     loaded through the very function the run uses — models through
//     resolveModelConfigInCatalog + latency.GetModelConfig, workload presets through
//     readCatalogPresetWorkload, storage devices through loadCatalogStorageDevices. Only the
//     two namespaces no run reads yet — models/<name>/model.yaml (identity/provenance) and
//     hardware/<gpu>.yaml (calibration still ships in the in-repo hardware_config.json) —
//     get a reader here, and the hardware one delegates its strict key policy to
//     latency.ParseHardwareCalibEntries rather than re-deriving it.
//
//  2. The GATE is a test, not a subcommand: cmd/catalog_load_test.go drives loadCatalog
//     against the committed fixture catalog (testdata/catalog) unconditionally, and against
//     the real blis-catalog checkout in the ci.yml `catalog-load` job via BLIS_CATALOG.
//
// SCOPE BOUNDARY, stated because it is the one place this file could be misread as doing
// less than the issue asks: the completeness rule ("a models/<name>/ dir missing either
// config.json or model.yaml fails") and the deployment-fact rule are properties of the
// CATALOG AS A WHOLE, enforced here. `blis run` is deliberately NOT tightened to require
// model.yaml: it resolves exactly ONE model and its refusal diagnostics are settled
// behaviour (#1771/#1774/#1776), so making a run depend on a provenance file it never reads
// would break working scratch clones for no fidelity gain. The config.json half of the rule
// IS the run path — a model whose config.json is absent or unparseable fails a run and fails
// this gate through the same code.

const (
	// catalogHardwareSubdir is the hardware namespace inside a catalog CLONE ROOT (#1774) —
	// one file per GPU, a SIBLING of models/, workloads/ and devices/.
	catalogHardwareSubdir = "hardware"
	// catalogModelEntryFile is the identity/provenance half of a model entry. The other
	// half is hfConfigFile (the vendor's config.json, committed verbatim).
	catalogModelEntryFile = "model.yaml"
	// catalogYAMLExt is the extension of every catalog-authored file (as opposed to the
	// vendor JSON configs).
	catalogYAMLExt = ".yaml"
)

// catalogModelSource records where a model entry's config.json came from, so a result can be
// audited and the fetch reproduced by hand.
type catalogModelSource struct {
	Provider  string `yaml:"provider"`
	Repo      string `yaml:"repo"`
	Revision  string `yaml:"revision"`
	Retrieved string `yaml:"retrieved"`
}

// catalogModelEntry is <catalog>/models/<name>/model.yaml: what this model IS and where its
// config.json came from. It is deliberately NOT a place for deployment facts — the GPU and
// the tensor-parallel degree are stated on the command line (NS-6, #1733), which
// rejectCatalogDeploymentFacts enforces for every catalog-authored file.
//
// No run reads this file today (resolveModelConfig reads config.json only), so the strict
// load below is the only thing standing between the catalog and an unnoticed typo in it.
type catalogModelEntry struct {
	Name   string             `yaml:"name"`
	Source catalogModelSource `yaml:"source"`
}

// catalogLoadReport is the outcome of one whole-catalog load: how much was loaded per
// namespace, plus every problem found. Counts are reported so the gate can assert
// NON-VACUITY — a loader that silently walked an empty tree would otherwise pass.
//
// Problems are ACCUMULATED rather than fail-fast, and sorted, so one CI run names every
// broken file instead of making an operator fix them one push at a time (and so the
// diagnostic is byte-identical across runs, INV-6).
type catalogLoadReport struct {
	Models        int
	Hardware      int
	Workloads     int
	DeviceClasses int
	Problems      []string
}

// Err returns a single error naming every problem found, or nil when the catalog loaded
// clean. The message leads with the count so a CI log line is readable without expanding it.
func (r catalogLoadReport) Err() error {
	if len(r.Problems) == 0 {
		return nil
	}
	return fmt.Errorf("catalog load found %d problem(s):\n  %s",
		len(r.Problems), strings.Join(r.Problems, "\n  "))
}

// loadCatalog loads EVERY entry in the catalog rooted at root, strictly, through the same
// code paths a run uses (see the file header). It returns what it loaded and what it
// rejected; it never writes to the catalog and never fetches anything.
//
// models/ must exist and hold at least one entry — a root with no models is not a catalog,
// and accepting it would make this gate pass on a mistyped path. The other three namespaces
// are loaded when present and skipped when absent, because a catalog is allowed to predate a
// namespace (nothing consumes hardware/ yet) and a hard requirement here would fail a
// perfectly usable catalog. The gate asserts the counts it expects, so "absent" cannot hide
// behind "loaded nothing".
func loadCatalog(root string) (catalogLoadReport, error) {
	if root == "" {
		return catalogLoadReport{}, fmt.Errorf("catalog root is empty; pass --catalog or set %s", catalogEnvVar)
	}
	info, err := os.Stat(root)
	if err != nil {
		return catalogLoadReport{}, fmt.Errorf("catalog root %q cannot be inspected: %w", root, err)
	}
	if !info.IsDir() {
		return catalogLoadReport{}, fmt.Errorf("catalog root %q is not a directory", root)
	}

	var report catalogLoadReport
	var problems []string
	for _, namespace := range []struct {
		count *int
		load  func(string) (int, []string)
	}{
		{&report.Models, loadCatalogModelEntries},
		{&report.Hardware, loadCatalogHardwareEntries},
		{&report.Workloads, loadCatalogWorkloadEntries},
		{&report.DeviceClasses, loadCatalogDeviceEntries},
	} {
		loaded, found := namespace.load(root)
		*namespace.count = loaded
		problems = append(problems, found...)
	}
	sort.Strings(problems)
	report.Problems = problems
	return report, nil
}

// loadCatalogModelEntries loads every models/<name>/ entry: both halves must be present, the
// vendor config.json must resolve and parse through the RUN path, and model.yaml must parse
// strictly and name its own directory.
func loadCatalogModelEntries(root string) (int, []string) {
	dir := filepath.Join(root, catalogModelsSubdir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, []string{fmt.Sprintf("%s: models namespace cannot be read: %v (a catalog must have a %s/ directory)",
			dir, err, catalogModelsSubdir)}
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		return 0, []string{fmt.Sprintf("%s: models namespace holds no model entries", dir)}
	}

	var problems []string
	loaded := 0
	for _, name := range names {
		entryProblems := loadCatalogModelEntry(root, name)
		problems = append(problems, entryProblems...)
		if len(entryProblems) == 0 {
			loaded++
		}
	}
	return loaded, problems
}

// loadCatalogModelEntry loads one models/<name>/ entry and returns every problem with it.
// Both halves are reported independently: a broken model.yaml must not hide a broken
// config.json, or an operator fixes one, re-pushes, and learns about the other.
func loadCatalogModelEntry(root, name string) []string {
	var problems []string
	entryDir := filepath.Join(root, catalogModelsSubdir, name)

	// config.json — the RUN path, verbatim: resolveModelConfigInCatalog is what
	// `blis run --model .../<name>` calls, and latency.GetModelConfig is what consumes its
	// result. A config that fails here fails a run, which is the property C6 asks CI to
	// prove over every entry.
	resolvedDir, err := resolveModelConfigInCatalog(name, root)
	switch {
	case err != nil:
		problems = append(problems, fmt.Sprintf("models/%s: %v", name, err))
	default:
		if _, cfgErr := latency.GetModelConfig(filepath.Join(resolvedDir, hfConfigFile)); cfgErr != nil {
			problems = append(problems, fmt.Sprintf("%s: %v", filepath.Join(resolvedDir, hfConfigFile), cfgErr))
		}
	}

	// model.yaml — no run reads it, so this is its only validation.
	if _, yamlErr := readCatalogModelEntry(entryDir, name); yamlErr != nil {
		problems = append(problems, yamlErr.Error())
	}
	return problems
}

// readCatalogModelEntry reads and strictly validates <catalog>/models/<name>/model.yaml.
//
// Every failure names the file (C6): absent, unreadable, an unknown key, a missing identity
// or provenance field, or a name that disagrees with the directory it sits in. Strict
// parsing (R10) so a misspelled key is refused rather than silently dropped — for a
// provenance record a dropped field means an unauditable result, which is the whole reason
// the file exists.
//
// The name/directory agreement check exists because the DIRECTORY is what --model resolves
// against: a model.yaml naming a different model makes the entry's provenance describe
// something other than the config.json beside it.
func readCatalogModelEntry(entryDir, name string) (*catalogModelEntry, error) {
	path := filepath.Join(entryDir, catalogModelEntryFile)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
			return nil, fmt.Errorf("%s: model entry is incomplete — no %s (a catalog model entry is "+
				"the vendor's %s plus %s, which records the name and the source repo/revision it was "+
				"fetched from)", path, catalogModelEntryFile, hfConfigFile, catalogModelEntryFile)
		}
		return nil, fmt.Errorf("%s: %s is not readable: %w", path, catalogModelEntryFile, err)
	}

	// Deployment facts first, so the specific "not catalog data" message wins over the
	// generic unknown-key one (the same precedence rejectLegacyInterNodeLatencyKey takes
	// over rejectUnknownHardwareCalibKeys).
	if err := rejectCatalogDeploymentFacts(path, data); err != nil {
		return nil, err
	}

	var entry catalogModelEntry
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&entry); err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%s: %s is not a valid model entry: %w", path, catalogModelEntryFile, err)
	}

	// Required fields, checked explicitly: KnownFields(true) refuses unknown keys but does
	// not require declared ones, so an omitted key is indistinguishable from an empty
	// string — the silent-zero class of defect one namespace over (parseCatalogStorageDevices
	// rescans for the same reason).
	missing := make([]string, 0, 4)
	for _, f := range []struct {
		key   string
		value string
	}{
		{"name", entry.Name},
		{"source.provider", entry.Source.Provider},
		{"source.repo", entry.Source.Repo},
		{"source.revision", entry.Source.Revision},
	} {
		if strings.TrimSpace(f.value) == "" {
			missing = append(missing, f.key)
		}
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("%s: %s is missing required field(s) %v (an entry states its name and "+
			"where its %s came from: provider, repo and the exact revision)",
			path, catalogModelEntryFile, missing, hfConfigFile)
	}
	if entry.Name != name {
		return nil, fmt.Errorf("%s: %s declares name %q but sits in directory %q; the directory name is "+
			"what --model resolves against, so the two must agree", path, catalogModelEntryFile, entry.Name, name)
	}
	return &entry, nil
}

// loadCatalogHardwareEntries loads every hardware/<gpu>.yaml file. Absent namespace = not a
// problem (nothing consumes it yet); present namespace = every file in it must load.
func loadCatalogHardwareEntries(root string) (int, []string) {
	dir := filepath.Join(root, catalogHardwareSubdir)
	paths, err := catalogYAMLFiles(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, []string{fmt.Sprintf("%s: hardware namespace cannot be read: %v", dir, err)}
	}
	var problems []string
	loaded := 0
	for _, path := range paths {
		if _, err := readCatalogHardwareEntry(path); err != nil {
			problems = append(problems, err.Error())
			continue
		}
		loaded++
	}
	return loaded, problems
}

// hardwareCalibRequiredKeys are the calibration keys a catalog hardware file must STATE. An
// omitted key decodes to 0, which for peak FLOPs, bandwidth, MFU or memory capacity is
// plausible-but-wrong physics with no diagnostic — the defect strict parsing exists to
// prevent (R9/R10), and one KnownFields(true) cannot catch because absent and zero are the
// same thing for a non-pointer float.
//
// TFlopsFP8 is required to be PRESENT but may legitimately be 0 (an A100 has no FP8 path),
// so it is listed here and excluded from the positivity check below.
var hardwareCalibRequiredKeys = []string{
	"TFlopsPeak", "TFlopsFP8", "BwPeakTBs", "mfuPrefill", "mfuDecode", "MemoryGiB",
}

// hardwareCalibPositiveKeys are the required keys whose value must additionally be > 0.
var hardwareCalibPositiveKeys = []string{
	"TFlopsPeak", "BwPeakTBs", "mfuPrefill", "mfuDecode", "MemoryGiB",
}

// readCatalogHardwareEntry reads and strictly validates one <catalog>/hardware/<gpu>.yaml:
// a flat mapping of sim.HardwareCalib fields (no wrapping GPU key — the filename is the GPU).
//
// The strict KEY policy is NOT re-implemented here. The file's mapping is converted to the
// hardware-config payload shape and decoded by latency.ParseHardwareCalibEntries, the same
// function parseHWConfig uses for the in-repo hardware_config.json (R23) — so the catalog and
// the bundled file accept exactly the same keys, reject the same typos with the same
// case-mismatch diagnostic, and reject the retired per-collective InterNodeLatencyUs key
// identically. A second key list would be free to drift from the one the decoder honours.
//
// The interconnect pair is validated through sim.HardwareCalib.ValidateInterconnect, the same
// check GetHWConfig applies at the load boundary, so a half-set bandwidth pair fails here too.
func readCatalogHardwareEntry(path string) (sim.HardwareCalib, error) {
	gpu := strings.TrimSuffix(filepath.Base(path), catalogYAMLExt)
	data, err := os.ReadFile(path)
	if err != nil {
		return sim.HardwareCalib{}, fmt.Errorf("%s: hardware entry is not readable: %w", path, err)
	}
	if err := rejectCatalogDeploymentFacts(path, data); err != nil {
		return sim.HardwareCalib{}, err
	}

	var fields map[string]any
	if err := yaml.Unmarshal(data, &fields); err != nil {
		return sim.HardwareCalib{}, fmt.Errorf("%s: hardware entry is not a valid YAML mapping: %w", path, err)
	}
	if len(fields) == 0 {
		return sim.HardwareCalib{}, fmt.Errorf("%s: hardware entry declares no calibration fields", path)
	}
	for _, req := range hardwareCalibRequiredKeys {
		if _, ok := fields[req]; !ok {
			return sim.HardwareCalib{}, fmt.Errorf("%s: hardware entry is missing required field %q "+
				"(an omitted calibration field would silently read 0)", path, req)
		}
	}

	// Hand the mapping to the shared strict decoder in the payload shape it contracts to:
	// {"<gpu>": {<fields>}}.
	payload, err := json.Marshal(map[string]any{gpu: fields})
	if err != nil {
		return sim.HardwareCalib{}, fmt.Errorf("%s: hardware entry holds a value that is not representable "+
			"as hardware calibration: %w", path, err)
	}
	calibs, err := latency.ParseHardwareCalibEntries(payload)
	if err != nil {
		return sim.HardwareCalib{}, fmt.Errorf("%s: %w", path, err)
	}
	calib, ok := calibs[gpu]
	if !ok {
		// Defensive: the payload was built with exactly this key.
		return sim.HardwareCalib{}, fmt.Errorf("%s: hardware entry for %q did not decode", path, gpu)
	}
	for _, key := range hardwareCalibPositiveKeys {
		if v, numeric := yamlFloat(fields[key]); !numeric || v <= 0 {
			return sim.HardwareCalib{}, fmt.Errorf("%s: hardware entry field %q must be a number > 0, got %v",
				path, key, fields[key])
		}
	}
	if err := calib.ValidateInterconnect(); err != nil {
		return sim.HardwareCalib{}, fmt.Errorf("%s: %w", path, err)
	}
	return calib, nil
}

// yamlFloat reports a YAML scalar's numeric value. yaml.v3 decodes an integer literal to int
// and a float literal to float64, so a single type assertion would reject half the valid
// files (MemoryGiB: 80.0 vs IntraNodeBwGBps: 450).
func yamlFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case float64:
		return n, true
	default:
		return 0, false
	}
}

// loadCatalogWorkloadEntries loads every workloads/<name>.yaml preset through
// readCatalogPresetWorkload — the production reader shared by `blis run --workload`,
// `blis observe --workload` and `blis convert preset --name` (#1769).
func loadCatalogWorkloadEntries(root string) (int, []string) {
	dir := filepath.Join(root, catalogWorkloadsSubdir)
	paths, err := catalogYAMLFiles(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, []string{fmt.Sprintf("%s: workloads namespace cannot be read: %v", dir, err)}
	}
	var problems []string
	loaded := 0
	for _, path := range paths {
		name := strings.TrimSuffix(filepath.Base(path), presetFileExt)
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			problems = append(problems, fmt.Sprintf("%s: workload preset is not readable: %v", path, readErr))
			continue
		}
		if factErr := rejectCatalogDeploymentFacts(path, data); factErr != nil {
			problems = append(problems, factErr.Error())
			continue
		}
		if _, err := readCatalogPresetWorkload(name, root); err != nil {
			problems = append(problems, err.Error())
			continue
		}
		loaded++
	}
	return loaded, problems
}

// loadCatalogDeviceEntries loads devices/storage.yaml through loadCatalogStorageDevices —
// the production reader a run uses when a --kv-offload-config secondary tier names a
// device_class (#1770). Absent file = not a problem (the run-time dependency is lazy).
func loadCatalogDeviceEntries(root string) (int, []string) {
	path := catalogStorageDevicesPath(root)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, []string{fmt.Sprintf("%s: storage-device table is not readable: %v", path, err)}
	}
	if factErr := rejectCatalogDeploymentFacts(path, data); factErr != nil {
		return 0, []string{factErr.Error()}
	}
	devices, err := loadCatalogStorageDevices(root)
	if err != nil {
		return 0, []string{err.Error()}
	}
	return len(devices), nil
}

// catalogYAMLFiles lists the .yaml files directly inside dir, sorted so a load is
// deterministic (INV-6). A missing directory is returned as os.ErrNotExist for the caller to
// treat as "namespace absent" rather than "namespace broken".
func catalogYAMLFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), catalogYAMLExt) {
			continue
		}
		paths = append(paths, filepath.Join(dir, e.Name()))
	}
	sort.Strings(paths)
	return paths, nil
}

// catalogDeploymentFactKeys are the key spellings a catalog file may NOT use, in the
// normalized form normalizeCatalogKey produces. Deployment choices — which GPU, how many
// tensor-parallel ranks — are stated on the command line and required there
// (requireDeploymentFlags, NS-6 / #1733); a catalog that also stated them would give a run
// two disagreeing sources for one fact, and the catalog's copy would be the one no operator
// chose.
//
// KEY-based, deliberately: a VALUE scan would have to guess which strings name GPUs
// ("h100" is also a plausible nickname in a provenance comment), while a key is a
// declaration. The set covers the spellings a person would actually reach for, including the
// vLLM/BLIS flag names.
var catalogDeploymentFactKeys = map[string]string{
	"gpu":                     "GPU type is a deployment choice (--hardware)",
	"gpus":                    "GPU type is a deployment choice (--hardware)",
	"gputype":                 "GPU type is a deployment choice (--hardware)",
	"hardware":                "GPU type is a deployment choice (--hardware)",
	"tp":                      "tensor-parallel degree is a deployment choice (--tp)",
	"tpsize":                  "tensor-parallel degree is a deployment choice (--tp)",
	"tpdegree":                "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallel":          "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallelism":       "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallelsize":      "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparalleldegree":    "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallelismsize":   "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallelismdegree": "tensor-parallel degree is a deployment choice (--tp)",
}

// normalizeCatalogKey folds a YAML key to the form catalogDeploymentFactKeys is written in:
// lowercased with separators removed, so tensor_parallel_size, tensor-parallel-size,
// tensorParallelSize and TENSOR PARALLEL SIZE are one key.
func normalizeCatalogKey(key string) string {
	var b strings.Builder
	b.Grow(len(key))
	for _, r := range strings.ToLower(key) {
		switch r {
		case '_', '-', ' ', '.':
			continue
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// rejectCatalogDeploymentFacts refuses a catalog-authored YAML file that states a GPU or a
// tensor-parallel degree, at ANY nesting depth, naming the file and the offending key path.
//
// SCOPE: the catalog's own YAML (models/*/model.yaml, hardware/*.yaml, workloads/*.yaml,
// devices/*.yaml) — NOT the vendor config.json, which is committed VERBATIM and never
// edited. That boundary is load-bearing rather than tidy: 10 of the 23 committed vendor
// configs carry `pretraining_tp`, the TP degree a model was PRETRAINED with — an
// architectural fact of the checkpoint, not a choice about how to serve it. Scanning the
// vendor file would reject most of the catalog for stating something it is right to state.
//
// A file that is not a YAML mapping at all defers to the typed parse, which produces the
// real error (the same nilerr shape the sibling guards in sim/latency/config.go use).
func rejectCatalogDeploymentFacts(path string, data []byte) error {
	var doc any
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil //nolint:nilerr // defer the diagnostic to the typed parse
	}
	offenders := collectCatalogDeploymentFacts(doc, "")
	if len(offenders) == 0 {
		return nil
	}
	// Sorted so a multi-offender diagnostic is byte-identical across runs (INV-6).
	sort.Strings(offenders)
	return fmt.Errorf("%s: catalog file states deployment fact(s): %s. A catalog says what a model, "+
		"a chip, a workload or a storage tier IS; the deployment — which GPU, how many tensor-parallel "+
		"ranks — is stated on the command line and required there (--hardware / --tp). Remove the key(s)",
		path, strings.Join(offenders, "; "))
}

// collectCatalogDeploymentFacts walks a decoded YAML document and returns one entry per
// deployment-fact key, qualified by its path in the document.
func collectCatalogDeploymentFacts(node any, prefix string) []string {
	var offenders []string
	switch n := node.(type) {
	case map[string]any:
		for key, value := range n {
			qualified := key
			if prefix != "" {
				qualified = prefix + "." + key
			}
			if reason, bad := catalogDeploymentFactKeys[normalizeCatalogKey(key)]; bad {
				offenders = append(offenders, fmt.Sprintf("%q (%s)", qualified, reason))
			}
			offenders = append(offenders, collectCatalogDeploymentFacts(value, qualified)...)
		}
	case map[any]any:
		// yaml.v3 yields map[string]any for string keys; this arm covers a document with
		// non-string keys, which must still be walked rather than silently skipped.
		for key, value := range n {
			qualified := fmt.Sprintf("%v", key)
			if prefix != "" {
				qualified = prefix + "." + qualified
			}
			if reason, bad := catalogDeploymentFactKeys[normalizeCatalogKey(fmt.Sprintf("%v", key))]; bad {
				offenders = append(offenders, fmt.Sprintf("%q (%s)", qualified, reason))
			}
			offenders = append(offenders, collectCatalogDeploymentFacts(value, qualified)...)
		}
	case []any:
		for i, value := range n {
			offenders = append(offenders, collectCatalogDeploymentFacts(value, fmt.Sprintf("%s[%d]", prefix, i))...)
		}
	}
	return offenders
}

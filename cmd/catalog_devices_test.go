package cmd

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// Unit-level contracts for #1770: the KV-offload storage-device physics table is read
// from the CATALOG (<catalog>/devices/storage.yaml) and nowhere else. Before this it was
// duplicated between defaults.yaml (the copy actually read) and blis-catalog's
// devices/storage.yaml (read by nothing), with nothing keeping the copies in sync.
//
// The laws these pin:
//
//	BC-1  the table's location is <catalog>/devices/storage.yaml — a SIBLING of models/
//	BC-2  INV-6: the catalog table resolves the SAME numbers the deleted defaults.yaml
//	      block resolved, so an offload run naming a device_class is unchanged
//	BC-3  defaults.yaml no longer declares kv_offload_devices, and a stale block is
//	      refused at load rather than silently ignored (R10 one-way)
//	BC-4  an unknown device_class is refused naming the catalog file and the known classes
//	BC-5  an absent / malformed / empty table is a hard error naming the path (R1) —
//	      never silent zero physics and never a misleading "unknown device_class"
//	BC-6  the table is a LAZY dependency: a config whose tiers all give explicit
//	      bandwidth/latency triples resolves without any catalog devices/ namespace
//
// The CLI-level halves (the resolved numbers reaching a real run's trace header, and
// run/replay parity) live in catalog_devices_cli_test.go.

// historicalKVOffloadDevices is the pre-#1770 defaults.yaml `kv_offload_devices:` table,
// transcribed verbatim from that block, and is the INV-6 reference for BC-2. It is a
// FROZEN golden: it must not be regenerated from whatever the catalog currently says,
// because the whole point is to detect the catalog table drifting away from the numbers
// runs used to get. Changing a number here is a deliberate physics change and must be
// argued on its own.
func historicalKVOffloadDevices() map[string]kvOffloadDevice {
	return map[string]kvOffloadDevice{
		"nvme_gen4": {ReadBandwidth: 7.0e3, WriteBandwidth: 5.0e3, BaseLatency: 80.0},
		"nvme_gen3": {ReadBandwidth: 3.5e3, WriteBandwidth: 3.0e3, BaseLatency: 100.0},
		"sata_ssd":  {ReadBandwidth: 5.5e2, WriteBandwidth: 5.0e2, BaseLatency: 150.0},
		"cpu_dram":  {ReadBandwidth: 2.0e4, WriteBandwidth: 2.0e4, BaseLatency: 1.0},
		"s3":        {ReadBandwidth: 1.0e3, WriteBandwidth: 1.0e3, BaseLatency: 30000.0},
	}
}

// writeCatalogStorageDevices builds a catalog root holding devices/storage.yaml with the
// given body and returns the root. Used by the device-table tests here and by the
// retargeted YAML-parse tests in kv_offload_test.go / kv_offload_device_test.go.
func writeCatalogStorageDevices(t *testing.T, body string) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, catalogDevicesSubdir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, catalogStorageDevicesFile), []byte(body), 0o644); err != nil {
		t.Fatalf("write storage.yaml: %v", err)
	}
	return root
}

// ---------------------------------------------------------------------------
// BC-1: the table lives at <catalog>/devices/storage.yaml
// ---------------------------------------------------------------------------

// TestCatalogStorageDevicesPath_IsDevicesSiblingOfModels pins the location law. The path
// is built from the catalog CLONE ROOT (#1774) with devices/ as a sibling of models/, and
// the root is passed through verbatim — a relative root stays relative (resolved against
// the process working directory) and an absolute root is used as given, exactly as
// catalogModelDirs treats it.
func TestCatalogStorageDevicesPath_IsDevicesSiblingOfModels(t *testing.T) {
	for _, root := range []string{"model_configs", "/abs/catalog", "./rel/catalog"} {
		got := catalogStorageDevicesPath(root)
		want := filepath.Join(root, "devices", "storage.yaml")
		if got != want {
			t.Errorf("catalogStorageDevicesPath(%q) = %q, want %q", root, got, want)
		}
		// Sibling, not nested: the models namespace must not appear in the devices path.
		if strings.Contains(got, string(filepath.Separator)+catalogModelsSubdir+string(filepath.Separator)) {
			t.Errorf("devices path %q must be a SIBLING of %s/, not nested inside it", got, catalogModelsSubdir)
		}
	}
}

// TestResolveKVOffloadDevices_ReadsCatalogLocatedByFlagAndEnv is BC-1 through the
// production wrapper: the devices table is located by the SAME --catalog / BLIS_CATALOG
// resolver the model config uses, so an operator states the catalog location once.
func TestResolveKVOffloadDevices_ReadsCatalogLocatedByFlagAndEnv(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })

	flagCatalog := writeCatalogStorageDevices(t,
		"nvme_gen4: {read_bandwidth: 1.0, write_bandwidth: 2.0, base_latency: 3.0}\n")
	envCatalog := writeCatalogStorageDevices(t,
		"nvme_gen4: {read_bandwidth: 9.0, write_bandwidth: 8.0, base_latency: 7.0}\n")

	// Env only.
	catalogPath = ""
	t.Setenv(catalogEnvVar, envCatalog)
	devices, err := resolveKVOffloadDevices(validBlock())
	if err != nil {
		t.Fatalf("env-only: %v", err)
	}
	if devices["nvme_gen4"].ReadBandwidth != 9.0 {
		t.Errorf("env-only: read the wrong catalog: %+v", devices["nvme_gen4"])
	}

	// Flag wins over env (same precedence law as the model config).
	catalogPath = flagCatalog
	devices, err = resolveKVOffloadDevices(validBlock())
	if err != nil {
		t.Fatalf("flag+env: %v", err)
	}
	if devices["nvme_gen4"].ReadBandwidth != 1.0 {
		t.Errorf("flag+env: --catalog must win, got %+v", devices["nvme_gen4"])
	}

	// Neither: a device_class cannot be resolved, and the refusal names both forms and
	// the file it was going to read.
	catalogPath = ""
	t.Setenv(catalogEnvVar, "")
	if _, err = resolveKVOffloadDevices(validBlock()); err == nil {
		t.Fatal("a device_class with no catalog located must be refused")
	} else {
		for _, frag := range []string{"--catalog", catalogEnvVar, catalogStorageDevicesRelPath} {
			if !strings.Contains(err.Error(), frag) {
				t.Errorf("refusal must name %q, got: %v", frag, err)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// BC-2: INV-6 — the catalog table resolves the historical numbers
// ---------------------------------------------------------------------------

// TestBundledCatalogDevices_MatchHistoricalDefaultsBlock is the cutover's INV-6 guard at
// the data layer: the bundled catalog's table must be class-for-class identical to the
// deleted defaults.yaml block, including the optional #1581 device-model fields (all
// absent, so ramp/jitter stay off). Same numbers in ⇒ same numbers out, whatever the
// simulator does with them — which is why this comparison, not a run-level diff, is the
// load-bearing equivalence check.
func TestBundledCatalogDevices_MatchHistoricalDefaultsBlock(t *testing.T) {
	got, err := loadCatalogStorageDevices("../model_configs")
	if err != nil {
		t.Fatalf("bundled catalog device table must load: %v", err)
	}
	want := historicalKVOffloadDevices()

	if !reflect.DeepEqual(got, want) {
		// Report per class so a single drifted number is obvious.
		names := map[string]bool{}
		for n := range got {
			names[n] = true
		}
		for n := range want {
			names[n] = true
		}
		sorted := make([]string, 0, len(names))
		for n := range names {
			sorted = append(sorted, n)
		}
		sort.Strings(sorted)
		for _, n := range sorted {
			g, gok := got[n]
			w, wok := want[n]
			switch {
			case !gok:
				t.Errorf("class %q was in the pre-#1770 defaults.yaml block but is missing from the catalog table", n)
			case !wok:
				t.Errorf("class %q is new in the catalog table (not in the pre-#1770 defaults.yaml block)", n)
			case !reflect.DeepEqual(g, w):
				t.Errorf("class %q drifted from the pre-#1770 numbers:\n got  %+v\n want %+v", n, g, w)
			}
		}
	}
}

// TestResolveKVOffload_CatalogClassResolvesHistoricalTriple is the same guard one layer
// up, through the real resolver: a tier naming each historical class resolves to exactly
// that class's triple with the device model off. This is the value an offload run
// actually consumes, so it is what INV-6 is about.
func TestResolveKVOffload_CatalogClassResolvesHistoricalTriple(t *testing.T) {
	devices, err := loadCatalogStorageDevices("../model_configs")
	if err != nil {
		t.Fatalf("bundled catalog device table must load: %v", err)
	}
	for class, want := range historicalKVOffloadDevices() {
		cfg, err := resolveKVOffload(rampBlock(class, true), devices, 16)
		if err != nil {
			t.Fatalf("class %q must resolve: %v", class, err)
		}
		tr := cfg.Tiers[0]
		if tr.DeviceClass != class {
			t.Errorf("class %q: resolved device_class = %q", class, tr.DeviceClass)
		}
		if tr.ReadBandwidth != want.ReadBandwidth || tr.WriteBandwidth != want.WriteBandwidth || tr.BaseLatency != want.BaseLatency {
			t.Errorf("class %q: resolved triple (%v, %v, %v), want (%v, %v, %v)",
				class, tr.ReadBandwidth, tr.WriteBandwidth, tr.BaseLatency,
				want.ReadBandwidth, want.WriteBandwidth, want.BaseLatency)
		}
		if tr.SaturationQueueDepth != 1 || tr.SingleTransferFraction != 1.0 || tr.LatencyJitterStddev != 0 {
			t.Errorf("class %q: device model must be off (Qsat=%d f1=%v sigma=%v)",
				class, tr.SaturationQueueDepth, tr.SingleTransferFraction, tr.LatencyJitterStddev)
		}
	}
}

// ---------------------------------------------------------------------------
// BC-3: defaults.yaml no longer declares the block, and a stale one is refused
// ---------------------------------------------------------------------------

// TestKVOffloadDevicesBlockRemoved_BundledDefaultsHasNoBlock is the static half: the
// committed defaults.yaml must not carry a kv_offload_devices: key. A block left behind
// would be doubly wrong — it is refused at load (below), and it would re-create exactly
// the unsynchronised duplicate #1770 removed.
func TestKVOffloadDevicesBlockRemoved_BundledDefaultsHasNoBlock(t *testing.T) {
	data, err := os.ReadFile("../defaults.yaml")
	if err != nil {
		t.Fatalf("read defaults.yaml: %v", err)
	}
	for i, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "kv_offload_devices:") {
			t.Errorf("defaults.yaml:%d still declares a kv_offload_devices: block — the table "+
				"lives in the catalog (%s) since #1770", i+1, catalogStorageDevicesRelPath)
		}
	}
	// Non-vacuity: the file must still be the real one (a truncated read would pass above).
	if !strings.Contains(string(data), "trained_physics_coefficients:") {
		t.Fatal("non-vacuity: defaults.yaml does not look like the committed file")
	}
}

// TestKVOffloadDevicesBlockRemoved_StaleBlockIsRefused is the behavior change #1770 puts
// on the record, and it is the R10 one-way consequence: an operator's hand-maintained
// defaults.yaml that still carries a kv_offload_devices: block no longer loads. That is
// the intended outcome — re-declaring the Go field to accept such a file would reinstate
// a config key with no consumer, which is the silent-acceptance antipattern strict
// parsing exists to prevent. The refusal must name the offending field.
//
// loadDefaultsConfig reports a parse error via logrus.Fatalf, so this runs in a
// subprocess. The negative control (same fixture minus the block) proves the failure can
// only be that key.
func TestKVOffloadDevicesBlockRemoved_StaleBlockIsRefused(t *testing.T) {
	const staleBlock = `kv_offload_devices:
  nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3, base_latency: 80.0}
`
	// #1769 removed the `workloads:` field, so the negative-control remainder is now just a
	// minimal valid file — enough to prove the same fixture minus the kv_offload_devices: block loads.
	const rest = `version: "0.0.1"
`
	if os.Getenv("BLIS_STALE_KV_DEVICES_SUBPROCESS") == "1" {
		path := filepath.Join(t.TempDir(), "defaults.yaml")
		if err := os.WriteFile(path, []byte(staleBlock+rest), 0o644); err != nil {
			t.Fatal(err)
		}
		loadDefaultsConfig(path) // must Fatalf
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestKVOffloadDevicesBlockRemoved_StaleBlockIsRefused$")
	cmd.Env = append(os.Environ(), "BLIS_STALE_KV_DEVICES_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("a stale kv_offload_devices: block must be refused, but the load succeeded:\n%s", out)
	}
	if !strings.Contains(string(out), "kv_offload_devices") {
		t.Errorf("refusal must name the offending field, got:\n%s", out)
	}

	// Negative control: the same file WITHOUT the block loads, so the failure above is
	// attributable to that key alone.
	path := filepath.Join(t.TempDir(), "defaults.yaml")
	if err := os.WriteFile(path, []byte(rest), 0o644); err != nil {
		t.Fatal(err)
	}
	if cfg := loadDefaultsConfig(path); cfg.Version != "0.0.1" {
		t.Errorf("negative control: the same fixture minus the block must load, got version %q", cfg.Version)
	}
}

// ---------------------------------------------------------------------------
// BC-4: an unknown device_class names the catalog file and the known classes
// ---------------------------------------------------------------------------

func TestResolveKVOffload_UnknownDeviceClassNamesCatalogTable(t *testing.T) {
	devices := map[string]kvOffloadDevice{
		"sata_ssd":  {ReadBandwidth: 550, WriteBandwidth: 500, BaseLatency: 150},
		"nvme_gen4": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80},
	}
	_, err := resolveKVOffload(rampBlock("nvme_gen9", true), devices, 16)
	if err == nil {
		t.Fatal("an unknown device_class must be refused")
	}
	for _, frag := range []string{"nvme_gen9", catalogStorageDevicesRelPath} {
		if !strings.Contains(err.Error(), frag) {
			t.Errorf("refusal must name %q, got: %v", frag, err)
		}
	}
	// The known classes are listed in sorted order (INV-6: no map-iteration order in
	// any output, including diagnostics).
	if !strings.Contains(err.Error(), "nvme_gen4, sata_ssd") {
		t.Errorf("refusal must list the known classes sorted, got: %v", err)
	}
	// And it must NOT still point at defaults.yaml, which no longer holds the table.
	if strings.Contains(err.Error(), "defaults.yaml") {
		t.Errorf("refusal must not point at defaults.yaml, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// BC-5: an absent / malformed / empty table is a hard error naming the path
// ---------------------------------------------------------------------------

// TestLoadCatalogStorageDevices_UnusableTableIsRefused: every way the table can fail to
// answer is a refusal naming the path (R1). The alternatives are both silent mis-models:
// an empty map degrades into "device_class X is not defined (known: <none configured>)",
// which blames the operator's config for a missing catalog file, and a zero-valued device
// would give plausible-but-wrong physics with no diagnostic anywhere.
func TestLoadCatalogStorageDevices_UnusableTableIsRefused(t *testing.T) {
	cases := []struct {
		name string
		root func(t *testing.T) string
		// wantFrag distinguishes the failure CLASS, so a "no classes" file is not reported
		// as a malformed document (and vice versa) — the message is the operator's only
		// instruction about what to fix.
		wantFrag string
	}{
		{"no devices namespace", func(t *testing.T) string { return t.TempDir() }, "is not readable"},
		{"empty file", func(t *testing.T) string { return writeCatalogStorageDevices(t, "") }, "defines no device classes"},
		{"comments only", func(t *testing.T) string { return writeCatalogStorageDevices(t, "# no classes here\n") }, "defines no device classes"},
		{"no classes", func(t *testing.T) string { return writeCatalogStorageDevices(t, "{}\n") }, "defines no device classes"},
		{"malformed yaml", func(t *testing.T) string {
			return writeCatalogStorageDevices(t, "nvme_gen4: {read_bandwidth: [unclosed\n")
		}, "is malformed"},
		{"unknown physics key", func(t *testing.T) string {
			// R10: a misspelled key must be refused, never decoded to zero bandwidth.
			return writeCatalogStorageDevices(t, "nvme_gen4: {read_bandwidth: 7.0e3, write_bandwith: 5.0e3, base_latency: 80.0}\n")
		}, "write_bandwith"},
		{"missing required field", func(t *testing.T) string {
			// R9/R10: an OMITTED required physics field must be refused, not silently decoded
			// to 0. base_latency is the sharp case — read/write=0 is caught by Validate, but a
			// missing base_latency would otherwise resolve to zero-latency physics.
			return writeCatalogStorageDevices(t, "nvme_gen4: {read_bandwidth: 7.0e3, write_bandwidth: 5.0e3}\n")
		}, "base_latency"},
		{"table is a directory", func(t *testing.T) string {
			root := t.TempDir()
			if err := os.MkdirAll(filepath.Join(root, catalogDevicesSubdir, catalogStorageDevicesFile), 0o755); err != nil {
				t.Fatal(err)
			}
			return root
		}, "is not readable"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := tc.root(t)
			devices, err := loadCatalogStorageDevices(root)
			if err == nil {
				t.Fatalf("expected refusal, got %d classes", len(devices))
			}
			if devices != nil {
				t.Errorf("a failed load must return no table, got %+v", devices)
			}
			if want := catalogStorageDevicesPath(root); !strings.Contains(err.Error(), want) {
				t.Errorf("refusal must name the path %q, got: %v", want, err)
			}
			if !strings.Contains(err.Error(), tc.wantFrag) {
				t.Errorf("refusal must identify the failure class (%q), got: %v", tc.wantFrag, err)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// BC-6: the table is a lazy dependency
// ---------------------------------------------------------------------------

// TestResolveKVOffloadDevices_NoDeviceClassReadsNoCatalog: an offload config whose tiers
// all carry an explicit read/write/base triple must resolve with NO catalog devices table
// — that class of run never consulted the table before the cutover either, and making it
// suddenly require a catalog file would be a regression dressed up as a refactor.
//
// The check is strong because no catalog is located at all: --catalog is empty and
// BLIS_CATALOG is cleared, so ANY attempt to resolve the catalog would error.
func TestResolveKVOffloadDevices_NoDeviceClassReadsNoCatalog(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })
	catalogPath = ""
	t.Setenv(catalogEnvVar, "")

	explicit := &kvOffloadBlock{
		CPUBytesToUse: i64p(1 << 30),
		SecondaryTiers: []kvOffloadTierBlock{{
			Type:           strp("fs"),
			RootDir:        strp("/mnt/kv"),
			DirectIO:       boolp(true),
			ReadBandwidth:  f64p(7000),
			WriteBandwidth: f64p(5000),
			BaseLatency:    f64p(80),
		}},
	}
	devices, err := resolveKVOffloadDevices(explicit)
	if err != nil {
		t.Fatalf("an explicit-triple config must not need the catalog device table: %v", err)
	}
	if devices != nil {
		t.Errorf("no device_class named ⇒ no table should be loaded, got %+v", devices)
	}

	// It must still resolve end-to-end to the explicit numbers with a nil table.
	cfg, err := resolveKVOffload(explicit, devices, 16)
	if err != nil {
		t.Fatalf("explicit-triple resolve: %v", err)
	}
	tr := cfg.Tiers[0]
	if tr.ReadBandwidth != 7000 || tr.WriteBandwidth != 5000 || tr.BaseLatency != 80 || tr.DeviceClass != "" {
		t.Errorf("explicit triple wrong: %+v", tr)
	}

	// Non-vacuity: the SAME block with a device_class added does need the catalog, so the
	// pass above is the laziness and not an accidentally-optional dependency.
	withClass := *explicit
	withClass.SecondaryTiers = append([]kvOffloadTierBlock(nil), explicit.SecondaryTiers...)
	withClass.SecondaryTiers[0].DeviceClass = strp("nvme_gen4")
	if _, err := resolveKVOffloadDevices(&withClass); err == nil {
		t.Error("non-vacuity: a device_class with no catalog located must be refused")
	}
}

// ---------------------------------------------------------------------------
// BC-3 (static half): nothing in cmd/ can read the old table again
// ---------------------------------------------------------------------------

// TestKVOffloadDevicesBlockRemoved_StaticGuard asserts the cutover is structural rather
// than merely unused: no production source in cmd/ names the deleted symbols. A behavioral
// test can only show that today's inputs do not read the defaults.yaml table; this shows
// there is nothing left to read it with. Mirrors #1768's guard for the `defaults:` block.
func TestKVOffloadDevicesBlockRemoved_StaticGuard(t *testing.T) {
	bannedIdents := map[string]string{
		"KVOffloadDeviceDefaults": "the defaults.yaml device type was replaced by the catalog reader's kvOffloadDevice (#1770)",
		"KVOffloadDevices":        "Config.KVOffloadDevices was removed by #1770 — the table lives in the catalog",
	}
	if len(bannedIdents) == 0 {
		t.Fatal("non-vacuity: the guard has nothing to check")
	}

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob cmd/*.go: %v", err)
	}
	scanned := 0
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		scanned++
		fset := token.NewFileSet()
		parsed, err := parser.ParseFile(fset, file, nil, parser.ParseComments)
		if err != nil {
			t.Fatalf("parse %s: %v", file, err)
		}
		ast.Inspect(parsed, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			if why, banned := bannedIdents[id.Name]; banned {
				t.Errorf("%s:%d: %s must not appear in cmd/'s production sources — %s",
					file, fset.Position(id.Pos()).Line, id.Name, why)
			}
			return true
		})
	}
	if scanned == 0 {
		t.Fatal("non-vacuity: no production sources were scanned")
	}
}

// TestBlockNamesDeviceClass covers the laziness predicate directly, including the nil
// block (which resolveKVOffload rejects later with its own message — the predicate must
// not panic first).
func TestBlockNamesDeviceClass(t *testing.T) {
	if blockNamesDeviceClass(nil) {
		t.Error("a nil block names no device_class")
	}
	if blockNamesDeviceClass(&kvOffloadBlock{CPUBytesToUse: i64p(1)}) {
		t.Error("a block with no secondary tiers names no device_class")
	}
	twoTiers := &kvOffloadBlock{SecondaryTiers: []kvOffloadTierBlock{
		{Type: strp("fs")},
		{Type: strp("fs"), DeviceClass: strp("s3")},
	}}
	if !blockNamesDeviceClass(twoTiers) {
		t.Error("a device_class on ANY tier must require the table")
	}
}

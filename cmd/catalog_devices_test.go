package cmd

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/build"
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

// TestCatalogStorageDevicesPath_IsDevicesSiblingOfModels pins the location law. The path
// is built from the catalog CLONE ROOT (#1774) with devices/ as a sibling of models/, and
// the root is passed through verbatim — a relative root stays relative (resolved against
// the process working directory) and an absolute root is used as given, exactly as
// catalogModelDirs treats it.
func TestCatalogStorageDevicesPath_IsDevicesSiblingOfModels(t *testing.T) {
	for _, root := range []string{"rel/catalog", "/abs/catalog", "./rel/catalog"} {
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
		"nvme_gen4: {read_bandwidth_mb_s: 1.0, write_bandwidth_mb_s: 2.0, base_latency_us: 3.0}\n")
	envCatalog := writeCatalogStorageDevices(t,
		"nvme_gen4: {read_bandwidth_mb_s: 9.0, write_bandwidth_mb_s: 8.0, base_latency_us: 7.0}\n")

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
	if !strings.Contains(string(data), "\nlora:") {
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

	explicit := &build.KVOffloadBlock{
		CPUBytesToUse: i64p(1 << 30),
		SecondaryTiers: []build.KVOffloadTierBlock{{
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
	withClass.SecondaryTiers = append([]build.KVOffloadTierBlock(nil), explicit.SecondaryTiers...)
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
		"KVOffloadDeviceDefaults": "the defaults.yaml device type was replaced by the catalog reader's build.KVOffloadDevice (#1770)",
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

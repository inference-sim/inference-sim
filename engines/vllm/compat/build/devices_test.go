// Moved from inference-sim/cmd/catalog_devices_test.go at ace1528fe4bba59e510eda48e80ee9b8142ea982.

package build

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// historicalKVOffloadDevices is the pre-#1770 defaults.yaml `kv_offload_devices:` table,
// transcribed verbatim from that block, and is the INV-6 reference for BC-2. It is a
// FROZEN golden: it must not be regenerated from whatever the catalog currently says,
// because the whole point is to detect the catalog table drifting away from the numbers
// runs used to get. Changing a number here is a deliberate physics change and must be
// argued on its own.
func historicalKVOffloadDevices() map[string]KVOffloadDevice {
	return map[string]KVOffloadDevice{
		"nvme_gen4": {ReadBandwidth: 7.0e3, WriteBandwidth: 5.0e3, BaseLatency: 80.0},
		"nvme_gen3": {ReadBandwidth: 3.5e3, WriteBandwidth: 3.0e3, BaseLatency: 100.0},
		"sata_ssd":  {ReadBandwidth: 5.5e2, WriteBandwidth: 5.0e2, BaseLatency: 150.0},
		"cpu_dram":  {ReadBandwidth: 2.0e4, WriteBandwidth: 2.0e4, BaseLatency: 1.0},
		"s3":        {ReadBandwidth: 1.0e3, WriteBandwidth: 1.0e3, BaseLatency: 30000.0},
	}
}

// catalogKVOffloadDevices is the table the vendored catalog must state: the frozen
// historical block with the deliberate changes since, each argued in the catalog entry's
// own _comment (blis-catalog 5f6ecab): cpu_dram refitted to the H100 host link (InferenceX
// CollectiveX swap_blocks) and the assumed DGX H100 NVMe RAID0 class added.
func catalogKVOffloadDevices() map[string]KVOffloadDevice {
	t := historicalKVOffloadDevices()
	t["cpu_dram"] = KVOffloadDevice{ReadBandwidth: 27058.0, WriteBandwidth: 46089.0, BaseLatency: 19.3}
	t["nvme_gen4_raid0_8x"] = KVOffloadDevice{ReadBandwidth: 5.0e4, WriteBandwidth: 3.0e4, BaseLatency: 100.0}
	return t
}

// writeCatalogStorageDevices builds a catalog root holding devices/storage.yaml with the
// given body and returns the root. Used by the device-table tests here and by the
// retargeted YAML-parse tests in kv_offload_test.go / kv_offload_device_test.go.
func writeCatalogStorageDevices(t *testing.T, body string) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, CatalogDevicesSubdir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, CatalogStorageDevicesFile), []byte(body), 0o644); err != nil {
		t.Fatalf("write storage.yaml: %v", err)
	}
	return root
}

// ---------------------------------------------------------------------------
// BC-1: the table lives at <catalog>/devices/storage.yaml
// ---------------------------------------------------------------------------

// TestBundledCatalogDevices_MatchHistoricalDefaultsBlock is the cutover's INV-6 guard at
// the data layer: the bundled catalog's table must be class-for-class identical to the
// deleted defaults.yaml block, including the optional #1581 device-model fields (all
// absent, so ramp/jitter stay off). Same numbers in ⇒ same numbers out, whatever the
// simulator does with them — which is why this comparison, not a run-level diff, is the
// load-bearing equivalence check.
func TestBundledCatalogDevices_MatchHistoricalDefaultsBlock(t *testing.T) {
	got, err := LoadCatalogStorageDevices("../../testdata/catalog")
	if err != nil {
		t.Fatalf("bundled catalog device table must load: %v", err)
	}
	want := catalogKVOffloadDevices()

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
	devices, err := LoadCatalogStorageDevices("../../testdata/catalog")
	if err != nil {
		t.Fatalf("bundled catalog device table must load: %v", err)
	}
	for class, want := range catalogKVOffloadDevices() {
		cfg, err := ResolveKVOffload(rampBlock(class, true), devices, 16)
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

func TestResolveKVOffload_UnknownDeviceClassNamesCatalogTable(t *testing.T) {
	devices := map[string]KVOffloadDevice{
		"sata_ssd":  {ReadBandwidth: 550, WriteBandwidth: 500, BaseLatency: 150},
		"nvme_gen4": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80},
	}
	_, err := ResolveKVOffload(rampBlock("nvme_gen9", true), devices, 16)
	if err == nil {
		t.Fatal("an unknown device_class must be refused")
	}
	for _, frag := range []string{"nvme_gen9", CatalogStorageDevicesRelPath} {
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
		{"no classes", func(t *testing.T) string { return writeCatalogStorageDevices(t, "{}\n") }, "declares no storage tier"},
		{"malformed yaml", func(t *testing.T) string {
			return writeCatalogStorageDevices(t, "nvme_gen4: {read_bandwidth_mb_s: [unclosed\n")
		}, "is malformed"},
		{"unknown physics key", func(t *testing.T) string {
			// R10: a misspelled key must be refused, never decoded to zero bandwidth.
			return writeCatalogStorageDevices(t, "nvme_gen4: {read_bandwidth_mb_s: 7.0e3, write_bandwith_mb_s: 5.0e3, base_latency_us: 80.0}\n")
		}, "write_bandwith"},
		{"missing required field", func(t *testing.T) string {
			// R9/R10: an OMITTED required physics field must be refused, not silently decoded
			// to 0. base_latency is the sharp case — read/write=0 is caught by Validate, but a
			// missing base_latency would otherwise resolve to zero-latency physics.
			return writeCatalogStorageDevices(t, "nvme_gen4: {read_bandwidth_mb_s: 7.0e3, write_bandwidth_mb_s: 5.0e3}\n")
		}, "base_latency"},
		{"table is a directory", func(t *testing.T) string {
			root := t.TempDir()
			if err := os.MkdirAll(filepath.Join(root, CatalogDevicesSubdir, CatalogStorageDevicesFile), 0o755); err != nil {
				t.Fatal(err)
			}
			return root
		}, "is not readable"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := tc.root(t)
			devices, err := LoadCatalogStorageDevices(root)
			if err == nil {
				t.Fatalf("expected refusal, got %d classes", len(devices))
			}
			if devices != nil {
				t.Errorf("a failed load must return no table, got %+v", devices)
			}
			if want := CatalogStorageDevicesPath(root); !strings.Contains(err.Error(), want) {
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

// TestBlockNamesDeviceClass covers the laziness predicate directly, including the nil
// block (which ResolveKVOffload rejects later with its own message — the predicate must
// not panic first).
func TestBlockNamesDeviceClass(t *testing.T) {
	if BlockNamesDeviceClass(nil) {
		t.Error("a nil block names no device_class")
	}
	if BlockNamesDeviceClass(&KVOffloadBlock{CPUBytesToUse: i64p(1)}) {
		t.Error("a block with no secondary tiers names no device_class")
	}
	twoTiers := &KVOffloadBlock{SecondaryTiers: []KVOffloadTierBlock{
		{Type: strp("fs")},
		{Type: strp("fs"), DeviceClass: strp("s3")},
	}}
	if !BlockNamesDeviceClass(twoTiers) {
		t.Error("a device_class on ANY tier must require the table")
	}
}

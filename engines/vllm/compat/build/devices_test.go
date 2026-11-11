// Moved from inference-sim/cmd/catalog_devices_test.go at ace1528fe4bba59e510eda48e80ee9b8142ea982.

package build

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/load"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// catalogDeviceClasses are the tier classes the substrate catalog names. It states no rate for
// any of them: a tier's read rate, write rate and fixed cost are registry coefficients
// (offload_tier_<tier>_read_rate, _write_rate, _base_latency), and blis-substrate's own tests hold
// the figures the catalog used to carry to the digit.
var catalogDeviceClasses = []string{"cpu_dram", "nvme_gen3", "nvme_gen4", "nvme_gen4_raid0_8x", "s3", "sata_ssd"}

// dataWithStorageDevices is the default data with devices/storage.yaml replaced by body,
// through a catalog overlay, opened and checked by blis-substrate's loader as a run's data is.
func dataWithStorageDevices(t *testing.T, body string) (*load.Data, error) {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, CatalogDevicesSubdir), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, CatalogDevicesSubdir, CatalogStorageDevicesFile), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	o := kernelmodel.DataOptions()
	o.CatalogOverlays = []string{root}
	return load.Open(o)
}

// mustDataWithStorageDevices is dataWithStorageDevices for a table that must load.
func mustDataWithStorageDevices(t *testing.T, body string) *load.Data {
	t.Helper()
	d, err := dataWithStorageDevices(t, body)
	if err != nil {
		t.Fatalf("the device table must load: %v", err)
	}
	return d
}

// defaultData is the run's default data: the pinned substrate's embedded copy.
func defaultData(t *testing.T) *load.Data {
	t.Helper()
	d, err := kernelmodel.DefaultData()
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// ---------------------------------------------------------------------------
// BC-1: the table lives at <catalog>/devices/storage.yaml
// ---------------------------------------------------------------------------

// TestBundledCatalogDevices_NameTheClassesAndStateNoRate: the bundled catalog names exactly the
// tier classes runs resolve, and a class resolves to a name alone, with no rate, since the kernel
// prices every catalog-class tier from the registry. A class that carried a rate here would be a
// second source for a figure the registry owns.
func TestBundledCatalogDevices_NameTheClassesAndStateNoRate(t *testing.T) {
	got, err := StorageDevices(defaultData(t))
	if err != nil {
		t.Fatalf("the substrate catalog's device table must load: %v", err)
	}
	var names []string
	for n, d := range got {
		names = append(names, n)
		if !reflect.DeepEqual(d, KVOffloadDevice{}) {
			t.Errorf("class %q resolved to %+v; a catalog class states no rate", n, d)
		}
	}
	sort.Strings(names)
	if !reflect.DeepEqual(names, catalogDeviceClasses) {
		t.Errorf("the catalog names the classes %v, want %v", names, catalogDeviceClasses)
	}
}

// TestResolveKVOffload_CatalogClassResolvesToANameForTheKernelToPrice: a tier naming each class
// resolves to that class with the device model off and no rate of its own. The price comes from
// the kernel (priceOffload), which TestPoolsPriceOffload reaches.
func TestResolveKVOffload_CatalogClassResolvesToANameForTheKernelToPrice(t *testing.T) {
	devices, err := StorageDevices(defaultData(t))
	if err != nil {
		t.Fatalf("the substrate catalog's device table must load: %v", err)
	}
	for _, class := range catalogDeviceClasses {
		cfg, err := ResolveKVOffload(rampBlock(class, true), devices, 16)
		if err != nil {
			t.Fatalf("class %q must resolve: %v", class, err)
		}
		tr := cfg.Tiers[0]
		if tr.DeviceClass != class {
			t.Errorf("class %q: resolved device_class = %q", class, tr.DeviceClass)
		}
		if tr.ReadBandwidth != 0 || tr.WriteBandwidth != 0 || tr.BaseLatency != 0 {
			t.Errorf("class %q: resolved rates (%v, %v, %v), want none: the kernel prices the tier",
				class, tr.ReadBandwidth, tr.WriteBandwidth, tr.BaseLatency)
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

// TestAnUnusableDeviceTableIsRefused: every way the table can fail to answer is a refusal,
// never an empty or zero-valued table. The alternatives are both silent mis-models: an empty
// map degrades into "device_class X is not defined (known: <none configured>)", which blames
// the operator's config for a missing catalog file, and a zero-valued device would give
// plausible-but-wrong physics. The loader refuses a malformed table when the catalog opens;
// StorageDevices refuses a catalog with none, and data that is absent.
func TestAnUnusableDeviceTableIsRefused(t *testing.T) {
	for _, tc := range []struct{ name, body, wantFrag string }{
		{"no classes", "{}\n", "storage tier"},
		{"malformed yaml", "nvme_gen4: {name: [unclosed\n", "devices/storage.yaml"},
		// R10: a misspelled key must be refused, never decoded to nothing.
		{"unknown key", "nvme_gen4: {nmae: nvme_gen4}\n", "nmae"},
		// A table that still states a rate is refused, and told where the value goes: the rates
		// are registry coefficients, so a second copy in the catalog would be a second source.
		{"a rate in the catalog", "nvme_gen4: {read_bandwidth_mb_s: 7.0e3, write_bandwidth_mb_s: 5.0e3, base_latency_us: 80.0}\n", "offload_tier_<tier>_read_rate"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, err := dataWithStorageDevices(t, tc.body)
			if err == nil {
				var devices map[string]KVOffloadDevice
				devices, err = StorageDevices(d)
				if err == nil {
					t.Fatalf("expected refusal, got %d classes", len(devices))
				}
			}
			if !strings.Contains(err.Error(), tc.wantFrag) {
				t.Errorf("refusal must identify the failure (%q), got: %v", tc.wantFrag, err)
			}
		})
	}
	if _, err := StorageDevices(nil); err == nil || !strings.Contains(err.Error(), CatalogStorageDevicesRelPath) {
		t.Errorf("no data: %v, want a refusal naming %s", err, CatalogStorageDevicesRelPath)
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

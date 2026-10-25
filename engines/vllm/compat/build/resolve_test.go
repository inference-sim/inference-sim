// Original to blis-vllm.

package build

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// The device table is a lazy dependency: a block whose tiers all state their physics never
// asks for the catalog, and the same block naming a device_class does (non-vacuity). The
// locator records whether it was asked, which is what cmd's catalog resolver cost before the
// move (it warns on a --catalog/BLIS_CATALOG conflict each time it runs).
func TestResolveKVOffloadDevices_AsksForTheCatalogOnlyWhenATierNamesADevice(t *testing.T) {
	asked := 0
	noCatalog := func() (string, error) { asked++; return "", errors.New("no catalog located") }

	explicit := &KVOffloadBlock{
		CPUBytesToUse: i64p(1 << 30),
		SecondaryTiers: []KVOffloadTierBlock{{
			Type: strp("fs"), RootDir: strp("/mnt/kv"), DirectIO: boolp(true),
			ReadBandwidth: f64p(7000), WriteBandwidth: f64p(5000), BaseLatency: f64p(80),
		}},
	}
	devices, err := ResolveKVOffloadDevices(explicit, noCatalog)
	if err != nil || devices != nil || asked != 0 {
		t.Fatalf("explicit physics: devices %v, err %v, catalog asked %d times; want none, nil, 0", devices, err, asked)
	}

	withClass := *explicit
	withClass.SecondaryTiers = append([]KVOffloadTierBlock(nil), explicit.SecondaryTiers...)
	withClass.SecondaryTiers[0].DeviceClass = strp("nvme_gen4")
	_, err = ResolveKVOffloadDevices(&withClass, noCatalog)
	if err == nil || asked != 1 {
		t.Fatalf("a device_class with no catalog located: err %v, asked %d; want a refusal after one ask", err, asked)
	}
	for _, frag := range []string{"device_class", CatalogStorageDevicesRelPath, "no catalog located"} {
		if !strings.Contains(err.Error(), frag) {
			t.Errorf("refusal %q must name %q", err, frag)
		}
	}
}

// A --kv-offload-config tier that states its own bandwidth or latency is refused: the kernel
// prices a tier from its catalog device, so a second source for the same physics is not
// taken. Each of the three fields alone is enough.
func TestResolveKVOffloadFile_RefusesExplicitTierPhysics(t *testing.T) {
	catalog := writeCatalogStorageDevices(t,
		"nvme_gen4: {read_bandwidth_mb_s: 7.0e3, write_bandwidth_mb_s: 5.0e3, base_latency_us: 80.0}\n")
	locate := func() (string, error) { return catalog, nil }
	tier := "kv_offload:\n  cpu_bytes_to_use: 1073741824\n  secondary_tiers:\n" +
		"    - type: fs\n      root_dir: /mnt/kv\n      direct_io: true\n      device_class: nvme_gen4\n"
	write := func(extra string) string {
		p := filepath.Join(t.TempDir(), "offload.yaml")
		if err := os.WriteFile(p, []byte(tier+extra), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	if _, err := ResolveKVOffloadFile(write(""), 16, locate); err != nil {
		t.Fatalf("a tier naming only its device must resolve: %v", err)
	}
	for _, extra := range []string{"      read_bandwidth: 1\n", "      write_bandwidth: 1\n", "      base_latency: 1\n"} {
		_, err := ResolveKVOffloadFile(write(extra), 16, locate)
		if err == nil || !strings.Contains(err.Error(), "drop the explicit physics") {
			t.Errorf("tier with %q: err %v, want the explicit-physics refusal", strings.TrimSpace(extra), err)
		}
	}
}

// The two sources agree or the run is refused: the scenario's offload block and a
// --kv-offload-config restating it resolve to one config; a file that differs is refused
// naming the field, and with neither the config is inert.
func TestResolveKVOffloadConfig_TheScenarioAndTheFileMustAgree(t *testing.T) {
	locate := func() (string, error) {
		t.Fatal("no tier names a device_class; the catalog must not be asked")
		return "", nil
	}
	scenario := &deployment.Offload{Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1 << 30}}}
	write := func(body string) string {
		p := filepath.Join(t.TempDir(), "offload.yaml")
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	same := write("kv_offload:\n  cpu_bytes_to_use: 1073741824\n")
	other := write("kv_offload:\n  cpu_bytes_to_use: 2147483648\n")

	inert, err := ResolveKVOffloadConfig("s.yaml", nil, "", 16, locate)
	if err != nil || inert.IsEnabled() {
		t.Fatalf("neither source: %+v, %v; want the inert config", inert, err)
	}
	fromScenario, err := ResolveKVOffloadConfig("s.yaml", scenario, "", 16, locate)
	if err != nil || !fromScenario.IsEnabled() || fromScenario.CPUBytesToUse != 1<<30 {
		t.Fatalf("scenario only: %+v, %v", fromScenario, err)
	}
	both, err := ResolveKVOffloadConfig("s.yaml", scenario, same, 16, locate)
	if err != nil || OffloadConfigDiff(both, fromScenario) != "" {
		t.Fatalf("a restating file: %+v, %v; want the scenario's config", both, err)
	}
	if _, err := ResolveKVOffloadConfig("s.yaml", scenario, other, 16, locate); err == nil ||
		!strings.Contains(err.Error(), "CPUBytesToUse") || !strings.Contains(err.Error(), `"s.yaml"`) {
		t.Errorf("a disagreeing file: err %v, want a refusal naming the field and the scenario", err)
	}
}

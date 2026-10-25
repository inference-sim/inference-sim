// Moved from inference-sim/cmd/kv_offload_device_test.go at ace1528fe4bba59e510eda48e80ee9b8142ea982.

package build

import (
	"strings"
	"testing"
)

// BC-D7: the #1581 device-model fields have no catalog source. blis-schemas owns the
// devices/storage.yaml format, its StorageDevice does not define them, and its strict
// loader refuses a key it does not define -- so a catalog stating them is refused naming
// them, rather than half-read with the ramp silently dropped. The resolver half of the
// device model is exercised from a device map below, which is the only way it is reached.
func TestKVOffloadDevices_DeviceModelFieldsAreNotACatalogFormat(t *testing.T) {
	catalog := writeCatalogStorageDevices(t,
		"nvme_dm:\n"+
			"  read_bandwidth_mb_s: 7.0e3\n"+
			"  write_bandwidth_mb_s: 5.0e3\n"+
			"  base_latency_us: 80.0\n"+
			"  saturation_queue_depth: 8\n")
	devices, err := LoadCatalogStorageDevices(catalog)
	if err == nil {
		t.Fatalf("a catalog table stating saturation_queue_depth loaded (%d classes); the "+
			"catalog format does not define it", len(devices))
	}
	if !strings.Contains(err.Error(), "saturation_queue_depth") {
		t.Errorf("the refusal must name the undefined key, got: %v", err)
	}
}

// deviceModelDevices is a synthetic device map exercising the #1581 device model:
// nvme_ramp declares an O_DIRECT ramp + jitter and a fully-specified buffered
// regime; nvme_partial declares an O_DIRECT ramp but NO buffered fields (buffered
// must fall back to the O_DIRECT values).
func deviceModelDevices() map[string]KVOffloadDevice {
	return map[string]KVOffloadDevice{
		"nvme_ramp": {
			ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			SaturationQueueDepth:           i64p(8),
			SingleTransferFraction:         f64p(0.4),
			LatencyJitterStddev:            f64p(0.10),
			BufferedReadBandwidth:          f64p(4000),
			BufferedWriteBandwidth:         f64p(3000),
			BufferedBaseLatency:            f64p(120),
			BufferedSaturationQueueDepth:   i64p(4),
			BufferedSingleTransferFraction: f64p(0.6),
			BufferedLatencyJitterStddev:    f64p(0.25),
		},
		"nvme_partial": {
			ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			SaturationQueueDepth:   i64p(8),
			SingleTransferFraction: f64p(0.4),
			LatencyJitterStddev:    f64p(0.10),
		},
	}
}

func rampBlock(deviceClass string, directIO bool) *KVOffloadBlock {
	return &KVOffloadBlock{
		CPUBytesToUse: i64p(17179869184),
		SecondaryTiers: []KVOffloadTierBlock{{
			Type:        strp("fs"),
			RootDir:     strp("/mnt/kv"),
			DirectIO:    boolp(directIO),
			DeviceClass: strp(deviceClass),
		}},
	}
}

// BC-D7: a device with no device-model fields resolves to the ramp/jitter-off
// defaults (Qsat=1, f₁=1.0, σ=0) — byte-identical to pre-#1581.
func TestResolveKVOffload_DeviceModelDefaults(t *testing.T) {
	cfg, err := ResolveKVOffload(validBlock(), testDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	tr := cfg.Tiers[0]
	if tr.SaturationQueueDepth != 1 || tr.SingleTransferFraction != 1.0 || tr.LatencyJitterStddev != 0 {
		t.Errorf("device-model defaults must be off: Qsat=%d f1=%v sigma=%v", tr.SaturationQueueDepth, tr.SingleTransferFraction, tr.LatencyJitterStddev)
	}
}

// BC-D3: direct_io selects the O_DIRECT regime; the resolved ramp/jitter come from
// the O_DIRECT device fields.
func TestResolveKVOffload_ODirectRegimeSelected(t *testing.T) {
	cfg, err := ResolveKVOffload(rampBlock("nvme_ramp", true), deviceModelDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	tr := cfg.Tiers[0]
	if tr.ReadBandwidth != 7000 || tr.WriteBandwidth != 5000 || tr.BaseLatency != 80 {
		t.Errorf("O_DIRECT triple wrong: %+v", tr)
	}
	if tr.SaturationQueueDepth != 8 || tr.SingleTransferFraction != 0.4 || tr.LatencyJitterStddev != 0.10 {
		t.Errorf("O_DIRECT device model wrong: Qsat=%d f1=%v sigma=%v", tr.SaturationQueueDepth, tr.SingleTransferFraction, tr.LatencyJitterStddev)
	}
}

// BC-D3: direct_io=false selects the buffered regime, distinct from O_DIRECT.
func TestResolveKVOffload_BufferedRegimeSelected(t *testing.T) {
	cfg, err := ResolveKVOffload(rampBlock("nvme_ramp", false), deviceModelDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	tr := cfg.Tiers[0]
	if tr.ReadBandwidth != 4000 || tr.WriteBandwidth != 3000 || tr.BaseLatency != 120 {
		t.Errorf("buffered triple wrong: %+v", tr)
	}
	if tr.SaturationQueueDepth != 4 || tr.SingleTransferFraction != 0.6 || tr.LatencyJitterStddev != 0.25 {
		t.Errorf("buffered device model wrong: Qsat=%d f1=%v sigma=%v", tr.SaturationQueueDepth, tr.SingleTransferFraction, tr.LatencyJitterStddev)
	}
}

// BC-D3: when the device declares no buffered fields, the buffered regime falls
// back to the O_DIRECT values (identical resolution).
func TestResolveKVOffload_BufferedFallsBackToDirect(t *testing.T) {
	direct, err := ResolveKVOffload(rampBlock("nvme_partial", true), deviceModelDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	buffered, err := ResolveKVOffload(rampBlock("nvme_partial", false), deviceModelDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	d, b := direct.Tiers[0], buffered.Tiers[0]
	if d.ReadBandwidth != b.ReadBandwidth || d.WriteBandwidth != b.WriteBandwidth ||
		d.BaseLatency != b.BaseLatency || d.SaturationQueueDepth != b.SaturationQueueDepth ||
		d.SingleTransferFraction != b.SingleTransferFraction || d.LatencyJitterStddev != b.LatencyJitterStddev {
		t.Errorf("buffered must fall back to O_DIRECT: direct=%+v buffered=%+v", d, b)
	}
}

// BC-D7: a device-class ramp with an out-of-range single_transfer_fraction is
// rejected by post-resolve validation, naming the field.
func TestResolveKVOffload_DeviceModelRejects(t *testing.T) {
	bad := map[string]KVOffloadDevice{
		"nvme_badf1": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			SaturationQueueDepth: i64p(4), SingleTransferFraction: f64p(0)},
		"nvme_badqsat": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			SaturationQueueDepth: i64p(-2)},
		"nvme_badjitter": {ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			LatencyJitterStddev: f64p(-0.5)},
	}
	cases := []struct{ class, frag string }{
		{"nvme_badf1", "single_transfer_fraction"},
		{"nvme_badqsat", "saturation_queue_depth"},
		{"nvme_badjitter", "latency_jitter_stddev"},
	}
	for _, tc := range cases {
		_, err := ResolveKVOffload(rampBlock(tc.class, true), bad, 16)
		if err == nil {
			t.Fatalf("%s: expected resolve error, got nil", tc.class)
		}
		if !strings.Contains(err.Error(), tc.frag) {
			t.Errorf("%s: error %q must name %q", tc.class, err.Error(), tc.frag)
		}
	}
}

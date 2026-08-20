package cmd

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// BC-D7: a defaults.yaml device block carrying the #1581 device-model fields parses
// under strict YAML (KnownFields), and a tier resolved against it (direct_io=false)
// picks up the buffered regime + ramp + jitter — the full YAML→map→resolver path.
func TestKVOffloadDevices_DeviceModelYAMLParses(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "defaults.yaml")
	content := "version: 0.0.1\n" +
		"kv_offload_devices:\n" +
		"  nvme_dm:\n" +
		"    read_bandwidth: 7.0e3\n" +
		"    write_bandwidth: 5.0e3\n" +
		"    base_latency: 80.0\n" +
		"    saturation_queue_depth: 8\n" +
		"    single_transfer_fraction: 0.4\n" +
		"    latency_jitter_stddev: 0.1\n" +
		"    buffered_read_bandwidth: 4.0e3\n" +
		"    buffered_base_latency: 120.0\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	devices := loadDefaultsConfig(path).KVOffloadDevices
	dev, ok := devices["nvme_dm"]
	if !ok {
		t.Fatalf("nvme_dm device class must parse from defaults.yaml")
	}
	if dev.SaturationQueueDepth == nil || *dev.SaturationQueueDepth != 8 {
		t.Errorf("saturation_queue_depth did not parse: %v", dev.SaturationQueueDepth)
	}
	// Buffered regime: buffered_read_bandwidth + buffered_base_latency explicit,
	// buffered_write_bandwidth absent => falls back to O_DIRECT write_bandwidth 5000.
	buffered, err := resolveKVOffload(rampBlock("nvme_dm", false), devices, 16)
	if err != nil {
		t.Fatal(err)
	}
	tr := buffered.Tiers[0]
	if tr.ReadBandwidth != 4000 || tr.WriteBandwidth != 5000 || tr.BaseLatency != 120 {
		t.Errorf("buffered regime resolve wrong: %+v", tr)
	}
	if tr.SaturationQueueDepth != 8 || tr.SingleTransferFraction != 0.4 || tr.LatencyJitterStddev != 0.1 {
		t.Errorf("buffered inherited O_DIRECT ramp/jitter wrong: Qsat=%d f1=%v sigma=%v", tr.SaturationQueueDepth, tr.SingleTransferFraction, tr.LatencyJitterStddev)
	}
}

// deviceModelDevices is a synthetic device map exercising the #1581 device model:
// nvme_ramp declares an O_DIRECT ramp + jitter and a fully-specified buffered
// regime; nvme_partial declares an O_DIRECT ramp but NO buffered fields (buffered
// must fall back to the O_DIRECT values).
func deviceModelDevices() map[string]KVOffloadDeviceDefaults {
	return map[string]KVOffloadDeviceDefaults{
		"nvme_ramp": {
			ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			SaturationQueueDepth:   i64p(8),
			SingleTransferFraction: f64p(0.4),
			LatencyJitterStddev:    f64p(0.10),
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

func rampBlock(deviceClass string, directIO bool) *kvOffloadBlock {
	return &kvOffloadBlock{
		CPUBytesToUse: i64p(17179869184),
		SecondaryTiers: []kvOffloadTierBlock{{
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
	cfg, err := resolveKVOffload(validBlock(), testDevices(), 16)
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
	cfg, err := resolveKVOffload(rampBlock("nvme_ramp", true), deviceModelDevices(), 16)
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
	cfg, err := resolveKVOffload(rampBlock("nvme_ramp", false), deviceModelDevices(), 16)
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
	direct, err := resolveKVOffload(rampBlock("nvme_partial", true), deviceModelDevices(), 16)
	if err != nil {
		t.Fatal(err)
	}
	buffered, err := resolveKVOffload(rampBlock("nvme_partial", false), deviceModelDevices(), 16)
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

// BC-D6/BC-G6: the resolved device-model fields (non-default Qsat/f₁/σ) round-trip
// losslessly through the trace-header converters, for both regimes.
func TestKVOffloadHeaderConversion_DeviceModelRoundTrip(t *testing.T) {
	for _, directIO := range []bool{true, false} {
		cfg, err := resolveKVOffload(rampBlock("nvme_ramp", directIO), deviceModelDevices(), 16)
		if err != nil {
			t.Fatal(err)
		}
		back := headerToSimOffload(simToHeaderOffload(cfg))
		if !reflect.DeepEqual(cfg, back) {
			t.Errorf("directIO=%v: device-model round-trip not identity:\n got  %+v\n want %+v", directIO, back.Tiers[0], cfg.Tiers[0])
		}
	}
}

// BC-D7: a device-class ramp with an out-of-range single_transfer_fraction is
// rejected by post-resolve validation, naming the field.
func TestResolveKVOffload_DeviceModelRejects(t *testing.T) {
	bad := map[string]KVOffloadDeviceDefaults{
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
		_, err := resolveKVOffload(rampBlock(tc.class, true), bad, 16)
		if err == nil {
			t.Fatalf("%s: expected resolve error, got nil", tc.class)
		}
		if !strings.Contains(err.Error(), tc.frag) {
			t.Errorf("%s: error %q must name %q", tc.class, err.Error(), tc.frag)
		}
	}
}

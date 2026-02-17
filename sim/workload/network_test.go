package workload

import (
	"math"
	"testing"
)

func TestComputeClientTTFT_RTTOnly(t *testing.T) {
	// BC-9: RTT adds to TTFT
	net := &NetworkSpec{RTTMs: 10.0} // 10ms = 10000µs
	clientTTFT := ComputeClientTTFT(500, net, 100)
	// RTT only: client_ttft = server_ttft + rtt_us = 500 + 10000
	if clientTTFT != 10500 {
		t.Errorf("client TTFT = %.0f, want 10500", clientTTFT)
	}
}

func TestComputeClientTTFT_WithBandwidth(t *testing.T) {
	net := &NetworkSpec{RTTMs: 0, BandwidthMbps: 100} // 100 Mbps
	inputTokens := 1000 // 1000 tokens × 4 bytes = 4000 bytes = 32000 bits
	clientTTFT := ComputeClientTTFT(500, net, inputTokens)
	// upload_delay = 32000 bits / 100e6 bps = 0.00032s = 320µs
	expectedUpload := float64(32000) / float64(100e6) * 1e6
	expected := 500 + expectedUpload
	if math.Abs(clientTTFT-expected) > 1 {
		t.Errorf("client TTFT = %.1f, want ≈ %.1f (with upload delay)", clientTTFT, expected)
	}
}

func TestComputeClientTTFT_NilNetwork(t *testing.T) {
	clientTTFT := ComputeClientTTFT(500, nil, 100)
	if clientTTFT != 500 {
		t.Errorf("nil network should not adjust: got %.0f, want 500", clientTTFT)
	}
}

func TestComputeClientE2E_IncludesDownload(t *testing.T) {
	net := &NetworkSpec{RTTMs: 5.0, BandwidthMbps: 100}
	clientE2E := ComputeClientE2E(1000, net, 500, 200)
	// RTT = 5000µs, upload = 500*4*8/(100e6)*1e6 = 160µs, download = 200*4*8/(100e6)*1e6 = 64µs
	rtt := 5000.0
	upload := float64(500*4*8) / (100e6) * 1e6
	download := float64(200*4*8) / (100e6) * 1e6
	expected := 1000 + rtt + upload + download
	if math.Abs(clientE2E-expected) > 1 {
		t.Errorf("client E2E = %.1f, want ≈ %.1f", clientE2E, expected)
	}
}

func TestComputeClientE2E_ZeroBandwidth_NoDelay(t *testing.T) {
	net := &NetworkSpec{RTTMs: 2.0, BandwidthMbps: 0} // infinite bandwidth
	clientE2E := ComputeClientE2E(1000, net, 500, 200)
	// Only RTT, no bandwidth delay
	expected := 1000.0 + 2000.0 // server + RTT in µs
	if math.Abs(clientE2E-expected) > 1 {
		t.Errorf("client E2E = %.1f, want %.1f (RTT only, no bandwidth delay)", clientE2E, expected)
	}
}

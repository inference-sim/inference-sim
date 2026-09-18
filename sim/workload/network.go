package workload

// ComputeClientTTFT computes client-perspective TTFT from server-side TTFT.
// Adds RTT and upload delay based on network config.
// If network is nil, returns server TTFT unchanged.
func ComputeClientTTFT(serverTTFT float64, network *NetworkSpec, inputTokens int) float64 {
	if network == nil {
		return serverTTFT
	}
	rttUs := network.RTTMs * 1000.0 // ms to µs
	uploadDelay := computeUploadDelay(network.BandwidthMbps, inputTokens)
	return serverTTFT + rttUs + uploadDelay
}

// ComputeClientE2E computes client-perspective E2E from server-side E2E.
// Adds RTT, upload delay, and download delay based on network config.
func ComputeClientE2E(serverE2E float64, network *NetworkSpec, inputTokens, outputTokens int) float64 {
	if network == nil {
		return serverE2E
	}
	rttUs := network.RTTMs * 1000.0
	uploadDelay := computeUploadDelay(network.BandwidthMbps, inputTokens)
	downloadDelay := computeDownloadDelay(network.BandwidthMbps, outputTokens)
	return serverE2E + rttUs + uploadDelay + downloadDelay
}

// computeUploadDelay computes upload delay in µs for input tokens.
// Each token ≈ 4 bytes (int32 token ID).
func computeUploadDelay(bandwidthMbps float64, tokenCount int) float64 {
	if bandwidthMbps <= 0 {
		return 0
	}
	inputBytes := float64(tokenCount * 4)
	inputBits := inputBytes * 8
	return inputBits / (bandwidthMbps * 1e6) * 1e6 // seconds → µs
}

// computeDownloadDelay computes download delay in µs for output tokens.
func computeDownloadDelay(bandwidthMbps float64, tokenCount int) float64 {
	if bandwidthMbps <= 0 {
		return 0
	}
	outputBytes := float64(tokenCount * 4)
	outputBits := outputBytes * 8
	return outputBits / (bandwidthMbps * 1e6) * 1e6
}

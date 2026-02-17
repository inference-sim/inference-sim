package workload

import (
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
)

// GenerateMultimodalTokens generates input tokens for a multimodal request.
// Returns the combined input tokens and per-modality counts.
// Total input = text + image*imageCount + audio*audioCount + video*videoCount.
func GenerateMultimodalTokens(rng *rand.Rand, mm *MultimodalSpec) ([]int, int, int, int, int) {
	textLen := 0
	imageLen := 0
	audioLen := 0
	videoLen := 0

	// Text tokens
	if mm.TextDist.Type != "" {
		sampler, err := NewLengthSampler(mm.TextDist)
		if err == nil {
			textLen = sampler.Sample(rng)
		}
	}

	// Image tokens (count × per-image tokens)
	if mm.ImageDist.Type != "" && mm.ImageCountDist.Type != "" {
		countSampler, err1 := NewLengthSampler(mm.ImageCountDist)
		tokenSampler, err2 := NewLengthSampler(mm.ImageDist)
		if err1 == nil && err2 == nil {
			count := countSampler.Sample(rng)
			for i := 0; i < count; i++ {
				imageLen += tokenSampler.Sample(rng)
			}
		}
	}

	// Audio tokens
	if mm.AudioDist.Type != "" && mm.AudioCountDist.Type != "" {
		countSampler, err1 := NewLengthSampler(mm.AudioCountDist)
		tokenSampler, err2 := NewLengthSampler(mm.AudioDist)
		if err1 == nil && err2 == nil {
			count := countSampler.Sample(rng)
			for i := 0; i < count; i++ {
				audioLen += tokenSampler.Sample(rng)
			}
		}
	}

	// Video tokens
	if mm.VideoDist.Type != "" && mm.VideoCountDist.Type != "" {
		countSampler, err1 := NewLengthSampler(mm.VideoCountDist)
		tokenSampler, err2 := NewLengthSampler(mm.VideoDist)
		if err1 == nil && err2 == nil {
			count := countSampler.Sample(rng)
			for i := 0; i < count; i++ {
				videoLen += tokenSampler.Sample(rng)
			}
		}
	}

	// Generate combined token IDs
	totalLen := textLen + imageLen + audioLen + videoLen
	if totalLen == 0 {
		totalLen = 1 // minimum 1 token
	}
	tokens := sim.GenerateRandomTokenIDs(rng, totalLen)

	return tokens, textLen, imageLen, audioLen, videoLen
}

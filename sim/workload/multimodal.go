package workload

import (
	"fmt"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
)

// GenerateMultimodalTokens generates input tokens for a multimodal request.
// Returns the combined input tokens, per-modality counts, and any error.
// Total input = text + image*imageCount + audio*audioCount + video*videoCount.
func GenerateMultimodalTokens(rng *rand.Rand, mm *MultimodalSpec) ([]int, int, int, int, int, error) {
	textLen := 0
	imageLen := 0
	audioLen := 0
	videoLen := 0

	// Text tokens
	if mm.TextDist.Type != "" {
		sampler, err := NewLengthSampler(mm.TextDist)
		if err != nil {
			return nil, 0, 0, 0, 0, fmt.Errorf("text distribution: %w", err)
		}
		textLen = sampler.Sample(rng)
	}

	// Image tokens (count × per-image tokens)
	if mm.ImageDist.Type != "" && mm.ImageCountDist.Type != "" {
		countSampler, err := NewLengthSampler(mm.ImageCountDist)
		if err != nil {
			return nil, 0, 0, 0, 0, fmt.Errorf("image count distribution: %w", err)
		}
		tokenSampler, err := NewLengthSampler(mm.ImageDist)
		if err != nil {
			return nil, 0, 0, 0, 0, fmt.Errorf("image distribution: %w", err)
		}
		count := countSampler.Sample(rng)
		for i := 0; i < count; i++ {
			imageLen += tokenSampler.Sample(rng)
		}
	}

	// Audio tokens
	if mm.AudioDist.Type != "" && mm.AudioCountDist.Type != "" {
		countSampler, err := NewLengthSampler(mm.AudioCountDist)
		if err != nil {
			return nil, 0, 0, 0, 0, fmt.Errorf("audio count distribution: %w", err)
		}
		tokenSampler, err := NewLengthSampler(mm.AudioDist)
		if err != nil {
			return nil, 0, 0, 0, 0, fmt.Errorf("audio distribution: %w", err)
		}
		count := countSampler.Sample(rng)
		for i := 0; i < count; i++ {
			audioLen += tokenSampler.Sample(rng)
		}
	}

	// Video tokens
	if mm.VideoDist.Type != "" && mm.VideoCountDist.Type != "" {
		countSampler, err := NewLengthSampler(mm.VideoCountDist)
		if err != nil {
			return nil, 0, 0, 0, 0, fmt.Errorf("video count distribution: %w", err)
		}
		tokenSampler, err := NewLengthSampler(mm.VideoDist)
		if err != nil {
			return nil, 0, 0, 0, 0, fmt.Errorf("video distribution: %w", err)
		}
		count := countSampler.Sample(rng)
		for i := 0; i < count; i++ {
			videoLen += tokenSampler.Sample(rng)
		}
	}

	// Generate combined token IDs
	totalLen := textLen + imageLen + audioLen + videoLen
	if totalLen == 0 {
		totalLen = 1 // minimum 1 token
	}
	tokens := sim.GenerateRandomTokenIDs(rng, totalLen)

	return tokens, textLen, imageLen, audioLen, videoLen, nil
}

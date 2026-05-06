package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

func parseServeGenFloatPDF(s string) (map[float64]float64, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == "{}" {
		return nil, fmt.Errorf("empty PDF string")
	}
	s = strings.TrimPrefix(s, "{")
	s = strings.TrimSuffix(s, "}")
	
	result := make(map[float64]float64)
	pairs := strings.Split(s, ",")
	for _, pair := range pairs {
		parts := strings.Split(strings.TrimSpace(pair), ":")
		if len(parts) != 2 {
			continue
		}
		keyStr := strings.TrimSpace(parts[0])
		keyStr = strings.Trim(keyStr, "\"")
		valStr := strings.TrimSpace(parts[1])
		
		key, err := strconv.ParseFloat(keyStr, 64)
		if err != nil {
			continue
		}
		if key < 0.0 || key > 1.0 {
			return nil, fmt.Errorf("ratio key %f out of range [0.0, 1.0]", key)
		}
		
		val, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			continue
		}
		result[key] = val
	}
	return result, nil
}

func main() {
	dataDir := "../../ServeGen/data/reason/deepseek-r1"
	
	traceFiles, _ := filepath.Glob(filepath.Join(dataDir, "chunk-*-trace.csv"))
	sort.Strings(traceFiles)
	
	var lowCount, highCount int
	threshold := 0.4
	
	for _, tracePath := range traceFiles {
		base := filepath.Base(tracePath)
		chunkID := strings.TrimPrefix(base, "chunk-")
		chunkID = strings.TrimSuffix(chunkID, "-trace.csv")
		datasetPath := filepath.Join(dataDir, fmt.Sprintf("chunk-%s-dataset.json", chunkID))
		
		data, err := os.ReadFile(datasetPath)
		if err != nil {
			continue
		}
		
		var raw map[string]map[string]string
		if err := json.Unmarshal(data, &raw); err != nil {
			continue
		}
		
		// Find first valid reason_ratio
		var reasonRatioPDF map[float64]float64
		timestamps := make([]string, 0, len(raw))
		for ts := range raw {
			timestamps = append(timestamps, ts)
		}
		sort.Strings(timestamps)
		
		for _, ts := range timestamps {
			window := raw[ts]
			if ratioStr, ok := window["reason_ratio"]; ok && ratioStr != "" && ratioStr != "{}" {
				parsed, parseErr := parseServeGenFloatPDF(ratioStr)
				if parseErr == nil {
					reasonRatioPDF = parsed
					break
				}
			}
		}
		
		if len(reasonRatioPDF) == 0 {
			continue
		}
		
		// Compute weighted mean
		var sumRatio, sumProb float64
		ratios := make([]float64, 0, len(reasonRatioPDF))
		for ratio := range reasonRatioPDF {
			ratios = append(ratios, ratio)
		}
		sort.Float64s(ratios)
		
		for _, ratio := range ratios {
			prob := reasonRatioPDF[ratio]
			if prob > 0 {
				sumRatio += ratio * prob
				sumProb += prob
			}
		}
		
		if sumProb <= 0 {
			continue
		}
		
		meanRatio := sumRatio / sumProb
		
		if meanRatio < threshold {
			lowCount++
			fmt.Printf("LOW  chunk-%s: mean_ratio=%.4f\n", chunkID, meanRatio)
		} else {
			highCount++
			fmt.Printf("HIGH chunk-%s: mean_ratio=%.4f\n", chunkID, meanRatio)
		}
	}
	
	fmt.Printf("\n=== SUMMARY ===\n")
	fmt.Printf("Total chunks: %d\n", lowCount+highCount)
	fmt.Printf("Low reasoning (< %.1f): %d (%.1f%%)\n", threshold, lowCount, 100*float64(lowCount)/float64(lowCount+highCount))
	fmt.Printf("High reasoning (>= %.1f): %d (%.1f%%)\n", threshold, highCount, 100*float64(highCount)/float64(lowCount+highCount))
}

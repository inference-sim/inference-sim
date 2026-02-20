package sim

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/sirupsen/logrus"
)

// MHAPrefillRow represents one row from prefill CSV: dtype,seq_len,latency_us,mfu
type MHAPrefillRow struct {
	SeqLen int
	MFU    float64
}

// MHADecodeRow represents one row from decode CSV: dtype,kv_dtype,batch_size,kv_len,latency_us,mfu
type MHADecodeRow struct {
	BatchSize int
	KVLen     int
	MFU       float64
}

// GEMMRow represents one row from GEMM CSV: m,k,n,latency_us,mfu
type GEMMRow struct {
	M   int
	K   int
	N   int
	MFU float64
}

// AttentionShape represents an attention configuration
type AttentionShape struct {
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	ConfigKey  string // "32-32-128"
}

// MFUDatabase holds all MFU benchmark data
type MFUDatabase struct {
	prefillData      map[string][]MHAPrefillRow // key: "32-32-128"
	decodeData       map[string][]MHADecodeRow  // key: "32-32-128-tp2"
	gemmData         []GEMMRow
	attentionConfig  string // Model's attention config
	availableConfigs []AttentionShape
	gpu              string
}

// computeAttentionConfig derives the attention config string from model config
func computeAttentionConfig(config ModelConfig) string {
	headDim := config.HiddenDim / config.NumHeads
	numKVHeads := config.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = config.NumHeads
	}
	return fmt.Sprintf("%d-%d-%d", config.NumHeads, numKVHeads, headDim)
}

// parseAttentionConfig parses "32-32-128" into AttentionShape
func parseAttentionConfig(configKey string) AttentionShape {
	var nh, nkv, hd int
	fmt.Sscanf(configKey, "%d-%d-%d", &nh, &nkv, &hd)
	return AttentionShape{
		NumHeads:   nh,
		NumKVHeads: nkv,
		HeadDim:    hd,
		ConfigKey:  configKey,
	}
}

// euclideanDistance computes distance between two attention shapes
func euclideanDistance(a, b AttentionShape) float64 {
	dh := float64(a.NumHeads - b.NumHeads)
	dkv := float64(a.NumKVHeads - b.NumKVHeads)
	dhd := float64(a.HeadDim - b.HeadDim)
	return math.Sqrt(dh*dh + dkv*dkv + dhd*dhd)
}

// loadPrefillCSV loads a single prefill CSV file
func loadPrefillCSV(path string) ([]MHAPrefillRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open prefill CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read prefill CSV: %w", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("prefill CSV empty or missing header")
	}

	var rows []MHAPrefillRow
	for i, record := range records[1:] { // Skip header
		if len(record) < 4 {
			return nil, fmt.Errorf("prefill CSV row %d: expected 4 columns", i+2)
		}

		seqLen, err := strconv.Atoi(record[1])
		if err != nil {
			return nil, fmt.Errorf("prefill CSV row %d: invalid seq_len: %w", i+2, err)
		}

		mfu, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			return nil, fmt.Errorf("prefill CSV row %d: invalid mfu: %w", i+2, err)
		}

		rows = append(rows, MHAPrefillRow{
			SeqLen: seqLen,
			MFU:    mfu,
		})
	}

	return rows, nil
}

// loadAllPrefillCSVs loads all prefill CSV files for the GPU
func loadAllPrefillCSVs(basePath, gpu string) (map[string][]MHAPrefillRow, []AttentionShape, error) {
	data := make(map[string][]MHAPrefillRow)
	var configs []AttentionShape

	pattern := filepath.Join(basePath, "mha", "prefill", gpu, "*.csv")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, nil, fmt.Errorf("glob prefill CSVs: %w", err)
	}

	if len(files) == 0 {
		return nil, nil, fmt.Errorf("no prefill CSV files found in %s", filepath.Join(basePath, "mha", "prefill", gpu))
	}

	for _, file := range files {
		// Extract config key: "32-32-128.csv" -> "32-32-128"
		base := filepath.Base(file)
		configKey := base[:len(base)-4] // Remove .csv

		rows, err := loadPrefillCSV(file)
		if err != nil {
			return nil, nil, fmt.Errorf("load %s: %w", file, err)
		}

		data[configKey] = rows
		configs = append(configs, parseAttentionConfig(configKey))
	}

	return data, configs, nil
}

// loadDecodeCSV loads a single decode CSV file
func loadDecodeCSV(path string) ([]MHADecodeRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open decode CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read decode CSV: %w", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("decode CSV empty or missing header")
	}

	var rows []MHADecodeRow
	for i, record := range records[1:] { // Skip header
		if len(record) < 6 {
			return nil, fmt.Errorf("decode CSV row %d: expected 6 columns", i+2)
		}

		batchSize, err := strconv.Atoi(record[2])
		if err != nil {
			return nil, fmt.Errorf("decode CSV row %d: invalid batch_size: %w", i+2, err)
		}

		kvLen, err := strconv.Atoi(record[3])
		if err != nil {
			return nil, fmt.Errorf("decode CSV row %d: invalid kv_len: %w", i+2, err)
		}

		mfu, err := strconv.ParseFloat(record[5], 64)
		if err != nil {
			return nil, fmt.Errorf("decode CSV row %d: invalid mfu: %w", i+2, err)
		}

		rows = append(rows, MHADecodeRow{
			BatchSize: batchSize,
			KVLen:     kvLen,
			MFU:       mfu,
		})
	}

	return rows, nil
}

// loadAllDecodeCSVs loads all decode CSV files for the GPU
func loadAllDecodeCSVs(basePath, gpu string) (map[string][]MHADecodeRow, error) {
	data := make(map[string][]MHADecodeRow)

	pattern := filepath.Join(basePath, "mha", "decode", gpu, "*.csv")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("glob decode CSVs: %w", err)
	}

	if len(files) == 0 {
		return nil, fmt.Errorf("no decode CSV files found in %s", filepath.Join(basePath, "mha", "decode", gpu))
	}

	for _, file := range files {
		// Extract config key: "32-32-128-tp2.csv" -> "32-32-128-tp2"
		base := filepath.Base(file)
		configKey := base[:len(base)-4] // Remove .csv

		rows, err := loadDecodeCSV(file)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", file, err)
		}

		data[configKey] = rows
	}

	return data, nil
}

// loadGEMMCSV loads the GEMM CSV file
func loadGEMMCSV(basePath, gpu string) ([]GEMMRow, error) {
	path := filepath.Join(basePath, "gemm", gpu, "data.csv")
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GEMM CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read GEMM CSV: %w", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("GEMM CSV empty or missing header")
	}

	var rows []GEMMRow
	for i, record := range records[1:] { // Skip header
		if len(record) < 5 {
			return nil, fmt.Errorf("GEMM CSV row %d: expected 5 columns", i+2)
		}

		m, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, fmt.Errorf("GEMM CSV row %d: invalid m: %w", i+2, err)
		}

		k, err := strconv.Atoi(record[1])
		if err != nil {
			return nil, fmt.Errorf("GEMM CSV row %d: invalid k: %w", i+2, err)
		}

		n, err := strconv.Atoi(record[2])
		if err != nil {
			return nil, fmt.Errorf("GEMM CSV row %d: invalid n: %w", i+2, err)
		}

		mfu, err := strconv.ParseFloat(record[4], 64)
		if err != nil {
			return nil, fmt.Errorf("GEMM CSV row %d: invalid mfu: %w", i+2, err)
		}

		rows = append(rows, GEMMRow{
			M:   m,
			K:   k,
			N:   n,
			MFU: mfu,
		})
	}

	return rows, nil
}

// findNearestConfig finds the nearest attention config using Euclidean distance
func findNearestConfig(target string, available []AttentionShape) string {
	if len(available) == 0 {
		return target
	}

	targetShape := parseAttentionConfig(target)
	minDist := math.MaxFloat64
	nearest := available[0].ConfigKey

	for _, shape := range available {
		dist := euclideanDistance(targetShape, shape)
		if dist < minDist {
			minDist = dist
			nearest = shape.ConfigKey
		}
	}

	return nearest
}

// NewMFUDatabase creates a new MFU database from benchmark data
func NewMFUDatabase(modelConfig ModelConfig, benchDataPath string, gpu string) (*MFUDatabase, error) {
	// Compute model's attention config
	attentionConfig := computeAttentionConfig(modelConfig)

	// Load all CSV files
	prefillData, availableConfigs, err := loadAllPrefillCSVs(benchDataPath, gpu)
	if err != nil {
		return nil, fmt.Errorf("load prefill data: %w", err)
	}

	decodeData, err := loadAllDecodeCSVs(benchDataPath, gpu)
	if err != nil {
		return nil, fmt.Errorf("load decode data: %w", err)
	}

	gemmData, err := loadGEMMCSV(benchDataPath, gpu)
	if err != nil {
		return nil, fmt.Errorf("load GEMM data: %w", err)
	}

	// Check if exact config exists, otherwise find nearest
	originalConfig := attentionConfig
	if _, exists := prefillData[attentionConfig]; !exists {
		nearest := findNearestConfig(attentionConfig, availableConfigs)
		logrus.Infof("Attention config %s not found, using nearest: %s", attentionConfig, nearest)
		attentionConfig = nearest
	}

	// Count decode rows for logging
	decodeRowCount := 0
	for _, rows := range decodeData {
		decodeRowCount += len(rows)
	}

	logrus.Infof("Loaded MFU database: %s, attention config %s (model: %s), %d prefill rows, %d decode rows, %d GEMM rows",
		gpu, attentionConfig, originalConfig, len(prefillData[attentionConfig]), decodeRowCount, len(gemmData))

	return &MFUDatabase{
		prefillData:      prefillData,
		decodeData:       decodeData,
		gemmData:         gemmData,
		attentionConfig:  attentionConfig,
		availableConfigs: availableConfigs,
		gpu:              gpu,
	}, nil
}

// GetAttnPrefillMFU returns MFU for prefill attention at given seq_len
// Uses nearest neighbor: prefers floor (largest seq_len <= target), falls back to ceiling
func (db *MFUDatabase) GetAttnPrefillMFU(seqLen int) float64 {
	rows := db.prefillData[db.attentionConfig]
	if len(rows) == 0 {
		logrus.Fatalf("No prefill MFU data for config %s - database corrupted", db.attentionConfig)
	}

	// Find nearest neighbor, preferring floor values
	var bestRow *MHAPrefillRow
	minDist := math.MaxFloat64
	isFloor := false

	for i := range rows {
		row := &rows[i]

		// Check if this is a floor match (seq_len <= target)
		rowIsFloor := row.SeqLen <= seqLen

		// Compute distance
		dist := math.Abs(float64(row.SeqLen - seqLen))

		// Prefer floor matches over ceiling matches
		if !isFloor && rowIsFloor {
			// First floor match found, switch to it
			bestRow = row
			minDist = dist
			isFloor = true
		} else if isFloor && !rowIsFloor {
			// We have a floor match, don't switch to ceiling
			continue
		} else if dist < minDist {
			// Both are floor or both are ceiling, pick closest
			bestRow = row
			minDist = dist
		}
	}

	if bestRow == nil {
		logrus.Fatalf("No prefill MFU data available for config %s - database empty", db.attentionConfig)
	}

	// Handle zero or near-zero MFU (division by zero protection)
	if bestRow.MFU < 0.0001 {
		// Find nearest non-zero MFU by distance
		var fallbackRow *MHAPrefillRow
		minFallbackDist := math.MaxFloat64

		for i := range rows {
			if rows[i].MFU >= 0.0001 {
				dist := math.Abs(float64(rows[i].SeqLen - seqLen))
				if dist < minFallbackDist {
					minFallbackDist = dist
					fallbackRow = &rows[i]
				}
			}
		}

		if fallbackRow != nil {
			logrus.Warnf("Prefill MFU=%.4f too small for seq_len=%d, using nearest non-zero seq_len=%d with MFU=%.4f (distance=%.0f)",
				bestRow.MFU, seqLen, fallbackRow.SeqLen, fallbackRow.MFU, minFallbackDist)
			return fallbackRow.MFU
		}

		logrus.Fatalf("All prefill MFU values are zero for config %s - invalid benchmark data", db.attentionConfig)
	}

	// Log if we're using approximation (not exact match)
	if bestRow.SeqLen != seqLen {
		logrus.Debugf("Prefill MFU lookup: requested seq_len=%d, using nearest seq_len=%d",
			seqLen, bestRow.SeqLen)
	}

	return bestRow.MFU
}

// GetAttnDecodeMFU returns MFU for decode attention
// Uses nearest neighbor with Euclidean distance on (batch_size, kv_len)
func (db *MFUDatabase) GetAttnDecodeMFU(batchSize, kvLen, tp int) float64 {
	configKey := fmt.Sprintf("%s-tp%d", db.attentionConfig, tp)
	rows := db.decodeData[configKey]
	if len(rows) == 0 {
		// Try fallback to tp=1
		fallbackKey := fmt.Sprintf("%s-tp1", db.attentionConfig)
		rows = db.decodeData[fallbackKey]
		if len(rows) == 0 {
			logrus.Fatalf("No decode MFU data for config %s (TP=%d) - database missing", db.attentionConfig, tp)
		}
		logrus.Infof("Using TP=1 decode data as fallback for TP=%d", tp)
	}

	// Find nearest neighbor using Euclidean distance
	// Prefer rows where bs <= target_bs AND kv <= target_kv (floor in both dimensions)
	var bestRow *MHADecodeRow
	minDist := math.MaxFloat64
	isFloor := false

	for i := range rows {
		row := &rows[i]

		// Check if this is a floor match (bs <= target AND kv <= target)
		rowIsFloor := row.BatchSize <= batchSize && row.KVLen <= kvLen

		// Compute Euclidean distance
		dbs := float64(row.BatchSize - batchSize)
		dkv := float64(row.KVLen - kvLen)
		dist := math.Sqrt(dbs*dbs + dkv*dkv)

		// Prefer floor matches over ceiling matches
		if !isFloor && rowIsFloor {
			// First floor match found, switch to it
			bestRow = row
			minDist = dist
			isFloor = true
		} else if isFloor && !rowIsFloor {
			// We have a floor match, don't switch to ceiling
			continue
		} else if dist < minDist {
			// Both are floor or both are ceiling, pick closest
			bestRow = row
			minDist = dist
		}
	}

	if bestRow == nil {
		logrus.Fatalf("No decode MFU data available for config %s - database empty", db.attentionConfig)
	}

	// Handle zero or near-zero MFU (division by zero protection)
	if bestRow.MFU < 0.0001 {
		// Find nearest non-zero MFU by 2D Euclidean distance
		var fallbackRow *MHADecodeRow
		minFallbackDist := math.MaxFloat64

		for i := range rows {
			if rows[i].MFU >= 0.0001 {
				dbs := float64(rows[i].BatchSize - batchSize)
				dkv := float64(rows[i].KVLen - kvLen)
				dist := math.Sqrt(dbs*dbs + dkv*dkv)

				if dist < minFallbackDist {
					minFallbackDist = dist
					fallbackRow = &rows[i]
				}
			}
		}

		if fallbackRow != nil {
			logrus.Warnf("Decode MFU=%.4f too small for (bs=%d, kv=%d), using nearest non-zero (bs=%d, kv=%d) with MFU=%.4f (distance=%.0f)",
				bestRow.MFU, batchSize, kvLen, fallbackRow.BatchSize, fallbackRow.KVLen, fallbackRow.MFU, minFallbackDist)
			return fallbackRow.MFU
		}

		logrus.Fatalf("All decode MFU values are zero for config %s - invalid benchmark data", db.attentionConfig)
	}

	// Log if we're using approximation (not exact floor match)
	if bestRow.BatchSize != batchSize || bestRow.KVLen != kvLen {
		logrus.Debugf("Decode MFU lookup: requested (bs=%d, kv=%d), using nearest (bs=%d, kv=%d)",
			batchSize, kvLen, bestRow.BatchSize, bestRow.KVLen)
	}

	return bestRow.MFU
}

// GetGEMMmfu returns MFU for GEMM operation (m, k, n)
// Stage 1: Find smallest (k, n) where k >= target_k AND n >= target_n
// Stage 2: Within that (k, n), find largest m <= target_m
func (db *MFUDatabase) GetGEMMmfu(m, k, n int) float64 {
	if len(db.gemmData) == 0 {
		logrus.Fatalf("No GEMM MFU data available - database missing")
	}

	// Stage 1: Find smallest (k, n) >= target
	targetK := -1
	targetN := -1
	minDist := math.MaxFloat64

	for _, row := range db.gemmData {
		if row.K >= k && row.N >= n {
			dk := float64(row.K - k)
			dn := float64(row.N - n)
			dist := math.Sqrt(dk*dk + dn*dn)
			if dist < minDist {
				minDist = dist
				targetK = row.K
				targetN = row.N
			}
		}
	}

	if targetK == -1 || targetN == -1 {
		// No match found, use largest available
		logrus.Infof("No GEMM (k,n) >= (%d,%d), using largest available", k, n)
		targetK = db.gemmData[len(db.gemmData)-1].K
		targetN = db.gemmData[len(db.gemmData)-1].N
	}

	// Stage 2: Within (targetK, targetN), find largest m <= target_m
	mfu := 0.0
	foundExact := false
	for _, row := range db.gemmData {
		if row.K == targetK && row.N == targetN && row.M <= m {
			mfu = row.MFU
			foundExact = true
		}
	}

	if !foundExact {
		// Nearest neighbor fallback: use smallest m available for this (k, n)
		minM := math.MaxInt32
		for _, row := range db.gemmData {
			if row.K == targetK && row.N == targetN && row.M < minM {
				minM = row.M
				mfu = row.MFU
			}
		}
		if minM != math.MaxInt32 {
			logrus.Infof("No GEMM with m<=%d for (k=%d, n=%d), using nearest m=%d", m, targetK, targetN, minM)
		} else {
			logrus.Fatalf("No GEMM data found for (k=%d, n=%d) - database corrupted", targetK, targetN)
		}
	}

	// Handle zero or near-zero MFU (division by zero protection)
	if mfu < 0.0001 {
		// Find nearest non-zero MFU in the database
		for _, row := range db.gemmData {
			if row.MFU >= 0.0001 {
				logrus.Infof("GEMM MFU=%.4f too small for (m=%d, k=%d, n=%d), using fallback (m=%d, k=%d, n=%d) with MFU=%.4f",
					mfu, m, k, n, row.M, row.K, row.N, row.MFU)
				return row.MFU
			}
		}
		logrus.Fatalf("All GEMM MFU values are zero - invalid benchmark data")
	}

	return mfu
}

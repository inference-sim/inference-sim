# MFU-Based Roofline Model: Go Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Load pre-computed H100 MFU benchmark data and replace calibrated roofline constants with dynamic MFU lookups

**Architecture:** Port InferSim's CSV loading and lookup logic to Go, integrate with existing roofline_step.go by aggregating decode requests and bucketing prefill requests

**Tech Stack:** Go 1.21+, standard library (encoding/csv, math, fmt), existing sim package

---

## Task 1: Create MFU Data Structures

**Files:**
- Create: `sim/mfu_database.go`

**Step 1: Create basic data structures**

Add to `sim/mfu_database.go`:

```go
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
```

**Step 2: Verify file compiles**

Run: `go build ./sim`
Expected: SUCCESS (no syntax errors)

**Step 3: Commit**

```bash
git add sim/mfu_database.go
git commit -m "feat(mfu): add MFU data structures for CSV loading"
```

---

## Task 2: Implement Attention Config Computation

**Files:**
- Modify: `sim/mfu_database.go`

**Step 1: Add attention config computation**

Add to `sim/mfu_database.go`:

```go
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
```

**Step 2: Verify file compiles**

Run: `go build ./sim`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add sim/mfu_database.go
git commit -m "feat(mfu): add attention config computation helpers"
```

---

## Task 3: Implement CSV Loading Functions

**Files:**
- Modify: `sim/mfu_database.go`

**Step 1: Add prefill CSV loader**

Add to `sim/mfu_database.go`:

```go
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
```

**Step 2: Add decode CSV loader**

Add to `sim/mfu_database.go`:

```go
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
```

**Step 3: Add GEMM CSV loader**

Add to `sim/mfu_database.go`:

```go
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
```

**Step 4: Verify file compiles**

Run: `go build ./sim`
Expected: SUCCESS

**Step 5: Commit**

```bash
git add sim/mfu_database.go
git commit -m "feat(mfu): add CSV loading functions for MFU data"
```

---

## Task 4: Implement Database Initialization

**Files:**
- Modify: `sim/mfu_database.go`

**Step 1: Add NewMFUDatabase constructor**

Add to `sim/mfu_database.go`:

```go
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
```

**Step 2: Verify file compiles**

Run: `go build ./sim`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add sim/mfu_database.go
git commit -m "feat(mfu): add MFU database initialization with nearest neighbor fallback"
```

---

## Task 5: Implement Lookup Functions

**Files:**
- Modify: `sim/mfu_database.go`

**Step 1: Add prefill MFU lookup**

Add to `sim/mfu_database.go`:

```go
// GetAttnPrefillMFU returns MFU for prefill attention at given seq_len
// Uses floor logic: find largest seq_len <= target
func (db *MFUDatabase) GetAttnPrefillMFU(seqLen int) float64 {
	rows := db.prefillData[db.attentionConfig]
	if len(rows) == 0 {
		logrus.Warnf("No prefill MFU data for config %s", db.attentionConfig)
		return 0.8 // Fallback
	}

	// Find largest seq_len <= target
	mfu := rows[0].MFU
	for _, row := range rows {
		if row.SeqLen <= seqLen {
			mfu = row.MFU
		} else {
			break
		}
	}

	return mfu
}
```

**Step 2: Add decode MFU lookup**

Add to `sim/mfu_database.go`:

```go
// GetAttnDecodeMFU returns MFU for decode attention
// Uses floor logic on both batch_size and kv_len
func (db *MFUDatabase) GetAttnDecodeMFU(batchSize, kvLen, tp int) float64 {
	configKey := fmt.Sprintf("%s-tp%d", db.attentionConfig, tp)
	rows := db.decodeData[configKey]
	if len(rows) == 0 {
		// Try fallback to tp=1
		fallbackKey := fmt.Sprintf("%s-tp1", db.attentionConfig)
		rows = db.decodeData[fallbackKey]
		if len(rows) == 0 {
			logrus.Warnf("No decode MFU data for config %s (TP=%d)", db.attentionConfig, tp)
			return 0.01 // Fallback
		}
		logrus.Infof("Using TP=1 decode data as fallback for TP=%d", tp)
	}

	// Find largest batch_size <= target and largest kv_len <= target
	targetBS := 1
	targetKV := 1
	for _, row := range rows {
		if row.BatchSize <= batchSize {
			targetBS = row.BatchSize
		}
		if row.KVLen <= kvLen {
			targetKV = row.KVLen
		}
	}

	// Find exact match
	for _, row := range rows {
		if row.BatchSize == targetBS && row.KVLen == targetKV {
			return row.MFU
		}
	}

	logrus.Warnf("No decode MFU match for bs=%d, kv=%d (using target bs=%d, kv=%d)", batchSize, kvLen, targetBS, targetKV)
	return 0.01 // Fallback
}
```

**Step 3: Add GEMM MFU lookup**

Add to `sim/mfu_database.go`:

```go
// GetGEMMmfu returns MFU for GEMM operation (m, k, n)
// Stage 1: Find smallest (k, n) where k >= target_k AND n >= target_n
// Stage 2: Within that (k, n), find largest m <= target_m
func (db *MFUDatabase) GetGEMMmfu(m, k, n int) float64 {
	if len(db.gemmData) == 0 {
		logrus.Warn("No GEMM MFU data available")
		return 0.5 // Fallback
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
	mfu := 0.5 // Fallback
	for _, row := range db.gemmData {
		if row.K == targetK && row.N == targetN && row.M <= m {
			mfu = row.MFU
		}
	}

	return mfu
}
```

**Step 4: Verify file compiles**

Run: `go build ./sim`
Expected: SUCCESS

**Step 5: Commit**

```bash
git add sim/mfu_database.go
git commit -m "feat(mfu): implement MFU lookup functions with floor logic"
```

---

## Task 6: Add Helper Function for Individual GEMM Calculations

**Files:**
- Modify: `sim/roofline_step.go`

**Step 1: Add MFU database parameter to rooflineStepTime**

Modify function signature in `sim/roofline_step.go`:

```go
func rooflineStepTime(_ string, modelConfig ModelConfig, hwConfig HardwareCalib, stepConfig StepConfig, tp int, mfuDB *MFUDatabase) int64 {
```

**Step 2: Add helper function to compute individual GEMM times**

Add before `rooflineStepTime` function:

```go
// computeGEMMTime computes time for a single GEMM operation with MFU lookup
func computeGEMMTime(m, k, n int, peakFlops float64, mfuDB *MFUDatabase, fallbackMFU float64) float64 {
	// FLOPs for GEMM: 2*m*k*n
	flops := 2.0 * float64(m) * float64(k) * float64(n)

	var mfu float64
	if mfuDB != nil {
		mfu = mfuDB.GetGEMMmfu(m, k, n)
	} else {
		mfu = fallbackMFU
	}

	return flops / (peakFlops * mfu)
}

// computeTransformerGEMMTimes computes time for all GEMM operations in a transformer layer
// Returns total time in seconds for all GEMMs across all layers
func computeTransformerGEMMTimes(modelConfig ModelConfig, batchSize int, peakFlops float64, mfuDB *MFUDatabase, fallbackMFU float64, tpScaling float64) float64 {
	m := batchSize
	hiddenSize := modelConfig.HiddenDim
	intermediateSize := modelConfig.IntermediateDim
	if intermediateSize == 0 {
		intermediateSize = 4 * hiddenSize
	}

	nHeads := modelConfig.NumHeads
	nKVHeads := modelConfig.NumKVHeads
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}
	headDim := hiddenSize / nHeads

	// QKV projection: m × hidden_size → (nh*dh + 2*nkv*dh)
	qkvSize := nHeads*headDim + 2*nKVHeads*headDim
	qkvTime := computeGEMMTime(m, hiddenSize, qkvSize, peakFlops, mfuDB, fallbackMFU)

	// O projection: m × (nh*dh) → hidden_size
	oTime := computeGEMMTime(m, nHeads*headDim, hiddenSize, peakFlops, mfuDB, fallbackMFU)

	// MLP: Gate, Up, Down (3 GEMMs for SwiGLU)
	gateTime := computeGEMMTime(m, hiddenSize, intermediateSize, peakFlops, mfuDB, fallbackMFU)
	upTime := computeGEMMTime(m, hiddenSize, intermediateSize, peakFlops, mfuDB, fallbackMFU)
	downTime := computeGEMMTime(m, intermediateSize, hiddenSize, peakFlops, mfuDB, fallbackMFU)

	// Total per layer
	perLayerTime := qkvTime + oTime + gateTime + upTime + downTime

	// Scale by number of layers and TP scaling
	return (perLayerTime * float64(modelConfig.NumLayers)) / tpScaling
}
```

**Step 3: Verify file compiles**

Run: `go build ./sim`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add sim/roofline_step.go
git commit -m "feat(mfu): add helper functions for per-GEMM MFU lookups"
```

---

## Task 7: Integrate MFU Database into Roofline (Decode Phase)

**Files:**
- Modify: `sim/roofline_step.go`

**Step 1: Replace decode phase with per-GEMM lookups**

Find this section (around line 197-221):

```go
// 2. DECODE PHASE
if len(stepConfig.DecodeRequests) > 0 {
	hasDecode = true
	var dGemmFlops, dVectorFlops, dDynamicBytes float64

	for _, req := range stepConfig.DecodeRequests {
		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, 1, true, true)
		// Decode TP scaling - applies to both compute and memory
		dGemmFlops += f["gemm_ops"] / effectiveTpDecode
		dVectorFlops += f["sram_ops"] / effectiveTpDecode

		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
		dDynamicBytes += (m["total"] - m["model_weights"]) / effectiveTpDecode
	}

	baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
	dWeightBytes := baseMem["model_weights"] / effectiveTpDecode

	// Unified MFU for decode across all batch sizes
	adjustedDecodeMFU := hwConfig.MfuDecode * hwConfig.MfuDecodeMultiplier

	// Reduce effective bandwidth for decode (scattered KV cache access)
	decodeEffBW := effBW * hwConfig.DecodeBwFactor

	decodeComputeS = (dGemmFlops / (peakFlops * adjustedDecodeMFU)) + (dVectorFlops / vectorPeak)
	decodeMemoryS = (dWeightBytes + dDynamicBytes) / decodeEffBW
}
```

Replace with:

```go
// 2. DECODE PHASE
if len(stepConfig.DecodeRequests) > 0 {
	hasDecode = true
	var dVectorFlops, dDynamicBytes float64

	// Aggregate batch_size and find max kv_len
	totalBatchSize := len(stepConfig.DecodeRequests)
	maxKVLen := int64(0)

	for _, req := range stepConfig.DecodeRequests {
		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, 1, true, true)
		// Decode TP scaling - applies to attention core
		dVectorFlops += f["sram_ops"] / effectiveTpDecode

		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
		dDynamicBytes += (m["total"] - m["model_weights"]) / effectiveTpDecode

		if req.ProgressIndex > maxKVLen {
			maxKVLen = req.ProgressIndex
		}
	}

	baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
	dWeightBytes := baseMem["model_weights"] / effectiveTpDecode

	// Compute GEMM times with per-operation MFU lookups
	fallbackMFU := hwConfig.MfuDecode
	dGemmTimeS := computeTransformerGEMMTimes(modelConfig, totalBatchSize, peakFlops, mfuDB, fallbackMFU, effectiveTpDecode)

	// Compute attention core time with attention MFU
	// Note: InferSim uses same peakFlops for both GEMM and attention core
	// MFU differentiates efficiency (GEMM: 0.5-1.0, decode attn: ~0.003)
	var attnCoreTimeS float64
	if mfuDB != nil {
		attnMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, int(maxKVLen), tp)
		adjustedAttnMFU := attnMFU * hwConfig.MfuDecodeMultiplier
		attnCoreTimeS = dVectorFlops / (peakFlops * adjustedAttnMFU)
	} else {
		adjustedDecodeMFU := hwConfig.MfuDecode * hwConfig.MfuDecodeMultiplier
		attnCoreTimeS = dVectorFlops / (peakFlops * adjustedDecodeMFU)
	}

	// Reduce effective bandwidth for decode (scattered KV cache access)
	decodeEffBW := effBW * hwConfig.DecodeBwFactor

	// Sum GEMM time and attention core time (no vectorPeak division)
	decodeComputeS = dGemmTimeS + attnCoreTimeS
	decodeMemoryS = (dWeightBytes + dDynamicBytes) / decodeEffBW
}
```

**Step 2: Verify file compiles**

Run: `go build ./sim`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add sim/roofline_step.go
git commit -m "feat(mfu): integrate per-GEMM MFU lookups for decode phase

- GEMM projections use individual MFU lookups
- Attention core uses attention-specific decode MFU with peakFlops
- Aggregate batch_size and max kv_len for attention MFU
- Match InferSim formula: time = flops / (peakFlops * mfu)"
```

---

## Task 8: Integrate MFU Database into Roofline (Prefill Phase)

**Files:**
- Modify: `sim/roofline_step.go`

**Step 1: Replace prefill phase with per-GEMM lookups and bucketing**

Find this section (around line 166-194):

```go
// 1. PREFILL PHASE
if len(stepConfig.PrefillRequests) > 0 {
	hasPrefill = true
	var pGemmFlops, pVectorFlops, pDynamicBytes float64

	for _, req := range stepConfig.PrefillRequests {
		numTokens := int64(req.NumNewPrefillTokens)

		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, numTokens, true, true)
		// Use sublinear TP scaling for compute (prefill is compute-bound)
		pGemmFlops += f["gemm_ops"] / effectiveTpPrefill
		pVectorFlops += f["sram_ops"] / effectiveTpPrefill

		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
		// Memory still scales linearly with TP (each GPU has 1/tp of weights)
		pDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
	}

	baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
	pWeightBytes := baseMem["model_weights"] / tpFactor

	// Prefill MFU - calibrated to match vLLM measured performance
	adjustedPrefillMFU := hwConfig.MfuPrefill * hwConfig.MfuPrefillMultiplier

	// Reduce effective bandwidth for prefill (memory contention during KV cache operations)
	prefillEffBW := effBW * hwConfig.PrefillBwFactor

	prefillComputeS = (pGemmFlops / (peakFlops * adjustedPrefillMFU)) + (pVectorFlops / vectorPeak)
	prefillMemoryS = (pWeightBytes + pDynamicBytes) / prefillEffBW
}
```

Replace with:

```go
// 1. PREFILL PHASE
if len(stepConfig.PrefillRequests) > 0 {
	hasPrefill = true

	// Bucket prefill requests by seq_len
	buckets := make(map[int][]PrefillRequest)
	for _, req := range stepConfig.PrefillRequests {
		seqLen := int(req.NumNewPrefillTokens)
		buckets[seqLen] = append(buckets[seqLen], req)
	}

	// Process each bucket separately
	var totalPrefillComputeS, totalPrefillMemoryS float64
	prefillEffBW := effBW * hwConfig.PrefillBwFactor
	fallbackMFU := hwConfig.MfuPrefill

	for seqLen, requests := range buckets {
		var bucketVectorFlops, bucketDynamicBytes float64
		bucketBatchSize := len(requests)

		for _, req := range requests {
			numTokens := int64(req.NumNewPrefillTokens)

			f := calculateTransformerFlops(modelConfig, req.ProgressIndex, numTokens, true, true)
			bucketVectorFlops += f["sram_ops"] / effectiveTpPrefill

			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
			bucketDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
		}

		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		bucketWeightBytes := baseMem["model_weights"] / tpFactor

		// Compute GEMM times with per-operation MFU lookups for this bucket
		bucketGemmTimeS := computeTransformerGEMMTimes(modelConfig, bucketBatchSize*seqLen, peakFlops, mfuDB, fallbackMFU, effectiveTpPrefill)

		// Compute attention core time with attention MFU for this seq_len
		// Note: InferSim divides prefill attention by 1.8 (hardware-specific factor)
		var attnCoreTimeS float64
		if mfuDB != nil {
			attnMFU := mfuDB.GetAttnPrefillMFU(seqLen)
			adjustedAttnMFU := attnMFU * hwConfig.MfuPrefillMultiplier
			attnCoreTimeS = bucketVectorFlops / 1.8 / (peakFlops * adjustedAttnMFU)
		} else {
			adjustedPrefillMFU := hwConfig.MfuPrefill * hwConfig.MfuPrefillMultiplier
			attnCoreTimeS = bucketVectorFlops / 1.8 / (peakFlops * adjustedPrefillMFU)
		}

		// Sum GEMM time and attention core time (no vectorPeak division)
		bucketComputeS := bucketGemmTimeS + attnCoreTimeS
		bucketMemoryS := (bucketWeightBytes + bucketDynamicBytes) / prefillEffBW

		totalPrefillComputeS += bucketComputeS
		totalPrefillMemoryS += bucketMemoryS
	}

	prefillComputeS = totalPrefillComputeS
	prefillMemoryS = totalPrefillMemoryS
}
```

**Step 2: Verify file compiles**

Run: `go build ./sim`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add sim/roofline_step.go
git commit -m "feat(mfu): integrate per-GEMM MFU lookups for prefill phase

- Bucket prefill requests by seq_len
- GEMM projections use individual MFU lookups per bucket
- Attention core uses attention-specific prefill MFU with /1.8 factor
- Match InferSim formula: time = flops / 1.8 / (peakFlops * mfu)
- Sum times across all buckets"
```

---

## Task 9: Initialize MFU Database in Simulator

**Files:**
- Modify: `sim/simulator.go`

**Step 1: Find simulator initialization**

Locate the simulator initialization code (search for `func NewSimulator` or similar).

**Step 2: Add MFU database initialization**

After HardwareCalib is loaded, add:

```go
// Initialize MFU database
var mfuDB *MFUDatabase
if benchDataPath := "bench_data"; true {
	db, err := NewMFUDatabase(modelConfig, benchDataPath, gpu)
	if err != nil {
		logrus.Fatalf("Failed to initialize MFU database: %v", err)
	}
	mfuDB = db
}
```

**Step 3: Pass mfuDB to rooflineStepTime calls**

Find all calls to `rooflineStepTime` and add `mfuDB` parameter:

```go
stepTime := rooflineStepTime(model, modelConfig, hwConfig, stepConfig, tp, mfuDB)
```

**Step 4: Verify simulator compiles**

Run: `go build .`
Expected: SUCCESS

**Step 5: Commit**

```bash
git add sim/simulator.go
git commit -m "feat(mfu): initialize MFU database in simulator startup"
```

---

## Task 10: Test End-to-End Integration

**Files:**
- Test manually with simulator

**Step 1: Run simulator on Llama-2-7B**

Run: `./inference-sim --model model_configs/llama-2-7b-hf --gpu H100 --tp 1 --workload <test-workload>`

Expected:
- Startup log shows: "Loaded MFU database: H100, attention config 32-32-128..."
- Simulator runs without errors
- Predictions are generated

**Step 2: Check for nearest neighbor fallback**

Run with a model that doesn't have exact match (e.g., Qwen2.5-1.5B):

Run: `./inference-sim --model model_configs/qwen2.5-1.5b-instruct --gpu H100 --tp 1 --workload <test-workload>`

Expected:
- Info log: "Attention config X-Y-Z not found, using nearest: A-B-C"
- Simulator continues successfully

**Step 3: Test error handling**

Temporarily rename bench_data folder:

Run: `mv bench_data bench_data_backup && ./inference-sim ...`

Expected: Fatal error with message about missing CSV files

Restore: `mv bench_data_backup bench_data`

**Step 4: Commit if tests pass**

```bash
git commit --allow-empty -m "test(mfu): verify end-to-end MFU database integration"
```

---

## Task 11: Add Documentation

**Files:**
- Modify: `sim/mfu_database.go`

**Step 1: Add package documentation**

Add at top of `sim/mfu_database.go`:

```go
// Package sim implements MFU-based roofline modeling.
//
// MFUDatabase loads pre-computed MFU benchmark data from CSV files and provides
// lookup functions for attention (prefill/decode) and GEMM operations.
//
// Data structure:
// - bench_data/{gpu}/mha/prefill/{config}.csv - Prefill attention MFU
// - bench_data/{gpu}/mha/decode/{config}-tp{N}.csv - Decode attention MFU (TP-specific)
// - bench_data/{gpu}/gemm/data.csv - GEMM operation MFU
//
// Lookup strategies:
// - Prefill: Floor lookup on seq_len (largest seq_len <= target)
// - Decode: Floor lookup on (batch_size, kv_len) with TP-specific files
// - GEMM: Two-stage lookup matching InferSim's algorithm
//
// Fallback behavior:
// - Missing exact attention config -> nearest neighbor (Euclidean distance)
// - Missing CSV files -> fatal error
```

**Step 2: Add function comments**

Ensure all exported functions have comments (already done in previous steps).

**Step 3: Verify documentation**

Run: `go doc ./sim | grep MFU`

Expected: Shows MFUDatabase and related functions

**Step 4: Commit**

```bash
git add sim/mfu_database.go
git commit -m "docs(mfu): add package and function documentation"
```

---

## Task 12: Final Verification and Cleanup

**Files:**
- All modified files

**Step 1: Run full build**

Run: `go build ./...`
Expected: SUCCESS with no warnings

**Step 2: Run simulator with different models and TPs**

Test matrix:
- Llama-2-7B: TP=1, 2, 4
- Llama-2-70B: TP=1, 2, 4
- Mixtral-8x7B: TP=1, 2, 4

Expected: All run successfully with appropriate MFU lookups

**Step 3: Verify logging output**

Check that startup logs show:
- MFU database loaded successfully
- Attention config (with nearest neighbor if applicable)
- Row counts

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(mfu): complete MFU-based roofline integration

- Load H100 benchmark data at startup
- Per-GEMM MFU lookups for QKV, O, Gate, Up, Down projections
- Attention core uses peakFlops * mfu (matches InferSim)
- Prefill attention includes /1.8 hardware factor
- Aggregate decode requests by batch_size for attention MFU
- Bucket prefill requests by seq_len for attention MFU
- Nearest neighbor fallback for missing configs
- Error out on missing CSV files"
```

---

## Success Criteria Checklist

- ✅ All CSV files load successfully at startup
- ✅ Lookups return valid MFU values (0 < mfu ≤ 1.0)
- ✅ Nearest neighbor finds reasonable matches
- ✅ Simulator produces predictions without crashing
- ✅ Predictions differ from calibrated baseline (validates MFU is used)
- ✅ Clear error messages for missing data
- ✅ Info logs help users debug configuration issues

---

## References

- Design doc: `docs/plans/2026-02-20-mfu-roofline-go-design.md`
- InferSim reference: `InferSim/mfu/mfu.py`
- Benchmark data: `bench_data/h100/`

package sim

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/sirupsen/logrus"
)

// lerp performs linear interpolation between a and b at parameter t in [0,1].
func lerp(a, b, t float64) float64 {
	return a + (b-a)*t
}

// bracketIndex returns the indices of the floor and ceiling values in a sorted
// slice that bracket target. If target is at or below the minimum, both indices
// point to 0. If at or above the maximum, both point to len-1.
func bracketIndex(sorted []int, target int) (lo, hi int) {
	n := len(sorted)
	if n == 0 {
		return 0, 0
	}
	if target <= sorted[0] {
		return 0, 0
	}
	if target >= sorted[n-1] {
		return n - 1, n - 1
	}
	// Binary search for insertion point
	hi = sort.SearchInts(sorted, target)
	if hi < n && sorted[hi] == target {
		return hi, hi // exact match
	}
	return hi - 1, hi
}

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

// decodeGridKey identifies a cell in the (batch_size, kv_len) decode MFU grid.
type decodeGridKey struct{ bs, kv int }

// decodeGrid holds a pre-built 2D interpolation grid for decode MFU lookups.
// Built once at NewMFUDatabase time to avoid per-lookup map/sort overhead.
type decodeGrid struct {
	bsVals []int                      // sorted unique batch sizes
	kvVals []int                      // sorted unique KV lengths
	grid   map[decodeGridKey]float64  // (bs, kv) → MFU
	rows   []MHADecodeRow             // original rows for zero-MFU fallback search
}

// gemmKN is a composite key for GEMM data indexed by (K, N) dimensions.
type gemmKN struct{ k, n int }

// gemmMPoint is a (M, MFU) pair, used in pre-sorted slices for M-dimension interpolation.
type gemmMPoint struct {
	m   int
	mfu float64
}

// prefillGrid holds pre-sorted seq_len → MFU data for O(log n) lookup.
// Built once at construction to avoid per-call allocation and sort.
type prefillGrid struct {
	seqLens []int     // sorted ascending
	mfus    []float64 // parallel to seqLens
	rows    []MHAPrefillRow // original rows for zero-MFU fallback
}

// MFUDatabase holds all MFU benchmark data with pre-built indexes for
// efficient hot-path lookups. All indexes are built at construction time
// by NewMFUDatabase to avoid per-call allocations in the DES event loop.
type MFUDatabase struct {
	prefillData      map[string][]MHAPrefillRow // key: "32-32-128"
	prefillGrids     map[string]*prefillGrid    // pre-sorted prefill data (I5)
	decodeData       map[string][]MHADecodeRow  // key: "32-32-128-tp2"
	decodeGrids      map[string]*decodeGrid     // pre-built grids, same keys as decodeData
	decodeTPKeys     map[int]string             // pre-computed TP config keys (I7b)
	decodeSortedTPs  []int                      // sorted available TPs for nearest-TP fallback (I16)
	gemmData         []GEMMRow
	gemmIndex        map[gemmKN][]gemmMPoint    // pre-indexed GEMM data by (K,N), sorted by M (I6)
	gemmAllKNs       []gemmKN                   // all distinct (K,N) pairs for fallback search
	attentionConfig  string // Model's attention config
	availableConfigs []AttentionShape
	gpu              string
}

// computeAttentionConfig derives the attention config string from model config
func computeAttentionConfig(config ModelConfig) string {
	if config.NumHeads == 0 {
		return "0-0-0"
	}
	headDim := config.HiddenDim / config.NumHeads
	numKVHeads := config.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = config.NumHeads
	}
	return fmt.Sprintf("%d-%d-%d", config.NumHeads, numKVHeads, headDim)
}

// parseAttentionConfig parses "32-32-128" into AttentionShape.
// Logs a warning (R1) if the config key doesn't parse into exactly 3 fields.
func parseAttentionConfig(configKey string) AttentionShape {
	var nh, nkv, hd int
	n, _ := fmt.Sscanf(configKey, "%d-%d-%d", &nh, &nkv, &hd)
	if n != 3 {
		logrus.Warnf("parseAttentionConfig: %q parsed only %d of 3 fields", configKey, n)
	}
	return AttentionShape{
		NumHeads:   nh,
		NumKVHeads: nkv,
		HeadDim:    hd,
		ConfigKey:  configKey,
	}
}

// euclideanDistance computes distance between two attention shapes.
// Known approximation: dimensions (NumHeads, NumKVHeads, HeadDim) are unweighted,
// so a 32-head difference counts the same as a 32-HeadDim difference even though
// their impact on MFU differs. This is acceptable for nearest-config selection
// where exact matches are expected for most production models.
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
	defer func() { _ = file.Close() }()

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
		if math.IsNaN(mfu) || math.IsInf(mfu, 0) || mfu < 0 {
			return nil, fmt.Errorf("prefill CSV row %d: mfu must be finite and non-negative, got %v", i+2, mfu)
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
	defer func() { _ = file.Close() }()

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
		if math.IsNaN(mfu) || math.IsInf(mfu, 0) || mfu < 0 {
			return nil, fmt.Errorf("decode CSV row %d: mfu must be finite and non-negative, got %v", i+2, mfu)
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
	defer func() { _ = file.Close() }()

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
		if math.IsNaN(mfu) || math.IsInf(mfu, 0) || mfu < 0 {
			return nil, fmt.Errorf("GEMM CSV row %d: mfu must be finite and non-negative, got %v", i+2, mfu)
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

// buildDecodeGrid pre-builds the 2D interpolation grid for a set of decode rows.
func buildDecodeGrid(rows []MHADecodeRow) *decodeGrid {
	grid := make(map[decodeGridKey]float64, len(rows))
	bsSet := make(map[int]bool, len(rows))
	kvSet := make(map[int]bool, len(rows))

	for _, r := range rows {
		grid[decodeGridKey{r.BatchSize, r.KVLen}] = r.MFU
		bsSet[r.BatchSize] = true
		kvSet[r.KVLen] = true
	}

	bsVals := make([]int, 0, len(bsSet))
	for v := range bsSet {
		bsVals = append(bsVals, v)
	}
	sort.Ints(bsVals)

	kvVals := make([]int, 0, len(kvSet))
	for v := range kvSet {
		kvVals = append(kvVals, v)
	}
	sort.Ints(kvVals)

	return &decodeGrid{
		bsVals: bsVals,
		kvVals: kvVals,
		grid:   grid,
		rows:   rows,
	}
}

// buildPrefillGrid pre-sorts prefill rows by seq_len for O(log n) lookup.
func buildPrefillGrid(rows []MHAPrefillRow) *prefillGrid {
	sorted := make([]MHAPrefillRow, len(rows))
	copy(sorted, rows)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].SeqLen < sorted[j].SeqLen })

	seqLens := make([]int, len(sorted))
	mfus := make([]float64, len(sorted))
	for i, r := range sorted {
		seqLens[i] = r.SeqLen
		mfus[i] = r.MFU
	}
	return &prefillGrid{seqLens: seqLens, mfus: mfus, rows: rows}
}

// buildGEMMIndex pre-indexes GEMM data by (K, N) with M-sorted point lists.
// Converts O(n) per-lookup scans to O(1) map access + O(log m) binary search.
func buildGEMMIndex(rows []GEMMRow) (map[gemmKN][]gemmMPoint, []gemmKN) {
	idx := make(map[gemmKN][]gemmMPoint)
	for _, r := range rows {
		key := gemmKN{r.K, r.N}
		idx[key] = append(idx[key], gemmMPoint{r.M, r.MFU})
	}
	// Sort each M-point list by M for binary search / interpolation
	allKNs := make([]gemmKN, 0, len(idx))
	for key, pts := range idx {
		sort.Slice(pts, func(i, j int) bool { return pts[i].m < pts[j].m })
		idx[key] = pts
		allKNs = append(allKNs, key)
	}
	return idx, allKNs
}

// collectDecodeTPKeys pre-computes TP config key strings and collects available TPs.
func collectDecodeTPKeys(attentionConfig string, decodeData map[string][]MHADecodeRow) (map[int]string, []int) {
	tpSet := make(map[int]bool)
	for key := range decodeData {
		// Parse TP from key like "32-32-128-tp2"
		var tp int
		if _, err := fmt.Sscanf(key, attentionConfig+"-tp%d", &tp); err == nil {
			tpSet[tp] = true
		}
	}
	keys := make(map[int]string, len(tpSet))
	tps := make([]int, 0, len(tpSet))
	for tp := range tpSet {
		keys[tp] = fmt.Sprintf("%s-tp%d", attentionConfig, tp)
		tps = append(tps, tp)
	}
	sort.Ints(tps)
	return keys, tps
}

// NewMFUDatabase creates a new MFU database from benchmark data.
// All indexes are pre-built at construction time to avoid per-lookup
// allocations in the hot path (DES event loop).
func NewMFUDatabase(modelConfig ModelConfig, benchDataPath string, gpu string) (*MFUDatabase, error) {
	if modelConfig.NumHeads <= 0 {
		return nil, fmt.Errorf("NewMFUDatabase: ModelConfig.NumHeads must be > 0, got %d", modelConfig.NumHeads)
	}
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

	// Pre-build decode grids to avoid per-lookup map/sort overhead
	decodeGrids := make(map[string]*decodeGrid, len(decodeData))
	for key, rows := range decodeData {
		decodeGrids[key] = buildDecodeGrid(rows)
	}

	// Pre-build prefill grids (I5: avoid per-call sort/allocation)
	prefillGrids := make(map[string]*prefillGrid, len(prefillData))
	for key, rows := range prefillData {
		prefillGrids[key] = buildPrefillGrid(rows)
	}

	// Pre-index GEMM data (I6: avoid O(n) scans per call)
	gemmIdx, allKNs := buildGEMMIndex(gemmData)

	// Pre-compute TP config keys (I7b: avoid fmt.Sprintf per call)
	decodeTPKeys, sortedTPs := collectDecodeTPKeys(attentionConfig, decodeData)

	return &MFUDatabase{
		prefillData:      prefillData,
		prefillGrids:     prefillGrids,
		decodeData:       decodeData,
		decodeGrids:      decodeGrids,
		decodeTPKeys:     decodeTPKeys,
		decodeSortedTPs:  sortedTPs,
		gemmData:         gemmData,
		gemmIndex:        gemmIdx,
		gemmAllKNs:       allKNs,
		attentionConfig:  attentionConfig,
		availableConfigs: availableConfigs,
		gpu:              gpu,
	}, nil
}

// findNearestTPGrid returns the decode grid for the nearest available TP.
// Uses pre-sorted TP list for efficient nearest-neighbor lookup (I16).
func (db *MFUDatabase) findNearestTPGrid(targetTP int) *decodeGrid {
	if len(db.decodeSortedTPs) == 0 {
		return nil
	}
	// Find the nearest TP
	bestTP := db.decodeSortedTPs[0]
	bestDist := abs(bestTP - targetTP)
	for _, tp := range db.decodeSortedTPs[1:] {
		d := abs(tp - targetTP)
		if d < bestDist {
			bestDist = d
			bestTP = tp
		}
	}
	key := db.decodeTPKeys[bestTP]
	if key == "" {
		key = fmt.Sprintf("%s-tp%d", db.attentionConfig, bestTP)
	}
	logrus.Debugf("Using TP=%d decode data as fallback for TP=%d (nearest available)", bestTP, targetTP)
	return db.decodeGrids[key]
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// GetAttnPrefillMFU returns MFU for prefill attention at given seq_len.
// Linearly interpolates between the floor and ceiling seq_len grid points
// that bracket the target, producing smooth MFU transitions instead of
// step-function jumps at grid boundaries.
// Uses pre-sorted data from construction time for O(log n) lookup (I5).
func (db *MFUDatabase) GetAttnPrefillMFU(seqLen int) float64 {
	pg := db.prefillGrids[db.attentionConfig]
	if pg == nil || len(pg.seqLens) == 0 {
		logrus.Warnf("No prefill MFU data for config %s - returning floor MFU", db.attentionConfig)
		return 0.0001
	}

	// O(log n) bracket lookup on pre-sorted seq_len array
	lo, hi := bracketIndex(pg.seqLens, seqLen)

	var mfu float64
	if lo == hi {
		mfu = pg.mfus[lo]
	} else {
		t := float64(seqLen-pg.seqLens[lo]) / float64(pg.seqLens[hi]-pg.seqLens[lo])
		mfu = lerp(pg.mfus[lo], pg.mfus[hi], t)
	}

	// Zero-MFU protection: find nearest non-zero value as fallback
	if mfu < 0.0001 {
		var fallbackRow *MHAPrefillRow
		minFallbackDist := math.MaxFloat64
		for i := range pg.rows {
			if pg.rows[i].MFU >= 0.0001 {
				dist := math.Abs(float64(pg.rows[i].SeqLen - seqLen))
				if dist < minFallbackDist {
					minFallbackDist = dist
					fallbackRow = &pg.rows[i]
				}
			}
		}
		if fallbackRow != nil {
			logrus.Warnf("Prefill MFU=%.4f too small for seq_len=%d, using nearest non-zero seq_len=%d with MFU=%.4f",
				mfu, seqLen, fallbackRow.SeqLen, fallbackRow.MFU)
			return fallbackRow.MFU
		}
		logrus.Warnf("All prefill MFU values are zero for config %s - returning floor MFU", db.attentionConfig)
		return 0.0001
	}

	return mfu
}

// GetAttnDecodeMFU returns MFU for decode attention.
// Bilinearly interpolates on the (batch_size, kv_len) grid, producing smooth
// MFU transitions instead of step-function jumps at grid boundaries.
// Uses pre-built grids from NewMFUDatabase to avoid per-lookup allocation overhead.
func (db *MFUDatabase) GetAttnDecodeMFU(batchSize, kvLen, tp int) float64 {
	// Use pre-computed TP config key (I7b: no per-call fmt.Sprintf)
	configKey := db.decodeTPKeys[tp]
	if configKey == "" {
		configKey = fmt.Sprintf("%s-tp%d", db.attentionConfig, tp)
	}
	dg := db.decodeGrids[configKey]
	if dg == nil {
		// Try fallback to nearest available TP instead of always tp=1 (I16)
		dg = db.findNearestTPGrid(tp)
		if dg == nil {
			logrus.Warnf("No decode MFU data for config %s (TP=%d) - returning floor MFU", db.attentionConfig, tp)
			return 0.0001
		}
	}

	// Guard: empty axis slices should be unreachable (rows > 0 guarantees entries),
	// but defend against it to prevent index-out-of-bounds panics.
	if len(dg.bsVals) == 0 || len(dg.kvVals) == 0 {
		logrus.Warnf("Decode MFU grid has empty axis (bs=%d, kv=%d entries) - returning floor MFU", len(dg.bsVals), len(dg.kvVals))
		return 0.0001
	}

	// Find floor/ceiling indices in each dimension
	bsLo, bsHi := bracketIndex(dg.bsVals, batchSize)
	kvLo, kvHi := bracketIndex(dg.kvVals, kvLen)

	bs0, bs1 := dg.bsVals[bsLo], dg.bsVals[bsHi]
	kv0, kv1 := dg.kvVals[kvLo], dg.kvVals[kvHi]

	// Look up the four corner MFU values
	q00 := dg.grid[decodeGridKey{bs0, kv0}]
	q10 := dg.grid[decodeGridKey{bs1, kv0}]
	q01 := dg.grid[decodeGridKey{bs0, kv1}]
	q11 := dg.grid[decodeGridKey{bs1, kv1}]

	// Bilinear interpolation
	var mfu float64
	if bs0 == bs1 && kv0 == kv1 {
		mfu = q00
	} else if bs0 == bs1 {
		t := float64(kvLen-kv0) / float64(kv1-kv0)
		mfu = lerp(q00, q01, t)
	} else if kv0 == kv1 {
		t := float64(batchSize-bs0) / float64(bs1-bs0)
		mfu = lerp(q00, q10, t)
	} else {
		tBS := float64(batchSize-bs0) / float64(bs1-bs0)
		tKV := float64(kvLen-kv0) / float64(kv1-kv0)
		top := lerp(q00, q10, tBS)
		bot := lerp(q01, q11, tBS)
		mfu = lerp(top, bot, tKV)
	}

	// Zero-MFU protection: if interpolated value is still near zero,
	// find nearest non-zero value as fallback
	if mfu < 0.0001 {
		var fallbackRow *MHADecodeRow
		minFallbackDist := math.MaxFloat64
		for i := range dg.rows {
			if dg.rows[i].MFU >= 0.0001 {
				dbs := float64(dg.rows[i].BatchSize - batchSize)
				dkv := float64(dg.rows[i].KVLen - kvLen)
				dist := math.Sqrt(dbs*dbs + dkv*dkv)
				if dist < minFallbackDist {
					minFallbackDist = dist
					fallbackRow = &dg.rows[i]
				}
			}
		}
		if fallbackRow != nil {
			logrus.Warnf("Decode MFU=%.4f too small for (bs=%d, kv=%d), using nearest non-zero (bs=%d, kv=%d) with MFU=%.4f",
				mfu, batchSize, kvLen, fallbackRow.BatchSize, fallbackRow.KVLen, fallbackRow.MFU)
			return fallbackRow.MFU
		}
		logrus.Warnf("All decode MFU values are zero for config %s - returning floor MFU", db.attentionConfig)
		return 0.0001
	}

	return mfu
}

// GetGEMMmfu returns MFU for GEMM operation (m, k, n).
// Stage 1: Find smallest (k, n) where k >= target_k AND n >= target_n using pre-built index.
// Stage 2: Within that (k, n), interpolate M using pre-sorted M-points.
// Uses pre-indexed data from construction time for O(1) + O(log m) lookup (I6).
func (db *MFUDatabase) GetGEMMmfu(m, k, n int) float64 {
	if len(db.gemmIndex) == 0 {
		logrus.Warnf("No GEMM MFU data available - returning floor MFU")
		return 0.0001
	}

	// Stage 1: Find smallest (k, n) >= target using pre-built KN index
	var targetKN gemmKN
	found := false
	minDist := math.MaxFloat64

	for _, kn := range db.gemmAllKNs {
		if kn.k >= k && kn.n >= n {
			dk := float64(kn.k - k)
			dn := float64(kn.n - n)
			dist := dk*dk + dn*dn // skip sqrt for comparison
			if dist < minDist {
				minDist = dist
				targetKN = kn
				found = true
			}
		}
	}

	if !found {
		// No (k,n) >= target found; use the largest available by magnitude.
		logrus.Debugf("No GEMM (k,n) >= (%d,%d), using largest available", k, n)
		maxMag := -1.0
		for _, kn := range db.gemmAllKNs {
			mag := float64(kn.k)*float64(kn.k) + float64(kn.n)*float64(kn.n)
			if mag > maxMag {
				maxMag = mag
				targetKN = kn
				found = true
			}
		}
	}
	if !found {
		logrus.Warnf("No GEMM data found - returning floor MFU")
		return 0.0001
	}

	// Stage 2: O(1) map lookup + O(log m) interpolation on pre-sorted M-points
	mPoints := db.gemmIndex[targetKN]
	if len(mPoints) == 0 {
		logrus.Warnf("No GEMM data found for (k=%d, n=%d) - returning floor MFU", targetKN.k, targetKN.n)
		return 0.0001
	}

	// Binary search for bracket in pre-sorted M-points
	mVals := make([]int, len(mPoints))
	for i, p := range mPoints {
		mVals[i] = p.m
	}
	lo, hi := bracketIndex(mVals, m)

	var mfu float64
	if lo == hi {
		mfu = mPoints[lo].mfu
	} else {
		t := float64(m-mPoints[lo].m) / float64(mPoints[hi].m-mPoints[lo].m)
		mfu = lerp(mPoints[lo].mfu, mPoints[hi].mfu, t)
	}

	// Handle zero or near-zero MFU: find nearest non-zero by Euclidean distance (I5b)
	if mfu < 0.0001 {
		var bestRow *GEMMRow
		bestDist := math.MaxFloat64
		for i := range db.gemmData {
			if db.gemmData[i].MFU >= 0.0001 {
				dm := float64(db.gemmData[i].M - m)
				dk := float64(db.gemmData[i].K - k)
				dn := float64(db.gemmData[i].N - n)
				dist := dm*dm + dk*dk + dn*dn
				if dist < bestDist {
					bestDist = dist
					bestRow = &db.gemmData[i]
				}
			}
		}
		if bestRow != nil {
			logrus.Debugf("GEMM MFU=%.4f too small for (m=%d, k=%d, n=%d), using nearest (m=%d, k=%d, n=%d) with MFU=%.4f",
				mfu, m, k, n, bestRow.M, bestRow.K, bestRow.N, bestRow.MFU)
			return bestRow.MFU
		}
		logrus.Warnf("All GEMM MFU values are zero - returning floor MFU")
		return 0.0001
	}

	return mfu
}

// H6 MFU Grid-Boundary Discontinuity Sweep
//
// This program sweeps the MFU database across fine-grained parameter ranges
// and outputs CSV files showing the MFU values returned by nearest-neighbor
// lookup. The analyze.py script then counts discontinuities.
//
// Usage: go run mfu_sweep.go --bench-data <path> --gpu h100 --output-dir <dir>
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"github.com/inference-sim/inference-sim/sim"
)

func main() {
	benchData := flag.String("bench-data", "", "Path to InferSim bench_data directory")
	gpu := flag.String("gpu", "h100", "GPU type")
	outputDir := flag.String("output-dir", ".", "Output directory for CSV files")
	flag.Parse()

	if *benchData == "" {
		log.Fatal("--bench-data is required")
	}

	// Use Llama 3.1 8B config (32 heads, 8 KV heads, 128 head_dim)
	modelConfig := sim.ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		VocabSize:       128256,
		BytesPerParam:   2,
		IntermediateDim: 14336,
	}

	mfuDB, err := sim.NewMFUDatabase(modelConfig, *benchData, *gpu)
	if err != nil {
		log.Fatalf("Failed to load MFU database: %v", err)
	}

	// =============================================
	// Sweep 1: GEMM MFU across batch sizes 1-512
	// Fixed K=4096, N=6144 (Q-projection shape for Llama 3.1 8B)
	// =============================================
	fmt.Fprintf(os.Stderr, "Sweep 1: GEMM MFU (K=4096, N=6144) batch_size 1..512\n")
	gemmFile := filepath.Join(*outputDir, "gemm_sweep.csv")
	writeGEMMSweep(mfuDB, gemmFile, 4096, 6144, 1, 512)

	// =============================================
	// Sweep 2: GEMM MFU across batch sizes 1-512
	// Fixed K=4096, N=11008 (MLP gate/up shape for Llama 3.1 8B)
	// =============================================
	fmt.Fprintf(os.Stderr, "Sweep 2: GEMM MFU (K=4096, N=11008) batch_size 1..512\n")
	gemmFile2 := filepath.Join(*outputDir, "gemm_sweep_mlp.csv")
	writeGEMMSweep(mfuDB, gemmFile2, 4096, 11008, 1, 512)

	// =============================================
	// Sweep 3: Decode Attention MFU across batch sizes 1-256
	// Fixed KV lengths: 1024, 4096, 8192
	// =============================================
	for _, kvLen := range []int{1024, 4096, 8192} {
		fmt.Fprintf(os.Stderr, "Sweep 3: Decode Attn MFU (kv=%d) batch_size 1..256\n", kvLen)
		decFile := filepath.Join(*outputDir, fmt.Sprintf("decode_attn_kv%d_sweep.csv", kvLen))
		writeDecodeAttnSweep(mfuDB, decFile, kvLen, 1, 256, 1)
	}

	// =============================================
	// Sweep 4: Decode Attention MFU across KV lengths 128-16384
	// Fixed batch sizes: 1, 32, 128
	// =============================================
	for _, bs := range []int{1, 32, 128} {
		fmt.Fprintf(os.Stderr, "Sweep 4: Decode Attn MFU (bs=%d) kv 128..16384\n", bs)
		decFile := filepath.Join(*outputDir, fmt.Sprintf("decode_attn_bs%d_kvsweep.csv", bs))
		writeDecodeAttnKVSweep(mfuDB, decFile, bs, 128, 16384, 64, 1)
	}

	// =============================================
	// Sweep 5: Prefill Attention MFU across sequence lengths 512-32768
	// Step size: 64
	// =============================================
	fmt.Fprintf(os.Stderr, "Sweep 5: Prefill Attn MFU seq_len 512..32768\n")
	prefillFile := filepath.Join(*outputDir, "prefill_attn_sweep.csv")
	writePrefillAttnSweep(mfuDB, prefillFile, 512, 32768, 64)

	fmt.Fprintf(os.Stderr, "All sweeps complete. Output in %s\n", *outputDir)
}

func writeGEMMSweep(mfuDB *sim.MFUDatabase, outPath string, k, n, minM, maxM int) {
	f, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("Create %s: %v", outPath, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()
	w.Write([]string{"m", "k", "n", "mfu"})

	for m := minM; m <= maxM; m++ {
		mfu := mfuDB.GetGEMMmfu(m, k, n)
		w.Write([]string{
			strconv.Itoa(m),
			strconv.Itoa(k),
			strconv.Itoa(n),
			fmt.Sprintf("%.6f", mfu),
		})
	}
}

func writeDecodeAttnSweep(mfuDB *sim.MFUDatabase, outPath string, kvLen, minBS, maxBS, tp int) {
	f, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("Create %s: %v", outPath, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()
	w.Write([]string{"batch_size", "kv_len", "tp", "mfu"})

	for bs := minBS; bs <= maxBS; bs++ {
		mfu := mfuDB.GetAttnDecodeMFU(bs, kvLen, tp)
		w.Write([]string{
			strconv.Itoa(bs),
			strconv.Itoa(kvLen),
			strconv.Itoa(tp),
			fmt.Sprintf("%.6f", mfu),
		})
	}
}

func writeDecodeAttnKVSweep(mfuDB *sim.MFUDatabase, outPath string, bs, minKV, maxKV, step, tp int) {
	f, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("Create %s: %v", outPath, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()
	w.Write([]string{"batch_size", "kv_len", "tp", "mfu"})

	for kv := minKV; kv <= maxKV; kv += step {
		mfu := mfuDB.GetAttnDecodeMFU(bs, kv, tp)
		w.Write([]string{
			strconv.Itoa(bs),
			strconv.Itoa(kv),
			strconv.Itoa(tp),
			fmt.Sprintf("%.6f", mfu),
		})
	}
}

func writePrefillAttnSweep(mfuDB *sim.MFUDatabase, outPath string, minSeq, maxSeq, step int) {
	f, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("Create %s: %v", outPath, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()
	w.Write([]string{"seq_len", "mfu"})

	for seq := minSeq; seq <= maxSeq; seq += step {
		mfu := mfuDB.GetAttnPrefillMFU(seq)
		w.Write([]string{
			strconv.Itoa(seq),
			fmt.Sprintf("%.6f", mfu),
		})
	}
}

// Moved from blis-latency-kernel/bench_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
)

// StepTime is called once per simulated step, millions of times over a run, so its cost
// and its allocation count both matter. The per-stage composition walks one entry per
// distinct layer KIND rather than per layer — three entries for a 72-layer model — and
// accumulates into a fixed array. What allocates is the published result map and a few
// per-request scratch slices; -benchmem reports both.
func BenchmarkStepTimeDecode(b *testing.B) {
	k := benchKernel(b)
	batch := decodeBatch(256, 2, 8192)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.StepTime(batch)
	}
}

func BenchmarkStepTimeSingleRequest(b *testing.B) {
	k := benchKernel(b)
	batch := decodeBatch(1, 1, 8192)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.StepTime(batch)
	}
}

func BenchmarkStepTimeLargeBatch(b *testing.B) {
	// The batch loop is the one term that walks the requests, so a wide batch shows its
	// cost against the per-kind loop's fixed size.
	k := benchKernel(b)
	batch := decodeBatch(1024, 1, 32768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.StepTime(batch)
	}
}

func BenchmarkMemoryQueries(b *testing.B) {
	k := benchKernel(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.SequenceVariableBytes(8192)
		_ = k.FixedBytes()
	}
}

// benchKernel builds a kernel once, outside the timed loop.
func benchKernel(b *testing.B) *Kernel {
	b.Helper()
	return fixture(b, "minimax-m25-h200-ep8.yaml")
}

// BenchmarkStepTimeIntoDecode is the form a simulator's inner loop uses: one reused map, so
// what it allocates is only the per-request scratch.
func BenchmarkStepTimeIntoDecode(b *testing.B) {
	k := benchKernel(b)
	batch := decodeBatch(256, 2, 8192)
	per := make(map[kernel.Resource]time.Duration, 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.StepTimeInto(batch, per)
	}
}

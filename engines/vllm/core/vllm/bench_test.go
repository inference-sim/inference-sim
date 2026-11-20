// Original to blis-vllm.

package vllm

import (
	"fmt"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// BenchmarkEngineLongDecodes drives engine.Engine (Add, then Schedule and Complete) with 16
// requests that each decode 256 tokens after a prompt of the stated length, so a cost that grows
// with the context shows as a gap between the two sizes (go test -bench . -benchmem ./core/vllm).
func BenchmarkEngineLongDecodes(b *testing.B) {
	for _, prompt := range []int{512, 16384} {
		b.Run(fmt.Sprintf("prompt=%d", prompt), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				extra := fmt.Sprintf("max_num_seqs: 16\nmax_num_batched_tokens: 8192\nmax_model_len: %d\nnum_gpu_blocks: %d\n",
					prompt+512, 16*((prompt+512)/16+2)+8)
				e, err := build(b, extra)()
				if err != nil {
					b.Fatal(err)
				}
				for j := 0; j < 16; j++ {
					if err := e.Add(req(fmt.Sprintf("r%d", j), prompt, 256), 0); err != nil {
						b.Fatal(err)
					}
				}
				b.StartTimer()
				now := engine.Time(0)
				for steps := 0; steps < 100000; steps++ {
					bt, err := e.Schedule(now)
					if err != nil {
						b.Fatal(err)
					}
					if bt.Empty() {
						break
					}
					now += 1000
					if _, err := e.Complete(bt.ID, engine.Outcome{}, now); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

package lower

import (
	"testing"
)

// A pool that enables expert parallel spans its experts over the tensor-parallel group, which is
// SGLang's ep_size (--tp N --ep-size N). It was refused as parallelism beyond tensor and data.
func TestAnExpertParallelPoolStatesEpSizeEqualToTheTensorParallelWidth(t *testing.T) {
	d := pool(t, 8, "mem_fraction_static: 0.82\n")
	d.Pools[0].Parallel.EnableExpertParallel = true
	args, err := Args(PoolInput{Deployment: d})
	if err != nil {
		t.Fatalf("an expert-parallel pool: %v", err)
	}
	if got, ok := asInt(args["ep_size"]); !ok || got != 8 {
		t.Errorf("ep_size = %v, want the pool's tensor-parallel width 8", args["ep_size"])
	}
	// The resolution accepts it and carries it.
	d = pool(t, 8, "mem_fraction_static: 0.82\n")
	d.Pools[0].Parallel.EnableExpertParallel = true
	res, err := Resolve(PoolInput{Deployment: d, Model: graph(t, "deepseek-r1-0528"), Chip: h200(t)})
	if err != nil {
		t.Fatalf("resolving an expert-parallel pool: %v", err)
	}
	if res.Args.EpSize != 8 {
		t.Errorf("resolved ep_size %d, want 8", res.Args.EpSize)
	}
}

// What cannot be stated as an ep_size stays refused: data parallelism beside it (the experts would
// span tp x dp), and an engine block that states another width.
func TestExpertParallelStaysRefusedWhereEpSizeCannotStateIt(t *testing.T) {
	d := pool(t, 8, "mem_fraction_static: 0.82\n")
	d.Pools[0].Parallel.EnableExpertParallel, d.Pools[0].Parallel.DP = true, 2
	if _, err := Args(PoolInput{Deployment: d}); !refused(err) {
		t.Errorf("expert parallel beside data parallel: want a NotLowered refusal, got %v", err)
	}
	d = pool(t, 8, "ep_size: 4\n")
	d.Pools[0].Parallel.EnableExpertParallel = true
	if _, err := Args(PoolInput{Deployment: d}); err == nil {
		t.Error("a block stating ep_size 4 beside an expert-parallel pool of tp 8 was accepted")
	}
}

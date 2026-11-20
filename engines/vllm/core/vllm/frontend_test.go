// Original to blis-vllm.

package vllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

const dpBlock = "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 32\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
	"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\n"

func frontendEngine(t *testing.T, extra string) engine.Engine {
	t.Helper()
	e, err := build(t, extra)()
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// An engine states the queue requests wait in before it admits them from its API-server
// settings: renderer_num_workers threads on each of api_server_count servers. Unset, they are
// vLLM's defaults, one thread on one server; a data-parallel group behind the internal balancer
// has one server per rank unless the count is stated.
func TestThePlanStatesTheAdmissionQueueFromTheAPIServerSettings(t *testing.T) {
	type frontender interface {
		engine.Planned
		Frontender
	}
	one := frontendEngine(t, dpBlock).(frontender)
	if got := one.Plan().Host.Admission; got != (plan.AdmissionQueue{Servers: 1, WorkersPerServer: 1}) {
		t.Errorf("unset: %+v, want vLLM's one thread on one server", got)
	}
	for _, c := range []struct {
		f    Frontend
		want plan.AdmissionQueue
	}{
		{Frontend{RendererNumWorkers: 2}, plan.AdmissionQueue{Servers: 1, WorkersPerServer: 2}},
		{Frontend{RendererNumWorkers: 1, APIServerCount: 3}, plan.AdmissionQueue{Servers: 3, WorkersPerServer: 1}},
		{Frontend{}, plan.AdmissionQueue{Servers: 1}},
	} {
		if err := one.SetFrontend(c.f); err != nil {
			t.Fatal(err)
		}
		if got := one.Plan().Host.Admission; got != c.want {
			t.Errorf("%+v: %+v, want %+v", c.f, got, c.want)
		}
	}
	if one.SetFrontend(Frontend{RendererNumWorkers: -1}) == nil {
		t.Error("a negative renderer_num_workers was accepted")
	}

	grp := frontendEngine(t, dpBlock+"data_parallel_size: 4\n").(frontender)
	if got := grp.Plan().Host.Admission; got != (plan.AdmissionQueue{Servers: 4, WorkersPerServer: 1}) {
		t.Errorf("a group of four ranks: %+v, want four servers of one thread", got)
	}
	if err := grp.SetFrontend(Frontend{RendererNumWorkers: 1, APIServerCount: 2}); err != nil {
		t.Fatal(err)
	}
	if got := grp.Plan().Host.Admission; got.Servers != 2 {
		t.Errorf("a stated api_server_count of 2 gave %d servers", got.Servers)
	}
}

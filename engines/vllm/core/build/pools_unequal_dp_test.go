// Original to blis-vllm.

package build

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// P/D pools are separate `vllm serve` processes with their own parallel config; nothing in vLLM ties the prefill
// engine's data-parallel size to the decode engine's (inference-sim/inference-sim#1952). The host counts and places
// instances by data-parallel group, so each pool is counted by its own group size.

const (
	poolTP8DP1 = "      tp: 8\n      pp: 1\n      dp: 1\n      enable_expert_parallel: false\n"
	poolTP4DP2 = "      tp: 4\n      pp: 1\n      dp: 2\n      enable_expert_parallel: true\n"
)

// unequalDP is the 3P1D fixture with the pools' parallel blocks replaced: prefill (first block), decode (second).
func unequalDP(t *testing.T, prefill, decode string) (scenarios string) {
	t.Helper()
	repos := pdVariant(t, "", "")
	body := readScenario(t, repos)
	if strings.Count(body, poolTP8DP1) != 2 {
		t.Fatalf("the fixture no longer states the tp 8 dp 1 block once in each pool")
	}
	i := strings.Index(body, poolTP8DP1)
	body = body[:i] + prefill + body[i+len(poolTP8DP1):]
	j := strings.LastIndex(body, poolTP8DP1)
	body = body[:j] + decode + body[j+len(poolTP8DP1):]
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// A decode pool at dp 2 behind a dp 1 prefill pool opens, and so does the reverse; each pool holds as many instances
// as its own GPUs hold groups of its own size, and a handoff is priced between the placed pairs.
func TestOpenPools_PoolsOfDifferentDataParallelWidthOpen(t *testing.T) {
	for _, c := range []struct {
		name            string
		prefill, decode string
		top             Topology
		wantPrefillCap  int
		wantDecodeCap   int
	}{
		{"decode wider in dp", poolTP8DP1, poolTP4DP2, Topology{NumInstances: 4, Prefill: 3, Decode: 1}, 3, 1},
		{"prefill wider in dp", poolTP4DP2, poolTP8DP1, Topology{NumInstances: 4, Prefill: 3, Decode: 1}, 3, 1},
	} {
		repos := vendoredRepos(t)
		repos.Scenarios = unequalDP(t, c.prefill, c.decode)
		p, _, err := OpenPools("variant.yaml", repos, true, c.top)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if got := p.Prefill.GroupCapacity(); got != c.wantPrefillCap {
			t.Errorf("%s: the prefill pool holds %d groups, want %d", c.name, got, c.wantPrefillCap)
		}
		if got := p.Decode.GroupCapacity(); got != c.wantDecodeCap {
			t.Errorf("%s: the decode pool holds %d groups, want %d", c.name, got, c.wantDecodeCap)
		}
		price, _, err := p.HandoffPricing("variant.yaml", false, c.top, 64, noFatal(t))
		if err != nil || price == nil {
			t.Errorf("%s: pricing: %v", c.name, err)
			continue
		}
		want := p.Prefill.PDTransferTicks(1024, p.Prefill.PlacementOf(0), p.Decode.PlacementOf(0))
		if got := price(1024, "instance_0", "instance_3"); got != want || got <= 0 {
			t.Errorf("%s: instance_0 -> instance_3 priced %d ticks, want the kernel's %d", c.name, got, want)
		}
		if err := RequireGroupCapacity("variant.yaml", p.Prefill, p, c.top); err != nil {
			t.Errorf("%s: %v", c.name, err)
		}
	}
}

// The checks that are real constraints stay: pools that page in different block sizes are still refused when their
// data-parallel widths differ.
func TestOpenPools_DifferentWidthsStillNeedMatchingBlocks(t *testing.T) {
	repos := vendoredRepos(t)
	dir := unequalDP(t, poolTP8DP1, poolTP4DP2)
	raw, _ := os.ReadFile(filepath.Join(dir, "variant.yaml"))
	body := string(raw)
	k := strings.LastIndex(body, "block_size: 64")
	body = body[:k] + "block_size: 128" + body[k+len("block_size: 64"):]
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	repos.Scenarios = dir
	_, _, err := OpenPools("variant.yaml", repos, true, Topology{NumInstances: 4, Prefill: 3, Decode: 1})
	if err == nil || !strings.Contains(err.Error(), "whole blocks") {
		t.Errorf("pools paging differently: %v, want the block-size refusal", err)
	}
}

package cmd

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// p2pRun runs pd-p2p.yaml (two prefill nodes, one decode node, six requests) with the prefill pool at
// prefillTP and the decode pool at decodeTP, both with the V1 model runner when v1 is set (enable_dbo
// makes vLLM's V2 runner unsupported, and the KV layout then parallelism-agnostic), and returns the
// Result and stderr.
func p2pRun(t *testing.T, prefillTP, decodeTP string, v1 bool) (capDoc, string) {
	t.Helper()
	dir := exampleVariant(t, "pd-p2p.yaml", "tp: 8", "tp: "+prefillTP)
	path := filepath.Join(dir, "pd-p2p.yaml")
	raw := readFile(t, path)
	i := strings.Index(raw, "  - role: decode")
	raw = raw[:i] + strings.Replace(raw[i:], "tp: 8", "tp: "+decodeTP, 1)
	if v1 {
		raw = strings.ReplaceAll(raw, "      block_size: 16\n", "      block_size: 16\n      enable_dbo: true\n")
	}
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := runKernelCLI(t, "run", "--scenario", "pd-p2p.yaml", "--num-instances", "3", "--prefill-instances", "2",
		"--decode-instances", "1", "--rate", "2", "--num-requests", "6", "--seed", "1", "--scenarios", dir)
	if err != nil {
		t.Fatalf("p2p run tp %s/%s: %v\n%s", prefillTP, decodeTP, err, lastLines(stderr, 6))
	}
	return capDoc{t: t, stdout: stdout, doc: resultDoc(t, stdout)}, stderr
}

func pdLinks(d capDoc) map[string]bool {
	links := map[string]bool{}
	keysWithPrefix(d.doc, "link_peak_concurrent_transfers_", links)
	return links
}

// The Result's kv_transfer figures of a p2p run are the WRITEs that moved the prompts: the handoff
// itself is not a pull of the cluster's, so they were all zero (inference-sim#2220). Their sum is the
// WRITEs' summed duration, which the Result states beside them.
func TestP2PHandoffKVTransferReportsTheWrites(t *testing.T) {
	d, stderr := p2pRun(t, "4", "4", false)
	n := d.num("breakdowns.pd.kv_transfers")
	mean := d.num("breakdowns.pd.kv_transfer_mean_ms")
	if mean <= 0 {
		t.Fatalf("kv_transfer_mean_ms = %v, want the WRITEs' time", mean)
	}
	if total := d.num("breakdowns.pd.p2p_write_total_ms"); math.Abs(mean*n-total) > 0.01*total {
		t.Errorf("kv_transfer_mean_ms x kv_transfers = %.4f ms, the WRITEs' summed duration is %.4f ms", mean*n, total)
	}
	if w, h := d.num("breakdowns.pd.p2p_writes"), d.num("breakdowns.pd.p2p_handoffs"); w != h || h != 6 || d.num("breakdowns.pd.p2p_handoffs_without_write") != 0 {
		t.Errorf("%v WRITEs for %v handoffs (%v without), want one each of 6", w, h, d.num("breakdowns.pd.p2p_handoffs_without_write"))
	}
	if !pdLinks(d)["link_peak_concurrent_transfers_nic-egress-node-0"] {
		t.Errorf("the WRITE did not cross the prefill node's NIC: %v", pdLinks(d))
	}
	if strings.Contains(stderr, "moved no KV") {
		t.Errorf("a run whose handoffs all moved KV warned: %s", lastLines(stderr, 3))
	}
}

// With pools of unequal tensor parallelism and vLLM's default V2 model runner, the p2p tier's connect
// handshake rejects the peer (its run config names the tensor-parallel size), so vLLM, and the engine
// that reproduces it, go on without the prefill engine's KV: no WRITE is made and none can be priced.
// The run says so (inference-sim#2214, #2220) instead of reporting an unpriced success.
func TestP2PUnequalTPWithoutAgnosticLayoutSaysNoKVMoved(t *testing.T) {
	d, stderr := p2pRun(t, "4", "8", false)
	if n, h := d.num("breakdowns.pd.p2p_handoffs_without_write"), d.num("breakdowns.pd.p2p_handoffs"); h != 6 || n != h {
		t.Errorf("%v of %v p2p handoffs without a WRITE, want all 6", n, h)
	}
	if d.num("breakdowns.pd.p2p_writes") != 0 {
		t.Errorf("a rejected peer made %v WRITEs", d.num("breakdowns.pd.p2p_writes"))
	}
	if hit := d.num("breakdowns.pd.cpu_tier_hit_rate_decode"); hit != 0 {
		t.Errorf("the decode engine loaded %.3f of its prompts from the tier although no KV moved", hit)
	}
	for _, want := range []string{"6 of 6 p2p handoffs moved no KV", "tp 4", "tp 8", "inference-sim#2214"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr lacks %q: %s", want, lastLines(stderr, 4))
		}
	}
}

// Where the handshake succeeds (a parallelism-agnostic layout), the WRITE is priced over the NIC from
// the prefill node to the decode node whatever the tensor parallelism of either pool: unequal pools
// show the NIC links, a decode-side tier load and a transfer time, as equal pools do.
func TestP2PUnequalTPWithAgnosticLayoutPricesTheNICWrite(t *testing.T) {
	d, stderr := p2pRun(t, "4", "8", true)
	links := pdLinks(d)
	for _, l := range []string{"nic-egress-node-0", "nic-ingress-node-2", "tier-cpu_dram-from-node-2"} {
		if !links["link_peak_concurrent_transfers_"+l] {
			t.Errorf("unequal-TP p2p run has no %s link: %v", l, links)
		}
	}
	if d.num("breakdowns.pd.p2p_handoffs_without_write") != 0 || d.num("breakdowns.pd.kv_transfer_mean_ms") <= 0 {
		t.Errorf("handoffs without a WRITE: %v, kv_transfer_mean_ms %v", d.num("breakdowns.pd.p2p_handoffs_without_write"), d.num("breakdowns.pd.kv_transfer_mean_ms"))
	}
	if hit := d.num("breakdowns.pd.cpu_tier_hit_rate_decode"); hit < 0.9 {
		t.Errorf("the decode engine loaded %.3f of its prompts from the tier, want nearly all", hit)
	}
	if strings.Contains(stderr, "moved no KV") {
		t.Errorf("a run whose handoffs all moved KV warned: %s", lastLines(stderr, 3))
	}
}

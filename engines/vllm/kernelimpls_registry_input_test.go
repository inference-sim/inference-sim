// Original to blis-vllm.

package blisvllm

import (
	"bytes"
	"os"
	"testing"
)

// The substrate reads no file of an engine module (X7), so VAL-8's kernel-implementation tables
// reach it as registry/inputs/kernel_impls.<engine>.json, written by
// tools/kernelimpls/export_kernel_impls.py. This engine's copy must be the table it embeds, or
// the cross-check would hold the registry to kernel hashes this build does not publish.
func TestTheRegistryInputIsTheKernelImplTableThisBuildEmbeds(t *testing.T) {
	const path = "../../substrate/registry/inputs/kernel_impls." + Kind + ".json"
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, KernelImplsJSON) {
		t.Errorf("%s is not the kernel_impls.json this build embeds; run substrate/tools/kernelimpls/export_kernel_impls.py "+
			"--out-dir substrate/registry/inputs %s=engines/%s/kernel_impls.json", path, Kind, Kind)
	}
}

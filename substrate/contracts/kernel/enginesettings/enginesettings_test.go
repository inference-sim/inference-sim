// Original to blis-schemas.

package enginesettings

import (
	"reflect"
	"testing"
)

// Clone is a deep copy: editing every pointer, slice and nested value of the original leaves
// the clone as it was, so a kernel keeps the settings it was built from.
func TestCloneIsADeepCopy(t *testing.T) {
	on, size, off := true, 512, false
	orig := Settings{
		MaxModelLen: 4096, EnablePrefixCaching: &on, AsyncScheduling: &on, DisableCascadeAttn: &on,
		DisableCustomAllReduce: &on, DCPQReplicate: &on, MaxCUDAGraphCaptureSize: &size,
		CUDAGraphCaptureSizes: []int{1, 2, 4}, DBO: &DBO{Enabled: true},
		EPLB:         &EPLB{Enabled: true, NumRedundantExperts: 2},
		Speculative:  &Speculative{Method: "mtp", NumSpecTokens: 2, DraftModel: "drafter", EnableAdaptiveVerification: true},
		EngramConfig: &EngramConfig{CPUOffload: &on, DPSharedMemory: &on, UseTHP: &on},
	}
	c := orig.Clone()
	for _, b := range []*bool{orig.EnablePrefixCaching, orig.AsyncScheduling, orig.DisableCascadeAttn,
		orig.DisableCustomAllReduce, orig.DCPQReplicate} {
		*b = off
	}
	for name, b := range map[string]*bool{"EnablePrefixCaching": c.EnablePrefixCaching,
		"AsyncScheduling": c.AsyncScheduling, "DisableCascadeAttn": c.DisableCascadeAttn,
		"DisableCustomAllReduce": c.DisableCustomAllReduce, "DCPQReplicate": c.DCPQReplicate} {
		if !*b {
			t.Errorf("editing the original's %s changed its clone", name)
		}
	}
	on2 := true
	orig.EnablePrefixCaching = &on2
	*orig.EnablePrefixCaching, *orig.MaxCUDAGraphCaptureSize = off, 8
	orig.CUDAGraphCaptureSizes[0] = 99
	orig.DBO.Enabled, orig.EPLB.NumRedundantExperts, orig.Speculative.NumSpecTokens = false, 0, 0
	orig.Speculative.DraftModel = ""
	orig.Speculative.EnableAdaptiveVerification = false
	*orig.EngramConfig.CPUOffload = off
	switch {
	case !*c.EnablePrefixCaching || *c.MaxCUDAGraphCaptureSize != 512:
		t.Error("editing the original's pointers changed its clone")
	case !reflect.DeepEqual(c.CUDAGraphCaptureSizes, []int{1, 2, 4}):
		t.Errorf("editing the original's capture sizes changed its clone: %v", c.CUDAGraphCaptureSizes)
	case !c.DBO.Enabled || c.EPLB.NumRedundantExperts != 2 || c.Speculative.NumSpecTokens != 2 ||
		c.Speculative.DraftModel != "drafter" || !c.Speculative.EnableAdaptiveVerification:
		t.Error("editing the original's nested settings changed its clone")
	case !*c.EngramConfig.CPUOffload:
		t.Error("editing the original's Engram settings changed its clone")
	}
}

// MaxModelLen states a count, auto, or nothing.
func TestMaxModelLen(t *testing.T) {
	if n, ok := MaxModelLen(8192).Stated(); !ok || n != 8192 {
		t.Errorf("8192: Stated %d, %t", n, ok)
	}
	if _, ok := MaxModelLenAuto.Stated(); ok || !MaxModelLenAuto.Auto() {
		t.Error("auto: stated, or not auto")
	}
	if _, ok := MaxModelLen(0).Stated(); ok || MaxModelLen(0).Auto() {
		t.Error("unstated: stated, or auto")
	}
}

// The derived predicates read the fields they name.
func TestPredicates(t *testing.T) {
	if !(Settings{Frontend: FrontendRust}).RustFrontend() || (Settings{}).RustFrontend() {
		t.Error("RustFrontend")
	}
	if !(Settings{Speculative: &Speculative{NumSpecTokens: 1}}).Speculating() ||
		(Settings{Speculative: &Speculative{}}).Speculating() || (Settings{}).Speculating() {
		t.Error("Speculating")
	}
}

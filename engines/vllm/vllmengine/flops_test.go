package vllmengine

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// The engine states the model FLOPs of the step it scheduled, and one GPU's peak, so a host can
// compute a model-FLOPs utilization (engine.Flopped, inference-sim#2081): a longer prompt costs
// more FLOPs, and the utilization of its step is a fraction.
func TestEngineStatesStepFLOPsAndPeak(t *testing.T) {
	flops := func(prompt int) (f, secs, peak float64) {
		e, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.34.0+0fda86a35419", Roots: roots(t, "d058-032.yaml")}})
		if err != nil {
			t.Fatal(err)
		}
		fl, ok := e.(engine.Flopped)
		if !ok {
			t.Fatalf("%T does not implement engine.Flopped", e)
		}
		if err := e.Add(req("a", prompt, 1), 0); err != nil {
			t.Fatal(err)
		}
		b, err := e.Schedule(0)
		if err != nil || b.Empty() {
			t.Fatalf("schedule: %v, empty %v", err, b.Empty())
		}
		return fl.StepFLOPs(b), float64(e.StepTime(b)) / 1e6, fl.PeakFLOPsPerGPU()
	}
	short, _, peak := flops(256)
	long, secs, _ := flops(2048)
	if short <= 0 || peak <= 0 {
		t.Fatalf("step FLOPs %v, peak %v; want both stated", short, peak)
	}
	if long < 6*short {
		t.Errorf("2048-token step has %.3g FLOPs against %.3g for 256 tokens; want at least 6x", long, short)
	}
	if u := long / secs / peak; u <= 0 || u > 8 {
		t.Errorf("a step's FLOPs over its time and one GPU's peak is %.3f; want a fraction of the GPUs the pool has", u)
	}
}

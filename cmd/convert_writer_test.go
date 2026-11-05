package cmd

import (
	"bytes"
	"errors"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim/workload"
)

type refusingWriter struct{}

var errRefusedWrite = errors.New("broken pipe")

func (refusingWriter) Write([]byte) (int, error) { return 0, errRefusedWrite }

func convertTestSpec(t *testing.T) *workload.WorkloadSpec {
	t.Helper()
	spec, err := workload.ConvertPreset("chatbot", 10, 20, workload.PresetConfig{
		PromptTokensMean: 100, PromptTokensStdev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
		OutputTokensMean: 50, OutputTokensStdev: 5, OutputTokensMin: 5, OutputTokensMax: 100,
	})
	if err != nil {
		t.Fatal(err)
	}
	return spec
}

// SIM-2, #661: blis convert and blis compose print the spec on stdout; a write that fails
// (a closed pipe) is the command's error, not a silent truncation.
func TestConvert_AFailingSpecWriteIsReturned(t *testing.T) {
	if err := writeSpec(refusingWriter{}, convertTestSpec(t)); !errors.Is(err, errRefusedWrite) {
		t.Fatalf("a spec that was never written was reported as written: err = %v", err)
	}
}

// The bytes a successful write produces are the spec's YAML, unchanged.
func TestConvert_ASucceedingSpecWriteIsTheSpecYAML(t *testing.T) {
	spec := convertTestSpec(t)
	var b bytes.Buffer
	if err := writeSpec(&b, spec); err != nil {
		t.Fatal(err)
	}
	want, err := yaml.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b.Bytes(), want) {
		t.Errorf("writeSpec wrote %d bytes, want the %d bytes of yaml.Marshal", b.Len(), len(want))
	}
}

// SIM-2, #661: a Result document that cannot be written to stdout is the run's error, after
// --output holds the whole document.
func TestResult_AFailingStdoutIsReturned(t *testing.T) {
	out, _, err := runKernelCLI(t, resultRunArgs()...)
	if err != nil {
		t.Fatal(err)
	}
	r := parseResult(t, out)
	if werr := writeResult(refusingWriter{}, r); !errors.Is(werr, errRefusedWrite) {
		t.Fatalf("a Result that never reached stdout was reported as written: err = %v", werr)
	}
}

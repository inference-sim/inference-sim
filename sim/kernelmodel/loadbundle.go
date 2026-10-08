package kernelmodel

import (
	"errors"
	"fmt"
	"io"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// loadBundle reads a scenario and the deployment applied to it from one file, as two YAML
// documents separated by `---`.
//
// blis-schemas v0.2.0 keeps the two apart: a Scenario is the immutable problem (model,
// cluster inventory, coefficient and engine-version references) and a Deployment is the
// tunable configuration chosen against it (pools, each a parallelism layout with its own
// engine settings). One simulated run is one of each.
//
// They share a file because a measurement row addresses a deployment by a single filename
// -- `{"scenario": "glm-5-h200-fp8-sglang-tp8.yaml", ...}` -- and testdata/measurements
// holds thousands of such rows. A sibling-file layout would rewrite every one of them to
// say nothing it does not already say.
//
// WHY THIS IS DUPLICATED. blis-latency-kernel has the same function in its
// internal/harness package, and Go forbids importing another module's internal packages.
// The duplication is deliberate and bounded: it is loading, not modelling, and it holds
// no cost logic -- the same reason this file's sibling kernelmodel.go already reproduces
// that harness's Open. If the kernel ever promotes harness out of internal/, this should
// be deleted in favour of it.
//
// blis-schemas' own LoadScenario and LoadDeployment each open a path and read ONE
// document, so neither can reach the second half of a pair; this reproduces their strict
// decoding over a single stream instead. Strictness is the point: a misspelled key that
// parsed silently would leave a document that validates while omitting the setting its
// author intended, which is the failure mode hardest to notice.
func loadBundle(path string) (*scenario.Scenario, *deployment.Deployment, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)

	var sc scenario.Scenario
	if err := dec.Decode(&sc); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, nil, fmt.Errorf("%s is empty: a kernel needs a scenario and the "+
				"deployment applied to it", path)
		}
		return nil, nil, fmt.Errorf("decoding the scenario in %s: %w", path, err)
	}

	var dep deployment.Deployment
	if err := dec.Decode(&dep); err != nil {
		// A scenario with no deployment cannot build a kernel: there is no pool to price.
		// Saying so here names the file, where failing later would surface as an
		// out-of-range pool index far from the document that lacks one.
		if errors.Is(err, io.EOF) {
			return nil, nil, fmt.Errorf("%s holds a scenario but no deployment: the pools "+
				"a kernel prices live in a second YAML document, separated by `---`", path)
		}
		return nil, nil, fmt.Errorf("decoding the deployment in %s: %w", path, err)
	}
	return &sc, &dep, nil
}

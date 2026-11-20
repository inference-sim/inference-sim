// Moved from inference-sim/cmd/root.go at 5a1cb01f728efa858566e4caeae5a1cc9d2294dc.

package build

import (
	"fmt"
	"slices"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	vllm "github.com/inference-sim/blis-vllm/core/vllm"
)

// AdoptInputs name a run's scenario and the data it reads, as the command line gave them
// (--scenario, --scenarios, and the catalog and registry the simulator opened with
// blis-substrate's loader), and whether the command line states a P/D topology.
type AdoptInputs struct {
	Scenario, ScenariosDir string
	// Data is the run's catalog and registry, opened and checked by blis-substrate's loader
	// (load.Embedded, or load.Open with --catalog, --registry and the overlay flags).
	Data *load.Data
	// PDTopology is whether --prefill-instances or --decode-instances is non-zero.
	PDTopology bool
	// BeforeOpen runs after the shape is read and the topology checked, before the first pool
	// opens: the caller hands its routing overrides to the adapter there, as before.
	BeforeOpen func(kernelmodel.Shape) error
}

// Adopted is a run's deployment as its scenario states it and the kernel sizes it.
type Adopted struct {
	Shape         kernelmodel.Shape
	Disaggregated bool
	// Model is the kernel for the scenario's first pool.
	Model    *kernelmodel.Model
	Settings kernelmodel.Settings
	// Deployment is the first pool's model, hardware and parallelism.
	Deployment kernelmodel.Deployment
}

// Adopt resolves the deployment from the scenario, and refuses a run with no scenario,
// scenario directory or data.
//
// The scenario is the committed record of what was deployed: it states the model, the
// hardware, and each pool's tensor- and data-parallel width. Accepting the same facts as
// flags too would mean writing precedence logic to decide between two sources, which is
// complexity carrying no information -- so no flag restates them, and the rest are derived.
func Adopt(in AdoptInputs) (Adopted, error) {
	var missing []string
	if in.Scenario == "" {
		missing = append(missing, "--scenario")
	}
	if in.ScenariosDir == "" {
		missing = append(missing, "--scenarios")
	}
	if len(missing) > 0 {
		return Adopted{}, fmt.Errorf("blis run requires %s: the kernel prices a step from a "+
			"committed scenario plus the catalog and registry it names",
			strings.Join(missing, ", "))
	}
	if in.Data == nil {
		return Adopted{}, fmt.Errorf("scenario %q: no catalog and registry: AdoptInputs.Data is unset; "+
			"open them with blis-substrate's load.Embedded or load.Open", in.Scenario)
	}
	// The scenario's roles and the CLI topology describe one deployment. A disaggregated
	// scenario run without a P/D topology would serve every request on its first pool's
	// engine, as though colocated; the opposite mismatch is refused when the pools open.
	shape, err := kernelmodel.ShapeOf(in.Scenario, kernelmodel.Repos{Scenarios: in.ScenariosDir})
	if err != nil {
		return Adopted{}, fmt.Errorf("scenario %q: %v", in.Scenario, err)
	}
	// One build models one engine release (#2021 B6): a deployment stating vllm.Version runs on
	// blis-vllm's engine for it (core/vllm, through vllmengine). Any other release is refused
	// rather than run as that one.
	if shape.EngineVersion != vllm.Version {
		stated := fmt.Sprintf("states engine release %q", shape.EngineVersion)
		if shape.EngineVersion == "" {
			stated = "states no engine release (engine.version, or the deprecated engine_version)"
		}
		return Adopted{}, fmt.Errorf("scenario %q %s; blis-vllm models the release %s: state engine.version %q",
			in.Scenario, stated, vllm.Version, vllm.Version)
	}
	// A colocated run simulates one pool's engine; a second colocated pool would be dropped.
	if !slices.Contains(shape.Roles, deployment.RolePrefill) && len(shape.Roles) > 1 {
		return Adopted{}, fmt.Errorf("scenario %q states %d colocated pools; a run simulates one colocated "+
			"engine, so state one pool and scale it with --num-instances", in.Scenario, len(shape.Roles))
	}
	return Adopted{Shape: shape, Disaggregated: slices.Contains(shape.Roles, deployment.RolePrefill)}, nil
}

// OpenFirstPool opens the scenario's first pool and sizes it, once the caller has taken the
// shape's knobs and router inputs (Adopt): a disaggregated scenario needs a P/D topology.
func OpenFirstPool(in AdoptInputs, a *Adopted) error {
	if a.Disaggregated && !in.PDTopology {
		return fmt.Errorf("scenario %q is disaggregated (it states prefill and decode "+
			"pools), so the run needs a P/D topology: pass --prefill-instances and --decode-instances",
			in.Scenario)
	}
	if in.BeforeOpen != nil {
		if err := in.BeforeOpen(a.Shape); err != nil {
			return err
		}
	}
	m, err := kernelmodel.Open(in.Scenario, kernelmodel.Repos{Scenarios: in.ScenariosDir, Data: in.Data})
	if err != nil {
		// Refused rather than fallen back on: there is no other latency model to serve the run.
		return fmt.Errorf("scenario %q: %v", in.Scenario, err)
	}
	// blis-vllm's engine checks the KV cache at start-up over the model's groups as vLLM does
	// (kernelmodel.SettingsCheckedByGroups); a single-group check would refuse a hybrid model
	// vLLM starts.
	s, err := m.SettingsCheckedByGroups()
	if err != nil {
		return fmt.Errorf("scenario %q: %v", in.Scenario, err)
	}
	a.Model, a.Settings, a.Deployment = m, s, m.Deployment()
	return nil
}

// OpenModel opens the scenario's first pool for a caller that resolves the latency model
// without adopting the deployment first.
func OpenModel(scenario, scenariosDir string, data *load.Data) (*kernelmodel.Model, error) {
	m, err := kernelmodel.Open(scenario, kernelmodel.Repos{Scenarios: scenariosDir, Data: data})
	if err != nil {
		return nil, fmt.Errorf("scenario %q: %v", scenario, err)
	}
	return m, nil
}

// CheckParallelism refuses data and expert parallelism a dense model cannot run: dense DP is
// the router-replica mechanism, not a latency divisor (vLLM's --data-parallel-size on a dense
// model runs N independent engines behind an internal load balancer, which BLIS expresses as N
// instances), and vLLM refuses --enable-expert-parallel on a dense model
// (config/model.py:_verify_with_expert_parallelism). moe says the model graph routes experts.
func CheckParallelism(scenario, model string, dp int, ep, moe bool) error {
	if dp < 1 {
		return fmt.Errorf("scenario %q states dp %d; it must be >= 1", scenario, dp)
	}
	if dp > 1 && !moe {
		// The message is the CLI's, moved verbatim so its stderr is unchanged; it ends in a
		// sentence because it tells the operator what to state instead.
		//nolint:staticcheck // ST1005: a moved message, kept byte-identical.
		return fmt.Errorf("scenario %q states dp %d for a dense model; DP placement is modelled for MoE "+
			"models only. Express N independent dense engines as --num-instances N with dp 1.",
			scenario, dp)
	}
	if ep && !moe {
		return fmt.Errorf("scenario %q enables expert parallelism for dense model %q, which vLLM refuses",
			scenario, model)
	}
	return nil
}

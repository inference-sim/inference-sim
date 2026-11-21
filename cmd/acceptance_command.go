package cmd

import (
	"fmt"
	"slices"
	"strings"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/acceptance"
	specmodel "github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

// `blis acceptance` runs the lookup a run makes for its speculative acceptance (design D8,
// acceptance.Resolve) without the run, over the registry's profile sets, so a script learns
// whether a drafter's acceptance resolves, and through which step, before it spends a run on it
// (inference-sim#2141, design CFG-5). It states the registry layer alone: a workload value, a
// scenario pin or a command-line value would decide the figure without consulting a profile.

var (
	acceptModel      string
	acceptDrafter    string
	acceptMethod     string
	acceptK          int
	acceptCategory   string
	acceptThinking   string
	acceptHardware   string
	acceptEngine     string
	acceptExactOnly  bool
	acceptFallback   string
	acceptDraftSamp  string
	acceptRejectSamp string
	acceptLookupJSON bool
)

var acceptanceCmd = &cobra.Command{
	Use:   "acceptance --model <name> --method <m> --num-spec-tokens <K> --category <c> --thinking on|off",
	Short: "Look up the speculative acceptance the registry's profiles give a configuration",
	Long: "Run the acceptance lookup a drafting run makes (acceptance.Resolve, design D8) against the " +
		"registry in use, for the target --model (a catalog name or a source repository) and, with " +
		"--drafter, a separate drafter checkpoint named in the model's drafters (`blis catalog " +
		"resolve --model M --json` lists them); without it, the model's own in-graph speculator. " +
		"The output is the resolution: the acceptance length and rate, the layer (always registry), " +
		"the profile set and key, the step that found the value (exact, or the borrowing step), the " +
		"key fields in which a borrowed profile differs from the request, how K was brought to the " +
		"request's, and every profile read. A request nothing matches is refused with the " +
		"resolver's reason and a non-zero exit. --exact-only limits the lookup to the exact key. " +
		"--engine vllm states the engine and the draft sampler and verifier it defaults to, as a " +
		"run's engine does; without it those are unstated, and a profile that states them is " +
		"reached by the same_drafter_other_settings step, naming them.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		if err := checkAcceptanceLookupFlags(); err != nil {
			return err
		}
		data, err := runData()
		if err != nil {
			return err
		}
		id, err := resolveCatalogModel(data, acceptModel)
		if err != nil {
			return err
		}
		if id.Identity == nil {
			return fmt.Errorf("acceptance: --model %q is not in the catalog: %s", acceptModel, id.Reason)
		}
		g, err := data.Model(*id.Identity)
		if err != nil {
			return err
		}
		drafter := acceptDrafter
		if drafter == specmodel.InGraphSpeculator {
			drafter = ""
		}
		req := acceptance.Request{
			Model:             *id.Identity,
			Drafter:           drafter,
			SpeculativeMethod: acceptMethod,
			NumSpecTokens:     acceptK,
			Hardware:          acceptHardware,
			Architecture:      g.Global.Architecture,
			ExactOnly:         acceptExactOnly,
			Registry:          data.AcceptanceSets(),
			Workload:          acceptance.Workload{Category: acceptCategory, Thinking: acceptance.Thinking(acceptThinking)},
			FallbackCategory:  acceptFallback,
		}
		if acceptEngine == acceptanceEngineVLLM {
			req = vllmAcceptanceRequest(req, acceptDraftSamp, acceptRejectSamp)
		}
		res, err := acceptance.Resolve(req)
		if err != nil {
			return fmt.Errorf("acceptance: %v", err)
		}
		out := acceptanceLookupOf(req, res)
		if acceptLookupJSON {
			return writeJSON(cmd.OutOrStdout(), out)
		}
		_, err = fmt.Fprint(cmd.OutOrStdout(), out.text())
		return err
	},
}

func checkAcceptanceLookupFlags() error {
	for _, f := range []struct{ name, val string }{
		{"model", acceptModel}, {"method", acceptMethod}, {"category", acceptCategory}, {"thinking", acceptThinking},
	} {
		if f.val == "" {
			return fmt.Errorf("acceptance: --%s is required", f.name)
		}
	}
	if acceptK < 1 {
		return fmt.Errorf("acceptance: --num-spec-tokens is required and at least 1, got %d", acceptK)
	}
	if !slices.Contains(acceptance.AllThinkingModes(), acceptThinking) {
		return fmt.Errorf("acceptance: --thinking %q is not one of %s", acceptThinking,
			strings.Join(acceptance.AllThinkingModes(), ", "))
	}
	if acceptEngine != "" && acceptEngine != acceptanceEngineVLLM {
		return fmt.Errorf("acceptance: --engine %q is not one this build models (%s)", acceptEngine, acceptanceEngineVLLM)
	}
	if (acceptDraftSamp != "" || acceptRejectSamp != "") && acceptEngine == "" {
		return fmt.Errorf("acceptance: --draft-sample-method and --rejection-sample-method are the engine block's "+
			"settings; name the engine with --engine %s", acceptanceEngineVLLM)
	}
	return nil
}

// acceptanceLookup is the command's output: the request as resolved and the resolution.
type acceptanceLookup struct {
	Model            string    `json:"model"`
	Drafter          string    `json:"drafter"`
	Method           string    `json:"speculative_method"`
	NumSpecTokens    int       `json:"num_spec_tokens"`
	Category         string    `json:"category"`
	Thinking         string    `json:"thinking"`
	Hardware         string    `json:"hardware,omitempty"`
	Engine           string    `json:"engine,omitempty"`
	ExactOnly        bool      `json:"exact_only"`
	AcceptanceLength float64   `json:"acceptance_length"`
	AcceptanceRate   float64   `json:"acceptance_rate"`
	PerPositionRates []float64 `json:"per_position_rates"`
	RatesDerived     bool      `json:"rates_derived"`
	Layer            string    `json:"layer"`
	Set              string    `json:"set"`
	Profile          string    `json:"profile"`
	ProfileMethod    string    `json:"profile_method,omitempty"`
	Source           string    `json:"source,omitempty"`
	Step             string    `json:"step"`
	Differed         []string  `json:"differed"`
	KRule            string    `json:"k_rule,omitempty"`
	MeasuredOn       string    `json:"measured_on,omitempty"`
	FallbackFrom     string    `json:"fallback_from,omitempty"`
	Profiles         []string  `json:"profiles"`
}

// acceptanceLookupOf builds the output from the request and the resolution of a registry
// lookup, which always names the profile it read.
func acceptanceLookupOf(req acceptance.Request, r acceptance.Resolution) acceptanceLookup {
	o := acceptanceLookup{
		Model: req.Model, Drafter: req.Drafter, Method: req.SpeculativeMethod, NumSpecTokens: req.NumSpecTokens,
		Category: acceptCategory, Thinking: acceptThinking, Hardware: req.Hardware,
		Engine: req.Engine, ExactOnly: req.ExactOnly,
		AcceptanceLength: r.AcceptanceLength, AcceptanceRate: r.AcceptanceRate(),
		PerPositionRates: r.PerPositionRates, RatesDerived: r.RatesDerived,
		Layer: string(r.Layer), Set: r.Set, Source: r.Source, Step: string(r.Step),
		Differed: []string{}, KRule: string(r.KRule), MeasuredOn: r.MeasuredOn, FallbackFrom: r.FallbackFrom,
		Profiles: []string{},
	}
	if r.Borrowed() {
		o.Differed = append(o.Differed, r.Differed...)
	}
	if p := r.Profile; p != nil {
		o.Profile, o.ProfileMethod = p.Key().String(), string(p.Method)
	}
	for _, u := range r.Used {
		o.Profiles = append(o.Profiles, u.Set+" "+u.Profile.Key().String())
	}
	return o
}

func (o acceptanceLookup) text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "acceptance length %.4f (rate %.4f) at K=%d: %s step, profile %s %s\n",
		o.AcceptanceLength, o.AcceptanceRate, o.NumSpecTokens, o.Step, o.Set, o.Profile)
	if len(o.Differed) > 0 {
		fmt.Fprintf(&b, "differs in: %s\n", strings.Join(o.Differed, ", "))
	}
	if o.KRule != "" {
		fmt.Fprintf(&b, "K rule: %s\n", o.KRule)
	}
	if o.MeasuredOn != "" {
		fmt.Fprintf(&b, "measured on: %s\n", o.MeasuredOn)
	}
	if o.FallbackFrom != "" {
		fmt.Fprintf(&b, "fallback category replaced: %s\n", o.FallbackFrom)
	}
	fmt.Fprintf(&b, "profiles read: %s\n", strings.Join(o.Profiles, "; "))
	return b.String()
}

func init() {
	acceptanceCmd.Flags().StringVar(&acceptModel, "model", "", "the target model: a catalog entry name or a source repository (REQUIRED)")
	acceptanceCmd.Flags().StringVar(&acceptDrafter, "drafter", "", "a separate drafter checkpoint from the model's drafters; empty or \"speculator\" is the model's own in-graph speculator")
	acceptanceCmd.Flags().StringVar(&acceptMethod, "method", "", "the speculative method, as a deployment's engine.speculative.method states it, such as mtp or dspark (REQUIRED)")
	acceptanceCmd.Flags().IntVar(&acceptK, "num-spec-tokens", 0, "the drafted tokens per step K (REQUIRED, at least 1)")
	acceptanceCmd.Flags().StringVar(&acceptCategory, "category", "", "the workload category that selects the profile, such as coding (REQUIRED)")
	acceptanceCmd.Flags().StringVar(&acceptThinking, "thinking", "", "whether the target runs with thinking on or off (REQUIRED)")
	acceptanceCmd.Flags().StringVar(&acceptHardware, "hardware", "", "the hardware of the run, recorded as measured_on when the profile was measured elsewhere")
	acceptanceCmd.Flags().StringVar(&acceptEngine, "engine", "", "the serving engine, vllm: states the engine and its draft sampler and verifier defaults as a run does; without it they are unstated")
	acceptanceCmd.Flags().StringVar(&acceptDraftSamp, "draft-sample-method", "", "with --engine: engine.speculative.draft_sample_method (greedy or probabilistic); empty is the engine's default")
	acceptanceCmd.Flags().StringVar(&acceptRejectSamp, "rejection-sample-method", "", "with --engine: engine.speculative.rejection_sample_method (standard, block or synthetic); empty is the engine's default")
	acceptanceCmd.Flags().StringVar(&acceptFallback, "fallback-category", "", "a category to substitute, recorded, when the category has no profile")
	acceptanceCmd.Flags().BoolVar(&acceptExactOnly, "exact-only", false, "limit the lookup to the exact key: no borrowing")
	acceptanceCmd.Flags().BoolVar(&acceptLookupJSON, "json", false, "print JSON")
	registerDataFlags(acceptanceCmd)
	rootCmd.AddCommand(acceptanceCmd)
}

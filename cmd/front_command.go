package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/substrate/contracts/metrics"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/inference-sim/inference-sim/substrate/decision"
)

// `blis front` (design §15.5) takes the Result documents of the candidates of one decision case
// and the decision stage's artifact directory, and prints, for each point and each of the four
// fronts (XI, XT, IT, XIT), whether the point is predicted to be on the front and with what
// probability, and, for the pairs asked for, the probability that one point dominates the other.
// It reads; it writes no file and changes no Result (DEC-1).

var (
	frontArtifact string
	frontEpsilon  float64
	frontContext  string
	frontPairs    []string
	frontAllPairs bool
	frontFronts   []string
)

var frontCmd = &cobra.Command{
	Use:   "front [result-or-refusal-file ...]",
	Short: "Predict which candidates of a decision case are on each Pareto front",
	Long: "Read the Result documents of the candidates of one decision case (one model, workload and chip; " +
		"several configurations, each at several concurrencies) and the decision stage's artifact directory " +
		"(--artifact: artifact.json, model-<front>.json and fallback-<front>.json for XI, XT, IT, XIT; the artifact " +
		"shipped with this build when omitted). " +
		"For each point and front, print whether the point is on the predicted front (no other point dominates " +
		"it with probability above one half) and the probability that it is. --pair a:b and --all-pairs add the " +
		"probability that each point of a pair dominates the other. A point's objectives are read from its " +
		"Result: total_tokens_per_second_per_gpu, 1000 over itl_p90_ms, and ttft_p90_ms, at cluster scope. " +
		"A point the simulator did not simulate (a Result with no such figure, or a refusal document " +
		"{kind: Refusal, name, reason}) is estimated by the artifact's fallback model and listed with basis estimated, " +
		"the reason, and estimated: true on each front it was placed on without the simulator. A Result states its " +
		"configuration; a refusal document states it by carrying the engine, input and frame fields of a Result. " +
		"A point that states no configuration cannot be estimated: it is listed with basis not_simulated and the " +
		"reason, and is on no front. " +
		"--epsilon is the dominance tolerance of this command; an artifact fitted with another is refused (DEC-5). " +
		"--context names a JSON file that supplies what a Result does not hold: " +
		"{\"<result name or *>\": {\"m_tot\", \"m_act\", \"c_hbm\", \"c_bw\", \"c_fp8\", \"lisl\", \"losl\", \"src\"}}, " +
		"the case's model and chip properties as the stage's training data states them; an absent value is missing.",
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runFront(cmd.OutOrStdout(), frontOptions{
			Artifact: frontArtifact, Epsilon: frontEpsilon, Context: frontContext, Pairs: frontPairs,
			AllPairs: frontAllPairs, Fronts: frontFronts, Files: args,
		})
	},
}

func init() {
	f := frontCmd.Flags()
	f.StringVar(&frontArtifact, "artifact", "", "the decision stage's artifact directory (default: the artifact shipped with this build)")
	f.Float64Var(&frontEpsilon, "epsilon", decision.DefaultEpsilon, "dominance tolerance; the artifact must record the same")
	f.StringVar(&frontContext, "context", "", "JSON file of case properties a Result does not hold")
	f.StringArrayVar(&frontPairs, "pair", nil, "report the probabilities for the pair a:b of point names (repeatable)")
	f.BoolVar(&frontAllPairs, "all-pairs", false, "report the probabilities of every pair of simulated points")
	f.StringSliceVar(&frontFronts, "front", nil, "limit the output to these fronts of XI, XT, IT, XIT (default all four)")
	rootCmd.AddCommand(frontCmd)
}

type frontOptions struct {
	Artifact string
	Epsilon  float64
	Context  string
	Pairs    []string
	AllPairs bool
	Fronts   []string
	Files    []string
}

// The metrics a point's objectives are read from.
const (
	frontMetricX = "total_tokens_per_second_per_gpu"
	frontMetricI = "itl_p90_ms" // interactivity is 1000 over this
	frontMetricT = "ttft_p90_ms"
)

// frontReport is the output of blis front.
type frontReport struct {
	Kind          string            `json:"kind"`
	SchemaVersion int               `json:"schema_version"`
	Artifact      frontArtifactInfo `json:"artifact"`
	Epsilon       float64           `json:"epsilon"`
	Points        []frontPoint      `json:"points"`
	Pairs         []frontPairProb   `json:"pairs,omitempty"`
}

type frontArtifactInfo struct {
	Dir              string              `json:"dir"`
	TrainingDataHash string              `json:"training_data_hash"`
	RoleCounts       decision.RoleCounts `json:"role_counts"`
	WriterCommit     string              `json:"writer_commit"`
	Features         int                 `json:"features"`
}

// Basis says where a point's answers come from. A point the simulator simulated is "simulated". A
// point it did not simulate is "estimated" when the artifact has a fallback and the point states its
// configuration: the fallback places it and every answer for it is labelled estimated (DEC-3). Otherwise it is
// "not_simulated", listed with the reason, and on no front.
const (
	basisSimulated    = "simulated"
	basisEstimated    = "estimated"
	basisNotSimulated = "not_simulated"
)

type frontPoint struct {
	Name  string `json:"name"`
	Basis string `json:"basis"`
	// EstimatedBecause is why the simulator's figures are not used, for an estimated point.
	EstimatedBecause string `json:"estimated_because,omitempty"`
	Excluded         string `json:"excluded,omitempty"`
	// Predicted log objectives: throughput per GPU, interactivity, P90 TTFT in ms.
	LogX *float64 `json:"log_throughput_per_gpu,omitempty"`
	LogI *float64 `json:"log_interactivity,omitempty"`
	LogT *float64 `json:"log_p90_ttft_ms,omitempty"`
	// Fronts is keyed by front name.
	Fronts map[string]frontMembership `json:"fronts,omitempty"`
}

type frontMembership struct {
	// Answered is false when the simulator gave no value for an objective of the front.
	Answered bool `json:"answered"`
	// Estimated is true when the fallback placed the point on this front, because the simulator gave
	// no value for an objective of it (DEC-3).
	Estimated   bool    `json:"estimated,omitempty"`
	OnFront     bool    `json:"on_front"`
	Probability float64 `json:"probability"`
}

type frontPairProb struct {
	Front       string  `json:"front"`
	A           string  `json:"a"`
	B           string  `json:"b"`
	ADominatesB float64 `json:"p_a_dominates_b"`
	BDominatesA float64 `json:"p_b_dominates_a"`
	// Basis is the model that answered: "simulator" when the simulator answered both points on the
	// front, "fallback" otherwise, and then the pair's probabilities are estimates.
	Basis     string `json:"basis"`
	Estimated bool   `json:"estimated,omitempty"`
}

// refusal is the document of a point the simulator refused to simulate.
type refusal struct {
	Kind   string `yaml:"kind"`
	Name   string `yaml:"name"`
	Reason string `yaml:"reason"`
}

// caseContext holds case properties, as logs, for one point; a nil value is missing.
type caseContext struct {
	MTot *float64 `json:"m_tot"`
	MAct *float64 `json:"m_act"`
	CHBM *float64 `json:"c_hbm"`
	CBW  *float64 `json:"c_bw"`
	CFP8 *float64 `json:"c_fp8"`
	LISL *float64 `json:"lisl"`
	LOSL *float64 `json:"losl"`
	Src  *float64 `json:"src"`
}

func readContext(path string) (map[string]caseContext, error) {
	if path == "" {
		return nil, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]caseContext
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("--context %s: %w", path, err)
	}
	return m, nil
}

func (c caseContext) apply(p *decision.Point) {
	set := func(dst *float64, v *float64) {
		if v != nil {
			*dst = *v
		}
	}
	set(&p.MTot, c.MTot)
	set(&p.MAct, c.MAct)
	set(&p.CHBM, c.CHBM)
	set(&p.CBW, c.CBW)
	set(&p.CFP8, c.CFP8)
	set(&p.LISL, c.LISL)
	set(&p.LOSL, c.LOSL)
	set(&p.Src, c.Src)
}

func runFront(w io.Writer, o frontOptions) error {
	if len(o.Files) == 0 {
		return fmt.Errorf("front: give at least one Result file")
	}
	fronts := decision.Fronts
	if len(o.Fronts) > 0 {
		fronts = nil
		for _, n := range o.Fronts {
			f := decision.Front(strings.ToUpper(n))
			if !validFront(f) {
				return fmt.Errorf("front: --front %q is not one of XI, XT, IT, XIT", n)
			}
			fronts = append(fronts, f)
		}
	}
	var stage *decision.Stage
	var art *decision.Artifact
	var err error
	artifactName := "the shipped artifact"
	if o.Artifact == "" {
		stage, art, err = decision.LoadShipped(o.Epsilon)
	} else {
		stage, art, err = decision.LoadStage(o.Artifact, o.Epsilon)
		artifactName = filepath.Clean(o.Artifact)
	}
	if err != nil {
		return fmt.Errorf("front: %w", err)
	}
	ctx, err := readContext(o.Context)
	if err != nil {
		return fmt.Errorf("front: %w", err)
	}

	var points []decision.Point
	var report []frontPoint
	var placed []int // index into report of each point the stage places, parallel to points
	byName := map[string]int{}
	for _, file := range o.Files {
		c, err := readFrontPoint(file, ctx)
		if err != nil {
			return fmt.Errorf("front: %s: %w", file, err)
		}
		if _, dup := byName[c.Name]; dup {
			return fmt.Errorf("front: two documents name the point %q", c.Name)
		}
		byName[c.Name] = len(report)
		row := frontPoint{Name: c.Name, Basis: basisSimulated}
		switch {
		case c.Simulated:
		case stage.HasFallback() && c.Configured:
			row.Basis, row.EstimatedBecause = basisEstimated, c.Reason
		default:
			row.Basis, row.Excluded = basisNotSimulated, c.Reason
			if stage.HasFallback() {
				row.Excluded += "; it states no configuration, so the stage cannot estimate it"
			}
		}
		if row.Basis != basisNotSimulated {
			c.Point.ID = c.Name
			placed = append(placed, len(report))
			points = append(points, c.Point)
			finite := func(v float64) *float64 {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return nil
				}
				return &v
			}
			row.LogX, row.LogI, row.LogT = finite(c.Point.LogX), finite(c.Point.LogI), finite(c.Point.LogT)
			row.Fronts = map[string]frontMembership{}
		}
		report = append(report, row)
	}

	pairs, err := requestedPairs(o, byName, report)
	if err != nil {
		return fmt.Errorf("front: %w", err)
	}
	out := frontReport{Kind: "FrontReport", SchemaVersion: 1, Epsilon: stage.Epsilon, Points: report,
		Artifact: frontArtifactInfo{Dir: artifactName, TrainingDataHash: art.TrainingDataHash,
			RoleCounts: art.RoleCounts, WriterCommit: art.WriterCommit, Features: len(art.Features)}}
	local := map[string]int{} // point name -> index in points
	for k, ri := range placed {
		local[report[ri].Name] = k
	}
	for _, f := range fronts {
		mem, pi, err := stage.Front(f, points)
		if err != nil {
			return fmt.Errorf("front: %w", err)
		}
		for k, ri := range placed {
			if !mem[k].Answered && !mem[k].Estimated {
				report[ri].Fronts[string(f)] = frontMembership{} // placed on no front: the stage has no fallback for it
				continue
			}
			report[ri].Fronts[string(f)] = frontMembership{Answered: mem[k].Answered, Estimated: mem[k].Estimated, OnFront: mem[k].OnFront, Probability: mem[k].Probability}
		}
		for _, pr := range pairs {
			a, b := local[pr[0]], local[pr[1]]
			if !(mem[a].Answered || mem[a].Estimated) || !(mem[b].Answered || mem[b].Estimated) {
				continue
			}
			basis := stage.PairBasis(f, points[a], points[b])
			out.Pairs = append(out.Pairs, frontPairProb{Front: string(f), A: pr[0], B: pr[1], ADominatesB: pi[a][b], BDominatesA: pi[b][a],
				Basis: string(basis), Estimated: basis == decision.BasisFallback})
		}
	}
	out.Points = report
	return writeJSON(w, out)
}

func validFront(f decision.Front) bool {
	for _, g := range decision.Fronts {
		if f == g {
			return true
		}
	}
	return false
}

// requestedPairs resolves --pair and --all-pairs to pairs of the points the stage places: simulated or estimated.
func requestedPairs(o frontOptions, byName map[string]int, report []frontPoint) ([][2]string, error) {
	var out [][2]string
	for _, s := range o.Pairs {
		a, b, ok := strings.Cut(s, ":")
		if !ok || a == "" || b == "" || a == b {
			return nil, fmt.Errorf("--pair %q is not two different point names a:b", s)
		}
		for _, n := range []string{a, b} {
			i, known := byName[n]
			switch {
			case !known:
				return nil, fmt.Errorf("--pair %q: no point is named %q", s, n)
			case report[i].Basis == basisNotSimulated:
				return nil, fmt.Errorf("--pair %q: point %q was not simulated (%s)", s, n, report[i].Excluded)
			}
		}
		out = append(out, [2]string{a, b})
	}
	if o.AllPairs {
		var names []string
		for _, r := range report {
			if r.Basis != basisNotSimulated {
				names = append(names, r.Name)
			}
		}
		sort.Strings(names)
		for i := range names {
			for j := i + 1; j < len(names); j++ {
				out = append(out, [2]string{names[i], names[j]})
			}
		}
	}
	return out, nil
}

// frontCandidate is one document of the case: a point the simulator simulated, or one it did not.
type frontCandidate struct {
	Name  string
	Point decision.Point
	// Simulated: the document gives all three objectives. Otherwise Reason says why not.
	Simulated bool
	Reason    string
	// Configured: the document states the deployment, so the stage can estimate the point from its configuration.
	Configured bool
}

// refusalConfiguration is what a refusal document may carry beyond its name and reason: the
// engine, input and frame of the Result that the run would have produced, which state the
// candidate's configuration.
type refusalConfiguration struct {
	Engine result.Engine `yaml:"engine"`
	Input  result.Input  `yaml:"input"`
	Frame  result.Frame  `yaml:"frame"`
}

// readFrontPoint reads one candidate: a Result, or a refusal document.
func readFrontPoint(path string, ctx map[string]caseContext) (frontCandidate, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return frontCandidate{}, err
	}
	var head refusal
	if err := yaml.Unmarshal(raw, &head); err != nil {
		return frontCandidate{}, fmt.Errorf("not a Result or a refusal document: %w", err)
	}
	var c frontCandidate
	if head.Kind == "Refusal" {
		if head.Name == "" || head.Reason == "" {
			return c, fmt.Errorf("a refusal document states the point's name and the reason")
		}
		c.Name, c.Reason = head.Name, "the simulator refused it: "+head.Reason
		var cfg refusalConfiguration
		if err := yaml.Unmarshal(raw, &cfg); err != nil {
			return c, fmt.Errorf("the refusal document's configuration: %w", err)
		}
		r := result.Result{Engine: cfg.Engine, Input: cfg.Input, Frame: cfg.Frame}
		c.Point, _ = frontPointOf(&r)
		c.Configured = len(cfg.Input.Deployment.Pools) > 0
	} else {
		var r result.Result
		if err := result.Unmarshal(raw, &r); err != nil {
			return c, fmt.Errorf("not a Result this build reads: %w", err)
		}
		if r.Name == "" {
			return c, fmt.Errorf("the Result has no name")
		}
		c.Name = r.Name
		c.Point, c.Reason = frontPointOf(&r)
		c.Simulated = c.Reason == ""
		c.Configured = len(r.Input.Deployment.Pools) > 0
	}
	if c.Configured {
		for _, key := range []string{"*", c.Name} {
			if x, ok := ctx[key]; ok {
				x.apply(&c.Point)
			}
		}
	}
	return c, nil
}

func logOrNaN(x float64) float64 {
	if x > 0 && !math.IsInf(x, 0) {
		return math.Log(x)
	}
	return math.NaN()
}

// frontPointOf builds the decision stage's point from a Result: the configuration always, the
// objectives where the Result gives them. The reason is non-empty when the Result lacks an
// objective, which the point then holds as NaN: a run that failed an acceptance condition of its
// frame reports no figures (DEV-7).
func frontPointOf(r *result.Result) (decision.Point, string) {
	figure := func(id string) (float64, string) {
		f, ok := r.Metrics[id]
		switch {
		case !ok:
			return math.NaN(), fmt.Sprintf("the Result has no %s figure", id)
		case f.Scope != metrics.ScopeCluster:
			return math.NaN(), fmt.Sprintf("%s is at scope %q; the cluster figure is needed", id, f.Scope)
		case !(f.Value > 0) || math.IsInf(f.Value, 0):
			return math.NaN(), fmt.Sprintf("%s is %v; a positive figure is needed", id, f.Value)
		}
		return f.Value, ""
	}
	var reason string
	get := func(id string) float64 {
		v, why := figure(id)
		if why != "" && reason == "" {
			reason = why
		}
		return v
	}
	x, itl, ttft := get(frontMetricX), get(frontMetricI), get(frontMetricT)
	p := decision.Point{LogX: math.Log(x), LogI: math.Log(1000 / itl), LogT: math.Log(ttft)}
	nan := math.NaN()
	p.MTot, p.MAct, p.CHBM, p.CBW, p.CFP8, p.LISL, p.LOSL, p.Src = nan, nan, nan, nan, nan, nan, nan, 0
	p.Engine = 0
	if r.Engine.Kind == "sglang" {
		p.Engine = 1
	}
	if c := r.Frame.Generator.Load.Concurrency; c > 0 {
		p.LConc = math.Log(float64(c))
	} else {
		p.LConc = nan
	}
	configure(&p, r)
	return p, reason
}

// configure sets the configuration features from the Result's deployment. The layout kind is
// pd when the deployment has prefill and decode pools; else dpa for data-parallel attention
// (dp above 1 with expert parallelism), ep for expert parallelism, tp otherwise. For a pd
// layout the parallel degrees are the decode pool's, and the GPU counts are those of all the pools'
// workers (frontPoolGPUs), as the stage's training data states them.
func configure(p *decision.Point, r *result.Result) {
	nan := math.NaN()
	gpn := r.Input.Scenario.Cluster.GPUsPerNode
	if r.Input.Packaging != nil && r.Input.Packaging.GPUsPerNode > 0 {
		gpn = r.Input.Packaging.GPUsPerNode
	}
	pools := r.Input.Deployment.Pools
	var main *deployment.Pool
	prefill, total := 0, 0
	for i := range pools {
		pl := &pools[i]
		g := frontPoolGPUs(pl, gpn, r.Input.Options)
		total += g
		if pl.Role == deployment.RolePrefill {
			prefill += g
		}
		if main == nil || pl.Role == deployment.RoleDecode {
			main = pl
		}
	}
	p.TP, p.EP, p.DP, p.GPUs = nan, nan, nan, nan
	p.Seqs, p.Btok = nan, nan
	if main == nil {
		return
	}
	par := main.Parallel
	ep := 1
	if par.EnableExpertParallel {
		ep = par.TP * max(par.DP, 1)
	}
	p.TP, p.EP, p.DP = logOrNaN(float64(max(par.TP, 1))), logOrNaN(float64(ep)), logOrNaN(float64(max(par.DP, 1)))
	if total > 0 {
		p.GPUs = math.Log(float64(total))
	}
	switch {
	case hasRoles(pools, deployment.RolePrefill, deployment.RoleDecode):
		p.Kind, p.PD = decision.KindPD, 1
		if total > 0 {
			p.PDFrac = float64(prefill) / float64(total)
		}
	case par.EnableExpertParallel && par.DP > 1:
		p.Kind = decision.KindDPA
	case par.EnableExpertParallel:
		p.Kind = decision.KindEP
	default:
		p.Kind = decision.KindTP
	}
	if n := main.Engine.Node(); n != nil {
		if v, ok := engineInt(n, "max_num_seqs"); ok {
			p.Seqs = math.Log(float64(max(v, 1)))
		}
		if v, ok := engineInt(n, "max_num_batched_tokens"); ok {
			p.Btok = math.Log(float64(max(v, 1)))
		}
		if hasKey(n, "speculative_config") {
			p.Spec = 1
		}
	}
	for _, wp := range r.Input.WorkloadProperties {
		if wp.Name == result.SpeculativeAcceptance {
			p.Spec = 1
		}
	}
}

// instanceOption is the option that states how many instances (replicas) a pool runs.
var instanceOption = map[deployment.Role]string{
	deployment.RoleColocated: "num-instances",
	deployment.RolePrefill:   "prefill-instances",
	deployment.RoleDecode:    "decode-instances",
}

// frontPoolGPUs is the GPUs a pool occupies, as the stage's training data counts them: all the pool's
// workers, each tp x pp x dp. The instance count is a command-line option, not a pool field, so it
// is read from the Result's options; a pool whose count is not stated is counted by the nodes it
// holds. Two deployments that differ only in their instance count therefore differ in GPUs, as the
// training points do (the prefill and decode worker counts of a PD point are in its GPU count).
func frontPoolGPUs(pl *deployment.Pool, gpn int, opts []result.Option) int {
	name := instanceOption[pl.Role]
	for _, o := range opts {
		if o.Name != name {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(o.Value, "%d", &n); err == nil && n > 0 {
			par := pl.Parallel
			return n * sim.GPUsPerInstance(par.TP, par.PP, par.PCP, par.DP)
		}
	}
	return pl.Nodes * gpn
}

func hasRoles(pools []deployment.Pool, roles ...deployment.Role) bool {
	for _, want := range roles {
		found := false
		for _, pl := range pools {
			found = found || pl.Role == want
		}
		if !found {
			return false
		}
	}
	return true
}

func hasKey(n *yaml.Node, key string) bool {
	for i := 0; n.Kind == yaml.MappingNode && i+1 < len(n.Content); i += 2 {
		if n.Content[i].Value == key && n.Content[i+1].Tag != "!!null" {
			return true
		}
	}
	return false
}

func engineInt(n *yaml.Node, key string) (int, bool) {
	for i := 0; n.Kind == yaml.MappingNode && i+1 < len(n.Content); i += 2 {
		if n.Content[i].Value == key {
			var v int
			if n.Content[i+1].Decode(&v) == nil {
				return v, true
			}
		}
	}
	return 0, false
}

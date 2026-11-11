package routertable

// Statement is one difference between a router's mode and the llm-d policy that stands in for
// it, in words. A point record carries the statements of the entry it used.
type Statement struct {
	// Aspect names what differs, as a lower-case slug: for example "counter", "tie-break",
	// "state" or "scale".
	Aspect string `yaml:"aspect"`
	// Text says how the router's mode and the policy differ in that aspect, and what BLIS
	// does instead. It states a fact of the code, or says that an effect is not measured.
	Text string `yaml:"text"`
}

// Difference is a Statement with the citations that show it.
type Difference struct {
	Statement `yaml:",inline"`
	// Evidence lists the positions in the entry's evidence list of the citations the statement
	// rests on: at least one, each position once. A statement about a difference cites both
	// sides' code unless it says what BLIS does not model.
	Evidence []int `yaml:"evidence"`
}

// Statements returns the statements without their citation positions, which mean nothing
// outside the entry. A point record carries these.
func (e *Entry) Statements() []Statement {
	if len(e.Difference) == 0 {
		return nil
	}
	out := make([]Statement, len(e.Difference))
	for i, d := range e.Difference {
		out[i] = d.Statement
	}
	return out
}

// WorkloadKind names a family of workloads a bound was measured on. The list is closed: a
// bound is a statement about the workloads it names and no others, so a kind is added with the
// measurement that covers it.
type WorkloadKind string

const (
	// WorkloadBalanced: closed loop, every request 1,024 prompt tokens and 1,024 output
	// tokens, no shared prefix between requests.
	WorkloadBalanced WorkloadKind = "balanced"
	// WorkloadPrefillBound: closed loop, 8,192 prompt tokens and 16 output tokens.
	WorkloadPrefillBound WorkloadKind = "prefill-bound"
	// WorkloadDecodeBound: closed loop, 256 prompt tokens and 2,048 output tokens.
	WorkloadDecodeBound WorkloadKind = "decode-bound"
)

// WorkloadKinds lists the workload kinds a bound can name, in the order they are documented.
func WorkloadKinds() []WorkloadKind {
	return []WorkloadKind{WorkloadBalanced, WorkloadPrefillBound, WorkloadDecodeBound}
}

// Bound is a measured limit on how much the choice of router policy moves a figure.
//
// It is a sensitivity, not an error: it says how far BLIS's own prediction moves when the
// policy changes, so it says how much a wrong guess of the router's mode could cost. It does
// not say how far BLIS is from the router's real behaviour, which was not measured.
type Bound struct {
	// Evidence is the measurement the ratios come from.
	Evidence BoundEvidence `yaml:"evidence"`
	// Ratios are the limits, one per metric, regime and set of alternatives.
	Ratios []Ratio `yaml:"ratios"`
	// NotMeasured lists what the bound does not cover, each as a sentence: the regimes,
	// workloads and routers no run reached. At least one: a bound that claims to cover
	// everything is not a measured bound.
	NotMeasured []string `yaml:"not_measured"`
}

// BoundEvidence names the file and the producer of the measurement behind a bound.
type BoundEvidence struct {
	// File is the CSV of runs, from the repository's root. The ratios are recomputed from it.
	File string `yaml:"file"`
	// SHA256 is the lower-case hex SHA-256 of File, so a ratio cites the bytes it was read from.
	SHA256 string `yaml:"sha256"`
	// Producer is the repository BLIS was built from for the runs, and Commit the full commit.
	Producer string `yaml:"producer"`
	Commit   string `yaml:"commit"`
	// Command reproduces File from a checkout of the repository at the commit that holds it.
	Command string `yaml:"command"`
}

// Ratio is the largest ratio between the entry's Policy and a set of alternatives, for one
// figure, in one regime.
type Ratio struct {
	// Metric is the metric-registry identifier of the figure, for example "ttft_p50_ms".
	Metric string `yaml:"metric"`
	// Against names the alternatives the policy was compared with, each by the name of its
	// plugin composition in the evidence file's policy column. At least one, none equal to the
	// entry's own policy.
	Against []string `yaml:"against"`
	// Regime is where the ratio was measured.
	Regime Regime `yaml:"regime"`
	// MaxRatio is the largest, over every cell of the regime and every alternative, of the
	// larger of the two figures over the smaller, each figure taken as the mean over the seeds.
	// At least 1. It is rounded up to two decimals, so it never understates.
	MaxRatio float64 `yaml:"max_ratio"`
}

// Regime says where a ratio holds: the workload kind, the concurrency range, the model and the
// hardware, and the prefill and decode pool shapes. Outside it the ratio says nothing.
type Regime struct {
	Workload       WorkloadKind `yaml:"workload"`
	ConcurrencyMin int          `yaml:"concurrency_min"`
	ConcurrencyMax int          `yaml:"concurrency_max"`
	// Model and Hardware are catalog names.
	Model    string `yaml:"model"`
	Hardware string `yaml:"hardware"`
	// Shapes are the pool shapes the cells cover, each "<n>P<m>D": n prefill and m decode
	// instances.
	Shapes []string `yaml:"shapes"`
}

func (k WorkloadKind) valid() bool {
	for _, w := range WorkloadKinds() {
		if w == k {
			return true
		}
	}
	return false
}

package workload

import (
	"bytes"
	"fmt"
	"math"
	"os"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"
)

// v1ToV2SLOClasses maps deprecated v1 SLO class names to v2 equivalents.
var v1ToV2SLOClasses = map[string]string{
	"realtime":    "critical",
	"interactive": "standard",
}

// UpgradeV1ToV2 auto-upgrades a v1 WorkloadSpec to v2 format in-place.
// Maps deprecated SLO class names (realtime→critical, interactive→standard)
// and sets the version field to "2". Idempotent — calling on a v2 spec is safe.
// Emits logrus.Warn deprecation notices for mapped tier names.
func UpgradeV1ToV2(spec *WorkloadSpec) {
	if spec.Version == "" || spec.Version == "1" {
		spec.Version = "2"
	}
	for i := range spec.Clients {
		if newName, ok := v1ToV2SLOClasses[spec.Clients[i].SLOClass]; ok {
			logrus.Warnf("deprecated SLO class %q auto-mapped to %q; update your spec to use v2 tier names",
				spec.Clients[i].SLOClass, newName)
			spec.Clients[i].SLOClass = newName
		}
	}
}

// WorkloadSpec is the top-level workload configuration.
// Loaded from YAML via LoadWorkloadSpec(path).
type WorkloadSpec struct {
	Version       string       `yaml:"version"`
	Seed          int64        `yaml:"seed"`
	Category      string       `yaml:"category"`
	Clients       []ClientSpec `yaml:"clients"`
	AggregateRate float64      `yaml:"aggregate_rate"`
	Horizon       int64        `yaml:"horizon,omitempty"`
	NumRequests   int64        `yaml:"num_requests,omitempty"` // 0 = unlimited (use horizon only)
	ServeGenData  *ServeGenDataSpec  `yaml:"servegen_data,omitempty"`
	InferencePerf *InferencePerfSpec `yaml:"inference_perf,omitempty"`
}

// ClientSpec defines a single client's workload behavior.
type ClientSpec struct {
	ID           string        `yaml:"id"`
	TenantID     string        `yaml:"tenant_id"`
	SLOClass     string        `yaml:"slo_class"`
	Model        string        `yaml:"model,omitempty"`
	RateFraction float64       `yaml:"rate_fraction"`
	Arrival      ArrivalSpec   `yaml:"arrival"`
	InputDist    DistSpec      `yaml:"input_distribution"`
	OutputDist   DistSpec      `yaml:"output_distribution"`
	PrefixGroup  string        `yaml:"prefix_group,omitempty"`
	PrefixLength int           `yaml:"prefix_length,omitempty"` // shared prefix token count (default 50)
	Streaming    bool          `yaml:"streaming"`
	Network      *NetworkSpec  `yaml:"network,omitempty"`
	Lifecycle    *LifecycleSpec `yaml:"lifecycle,omitempty"`
	Multimodal   *MultimodalSpec `yaml:"multimodal,omitempty"`
	Reasoning    *ReasoningSpec  `yaml:"reasoning,omitempty"`
}

// ArrivalSpec configures the inter-arrival time process.
type ArrivalSpec struct {
	Process string   `yaml:"process"`
	CV      *float64 `yaml:"cv,omitempty"`
}

// DistSpec parameterizes a token length distribution.
type DistSpec struct {
	Type   string             `yaml:"type"`
	Params map[string]float64 `yaml:"params,omitempty"`
	File   string             `yaml:"file,omitempty"`
}

// NetworkSpec defines client-side network characteristics.
type NetworkSpec struct {
	RTTMs         float64 `yaml:"rtt_ms"`
	BandwidthMbps float64 `yaml:"bandwidth_mbps,omitempty"`
}

// LifecycleSpec defines client activity windows.
type LifecycleSpec struct {
	Windows []ActiveWindow `yaml:"windows"`
}

// ActiveWindow represents a period when a client is active.
type ActiveWindow struct {
	StartUs int64 `yaml:"start_us"`
	EndUs   int64 `yaml:"end_us"`
}

// MultimodalSpec configures multimodal request generation.
type MultimodalSpec struct {
	TextDist       DistSpec `yaml:"text_distribution"`
	ImageDist      DistSpec `yaml:"image_distribution"`
	ImageCountDist DistSpec `yaml:"image_count_distribution"`
	AudioDist      DistSpec `yaml:"audio_distribution"`
	AudioCountDist DistSpec `yaml:"audio_count_distribution"`
	VideoDist      DistSpec `yaml:"video_distribution"`
	VideoCountDist DistSpec `yaml:"video_count_distribution"`
}

// ReasoningSpec configures reasoning model workload generation.
type ReasoningSpec struct {
	ReasonRatioDist DistSpec       `yaml:"reason_ratio_distribution"`
	MultiTurn       *MultiTurnSpec `yaml:"multi_turn,omitempty"`
}

// MultiTurnSpec configures multi-turn conversation behavior.
type MultiTurnSpec struct {
	MaxRounds     int    `yaml:"max_rounds"`
	ThinkTimeUs   int64  `yaml:"think_time_us"`
	ContextGrowth string `yaml:"context_growth"`
}

// ServeGenDataSpec configures native ServeGen data file loading.
type ServeGenDataSpec struct {
	Path      string `yaml:"path"`
	SpanStart int64  `yaml:"span_start,omitempty"`
	SpanEnd   int64  `yaml:"span_end,omitempty"`
}

// Valid value registries.
var (
	validArrivalProcesses = map[string]bool{
		"poisson": true, "gamma": true, "weibull": true, "constant": true,
	}
	validDistTypes = map[string]bool{
		"gaussian": true, "exponential": true, "pareto_lognormal": true, "empirical": true, "constant": true,
	}
	validCategories = map[string]bool{
		"": true, "language": true, "multimodal": true, "reasoning": true,
	}
	validSLOClasses = map[string]bool{
		"": true, "critical": true, "standard": true, "sheddable": true, "batch": true, "background": true,
	}
)

// LoadWorkloadSpec reads and parses a YAML workload specification file.
// Uses strict parsing: unrecognized keys (typos) are rejected.
func LoadWorkloadSpec(path string) (*WorkloadSpec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading workload spec: %w", err)
	}
	var spec WorkloadSpec
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&spec); err != nil {
		return nil, fmt.Errorf("parsing workload spec: %w", err)
	}
	UpgradeV1ToV2(&spec)
	return &spec, nil
}

// Validate checks that all fields in the spec are valid.
func (s *WorkloadSpec) Validate() error {
	if !validCategories[s.Category] {
		return fmt.Errorf("unknown category %q; valid: language, multimodal, reasoning", s.Category)
	}
	if s.AggregateRate <= 0 {
		return fmt.Errorf("aggregate_rate must be positive, got %f", s.AggregateRate)
	}
	if len(s.Clients) == 0 && s.ServeGenData == nil {
		return fmt.Errorf("at least one client or servegen_data path required")
	}
	for i, c := range s.Clients {
		if err := validateClient(&c, i); err != nil {
			return err
		}
	}
	return nil
}

func validateClient(c *ClientSpec, idx int) error {
	prefix := fmt.Sprintf("client[%d]", idx)
	if !validSLOClasses[c.SLOClass] {
		return fmt.Errorf("%s: unknown slo_class %q; valid: critical, standard, sheddable, batch, background, or empty", prefix, c.SLOClass)
	}
	if c.RateFraction <= 0 {
		return fmt.Errorf("%s: rate_fraction must be positive, got %f", prefix, c.RateFraction)
	}
	if !validArrivalProcesses[c.Arrival.Process] {
		return fmt.Errorf("%s: unknown arrival process %q; valid: poisson, gamma, weibull", prefix, c.Arrival.Process)
	}
	if c.Arrival.Process == "weibull" && c.Arrival.CV != nil {
		cv := *c.Arrival.CV
		if cv < 0.01 || cv > 10.4 {
			return fmt.Errorf("%s: weibull CV must be in [0.01, 10.4], got %f", prefix, cv)
		}
	}
	if c.Arrival.CV != nil {
		if err := validateFinitePositive(prefix+".cv", *c.Arrival.CV); err != nil {
			return err
		}
	}
	if c.PrefixLength < 0 {
		return fmt.Errorf("%s: prefix_length must be non-negative, got %d", prefix, c.PrefixLength)
	}
	if err := validateDistSpec(prefix+".input_distribution", &c.InputDist); err != nil {
		return err
	}
	if err := validateDistSpec(prefix+".output_distribution", &c.OutputDist); err != nil {
		return err
	}
	return nil
}

func validateDistSpec(prefix string, d *DistSpec) error {
	if !validDistTypes[d.Type] {
		return fmt.Errorf("%s: unknown distribution type %q; valid: gaussian, exponential, pareto_lognormal, empirical, constant", prefix, d.Type)
	}
	for name, val := range d.Params {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return fmt.Errorf("%s.params.%s must be a finite number, got %f", prefix, name, val)
		}
	}
	return nil
}

// IsValidSLOClass reports whether name is a valid v2 SLO class.
// Valid classes: "", "critical", "standard", "sheddable", "batch", "background".
func IsValidSLOClass(name string) bool {
	return validSLOClasses[name]
}

func validateFinitePositive(name string, val float64) error {
	if math.IsNaN(val) || math.IsInf(val, 0) {
		return fmt.Errorf("%s must be a finite number, got %f", name, val)
	}
	if val <= 0 {
		return fmt.Errorf("%s must be positive, got %f", name, val)
	}
	return nil
}

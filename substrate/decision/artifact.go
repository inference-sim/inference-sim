package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
)

// ArtifactKind and ArtifactVersion identify the artifact.json document.
const (
	ArtifactKind    = "DecisionArtifact"
	ArtifactVersion = 1
)

// DefaultEpsilon is the dominance tolerance the dashboard and blis front use unless told another.
// An artifact fitted with a different one is refused (DEC-5).
const DefaultEpsilon = 0.05

// Artifact is artifact.json: what a fit of the stage records beside its models (DEC-5).
type Artifact struct {
	Kind          string `json:"kind"`
	SchemaVersion int    `json:"schema_version"`
	// Epsilon is the dominance tolerance of the training labels.
	Epsilon float64 `json:"epsilon"`
	// TrainingDataHash is the content hash of the training data, "sha256:" and 64 hex digits.
	TrainingDataHash string `json:"training_data_hash"`
	// RoleCounts counts the training units by role (design §10.2): fit, select, validate.
	RoleCounts RoleCounts `json:"role_counts"`
	// WriterCommit is the commit (40 hex digits) of the writer script that produced the models.
	WriterCommit string `json:"writer_commit"`
	// Features are the model inputs, in order. Every name is one of StageFeatureNames.
	Features []string `json:"features"`
	// Chosen records the forms and settings the select units chose; free-form.
	Chosen map[string]any `json:"chosen,omitempty"`
}

// RoleCounts counts the training units by role.
type RoleCounts struct {
	Fit      int `json:"fit"`
	Select   int `json:"select"`
	Validate int `json:"validate"`
}

var (
	hashRE   = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// ModelFile is the name of the model of front f in an artifact directory.
func ModelFile(f Front) string { return "model-" + string(f) + ".json" }

// Validate checks the document's fields. It does not read the models.
func (a *Artifact) Validate() error {
	var bad []string
	if a.Kind != ArtifactKind {
		bad = append(bad, fmt.Sprintf("kind is %q, want %q", a.Kind, ArtifactKind))
	}
	if a.SchemaVersion != ArtifactVersion {
		bad = append(bad, fmt.Sprintf("schema_version is %d, want %d", a.SchemaVersion, ArtifactVersion))
	}
	if !(a.Epsilon > 0 && a.Epsilon < 1) || math.IsNaN(a.Epsilon) {
		bad = append(bad, fmt.Sprintf("epsilon %v is not in (0, 1)", a.Epsilon))
	}
	if !hashRE.MatchString(a.TrainingDataHash) {
		bad = append(bad, fmt.Sprintf("training_data_hash %q is not sha256: and 64 lower-case hex digits", a.TrainingDataHash))
	}
	if !commitRE.MatchString(a.WriterCommit) {
		bad = append(bad, fmt.Sprintf("writer_commit %q is not 40 lower-case hex digits", a.WriterCommit))
	}
	rc := a.RoleCounts
	if rc.Fit < 1 || rc.Select < 0 || rc.Validate < 0 {
		bad = append(bad, fmt.Sprintf("role_counts %+v: fit must be at least 1 and no count negative", rc))
	}
	known := map[string]bool{}
	for _, n := range StageFeatureNames {
		known[n] = true
	}
	seen := map[string]bool{}
	if len(a.Features) == 0 {
		bad = append(bad, "features is empty")
	}
	for _, n := range a.Features {
		switch {
		case !known[n]:
			bad = append(bad, fmt.Sprintf("feature %q is not one the stage knows", n))
		case seen[n]:
			bad = append(bad, fmt.Sprintf("feature %q is listed twice", n))
		}
		seen[n] = true
	}
	for _, n := range []string{"dX", "dI", "dT"} {
		if !seen[n] {
			bad = append(bad, fmt.Sprintf("feature %q, a simulator difference the stage is monotone in, is missing", n))
		}
	}
	if len(bad) > 0 {
		sort.Strings(bad)
		return fmt.Errorf("artifact.json: %s", joinProblems(bad))
	}
	return nil
}

func joinProblems(p []string) string {
	var b bytes.Buffer
	for i, s := range p {
		if i > 0 {
			b.WriteString("; ")
		}
		b.WriteString(s)
	}
	return b.String()
}

// LoadStage reads an artifact directory: artifact.json and model-XI.json, model-XT.json,
// model-IT.json and model-XIT.json. It refuses an artifact whose epsilon differs from epsilon
// (DEC-5: the stage, the dashboard and blis front use one tolerance), whose fields are invalid,
// or whose model takes a number of inputs other than the feature list's length.
func LoadStage(dir string, epsilon float64) (*Stage, *Artifact, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "artifact.json"))
	if err != nil {
		return nil, nil, fmt.Errorf("artifact directory %s: %w", dir, err)
	}
	var a Artifact
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&a); err != nil {
		return nil, nil, fmt.Errorf("%s: artifact.json: %w", dir, err)
	}
	if err := a.Validate(); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", dir, err)
	}
	if a.Epsilon != epsilon {
		return nil, nil, fmt.Errorf("%s: the artifact was fitted with epsilon %v and this command uses %v; "+
			"the stage, the dashboard and blis front must use the artifact's one epsilon (DEC-5)", dir, a.Epsilon, epsilon)
	}
	s := &Stage{Epsilon: a.Epsilon, Features: a.Features, Models: map[Front]*Model{}}
	for _, f := range Fronts {
		m, err := LoadModel(filepath.Join(dir, ModelFile(f)))
		if err != nil {
			return nil, nil, err
		}
		if m.NumFeatures != len(a.Features) {
			return nil, nil, fmt.Errorf("%s: model of front %s takes %d inputs and artifact.json lists %d features",
				dir, f, m.NumFeatures, len(a.Features))
		}
		s.Models[f] = m
	}
	return s, &a, nil
}

package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"regexp"
	"sort"
)

// ArtifactKind and the schema versions identify the artifact.json document. Version 1 holds one
// model per front, the simulator model. Version 2 adds a fallback model per front, for the pairs
// the simulator does not answer (DEC-3); it is the version an artifact is written in.
const (
	ArtifactKind            = "DecisionArtifact"
	ArtifactVersion         = 1
	ArtifactVersionFallback = 2
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
	// FallbackMasked names the features the fallback models never read: the simulator's
	// differences. They were missing in every training row of the fallback, and the stage passes
	// them as missing. Version 2 requires it; version 1 has no fallback and must not have it.
	FallbackMasked []string `json:"fallback_masked_features,omitempty"`
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

// ModelFile is the name of the simulator model of front f in an artifact directory.
func ModelFile(f Front) string { return "model-" + string(f) + ".json" }

// FallbackFile is the name of the fallback model of front f (schema version 2).
func FallbackFile(f Front) string { return "fallback-" + string(f) + ".json" }

// Validate checks the document's fields. It does not read the models.
func (a *Artifact) Validate() error {
	var bad []string
	if a.Kind != ArtifactKind {
		bad = append(bad, fmt.Sprintf("kind is %q, want %q", a.Kind, ArtifactKind))
	}
	if a.SchemaVersion != ArtifactVersion && a.SchemaVersion != ArtifactVersionFallback {
		bad = append(bad, fmt.Sprintf("schema_version is %d, want %d or %d", a.SchemaVersion, ArtifactVersion, ArtifactVersionFallback))
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
	switch {
	case a.SchemaVersion == ArtifactVersion && len(a.FallbackMasked) > 0:
		bad = append(bad, "fallback_masked_features is set, and schema version 1 has no fallback models")
	case a.SchemaVersion == ArtifactVersionFallback && len(a.FallbackMasked) == 0:
		bad = append(bad, "fallback_masked_features is empty: a version 2 artifact names the simulator differences its fallback models never read")
	}
	for _, n := range a.FallbackMasked {
		if !seen[n] {
			bad = append(bad, fmt.Sprintf("fallback_masked_features names %q, which is not a feature", n))
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
// model-IT.json and model-XIT.json, and for schema version 2 also fallback-XI.json, fallback-XT.json,
// fallback-IT.json and fallback-XIT.json. It refuses an artifact whose epsilon differs from epsilon
// (DEC-5: the stage, the dashboard and blis front use one tolerance), whose fields are invalid,
// or whose model takes a number of inputs other than the feature list's length.
func LoadStage(dir string, epsilon float64) (*Stage, *Artifact, error) {
	return loadStageFS(os.DirFS(dir), dir, epsilon)
}

func loadStageFS(fsys fs.FS, label string, epsilon float64) (*Stage, *Artifact, error) {
	raw, err := fs.ReadFile(fsys, "artifact.json")
	if err != nil {
		return nil, nil, fmt.Errorf("artifact directory %s: %w", label, err)
	}
	var a Artifact
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&a); err != nil {
		return nil, nil, fmt.Errorf("%s: artifact.json: %w", label, err)
	}
	if err := a.Validate(); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", label, err)
	}
	if a.Epsilon != epsilon {
		return nil, nil, fmt.Errorf("%s: the artifact was fitted with epsilon %v and this command uses %v; "+
			"the stage, the dashboard and blis front must use the artifact's one epsilon (DEC-5)", label, a.Epsilon, epsilon)
	}
	s := &Stage{Epsilon: a.Epsilon, Features: a.Features, Models: map[Front]*Model{}}
	read := func(name string, f Front) (*Model, error) {
		b, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", label, err)
		}
		m, err := ParseModel(b)
		if err != nil {
			return nil, fmt.Errorf("%s: %s: %w", label, name, err)
		}
		if m.NumFeatures != len(a.Features) {
			return nil, fmt.Errorf("%s: model of front %s (%s) takes %d inputs and artifact.json lists %d features",
				label, f, name, m.NumFeatures, len(a.Features))
		}
		return m, nil
	}
	for _, f := range Fronts {
		if s.Models[f], err = read(ModelFile(f), f); err != nil {
			return nil, nil, err
		}
	}
	if a.SchemaVersion == ArtifactVersionFallback {
		s.Fallback = map[Front]*Model{}
		for _, f := range Fronts {
			if s.Fallback[f], err = read(FallbackFile(f), f); err != nil {
				return nil, nil, err
			}
		}
		s.setMasked(a.FallbackMasked)
	}
	return s, &a, nil
}

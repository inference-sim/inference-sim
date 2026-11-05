package model

import "github.com/inference-sim/blis-substrate/contracts/check"

// Identity is a catalog model entry's identity manifest: the file blis-catalog commits
// as models/<name>/model.yaml, beside the verbatim vendor config.json and the derived
// graph.yaml. It is a different artifact from the cost Graph this package otherwise
// describes — it prices nothing — so it carries no shape, only the name the entry claims
// and the provenance of the vendor file it was built from.
//
// Unlike a chip, a fabric or a workload, this file states its name IN the document rather
// than taking it from the filename: the catalog requires that stated name to match the
// directory it sits in, and the validate-catalog command checks that, as it does for a
// graph. Validate here checks only what the document can know about itself.
//
// The checks mirror the MODEL.YAML portion of blis-catalog's own validate_models: name must
// be a non-empty string (its match against the directory is the caller's check), and source
// must be present with a non-empty provider, repo and revision. validate_models applies no
// further restriction to those fields — no pattern on provider (any non-empty string,
// "huggingface" in practice) and no date-format check on retrieved — so retrieved is declared
// here only to name the key every real card carries, and is neither required nor
// format-checked. The sibling config.json is the OTHER half of validate_models' per-entry
// check, and the validate-catalog command covers it too: it asserts config.json
// structurally (present, parseable, a non-empty JSON object) without a schema type, since it
// is a verbatim vendor file whose keys the derived graph — not this schema — interprets.
// Between that structural check and this identity, the command reproduces validate_models'
// model-entry checks in full, so it can stand in for them when blis-catalog wires it into CI
// (#22, catalog#14).
//
// Field values aside, the loader (LoadModelIdentity) decodes strict: an unknown key is
// rejected, where the catalog's Python gate reads model.yaml leniently and ignores extra
// keys. That is intentional and NOT a claim of leniency-parity — an unknown key is a probable
// typo that would otherwise silently drop the setting its author meant, and rejecting it is
// the same filename-identity direction #27/#25 take. One consequence: unlike Chip and Fabric,
// Identity does not strip the catalog's underscore-prefixed _comment* keys (the same gap #28
// tracks for workload Shape), so such a key would be rejected rather than ignored; no
// committed model.yaml carries one today.
type Identity struct {
	Name   string  `yaml:"name"`
	Source *Source `yaml:"source"`
}

// Source records where a model entry's vendor config was retrieved from, so a derived
// graph can be audited against a pinned upstream revision rather than a moving branch.
type Source struct {
	// Provider names the hosting system the repo lives on (huggingface, in practice).
	Provider string `yaml:"provider"`
	// Repo is the provider-scoped repository path, e.g. "deepseek-ai/DeepSeek-V4-Pro".
	Repo string `yaml:"repo"`
	// Revision pins the exact commit the config was taken at. A branch name would let the
	// upstream move under a fixed graph, which is the drift the digest in the graph's
	// derived_from exists to catch; this pins the other end of that comparison.
	Revision string `yaml:"revision"`
	// Retrieved is the date the config was fetched, as provenance. Optional: it is not
	// part of the identity the catalog's gate enforces, so it is recorded where present
	// and never required.
	Retrieved string `yaml:"retrieved,omitempty"`
}

// Validate performs field-level validation of a model identity manifest: a name must be
// present (the directory-name match is checked by the caller that knows the directory),
// and the source provenance must name a provider, a repo and a revision.
func (id *Identity) Validate() *validate.Problems {
	p := &validate.Problems{}
	if id.Name == "" {
		p.Field("name", "required")
	}
	if id.Source == nil {
		p.Field("source", "required: a model entry records where its config was retrieved from")
		return p
	}
	p.Merge("source", id.Source.Validate())
	return p
}

// Validate performs field-level validation of a source provenance block.
func (s *Source) Validate() *validate.Problems {
	p := &validate.Problems{}
	if s.Provider == "" {
		p.Field("provider", "required")
	}
	if s.Repo == "" {
		p.Field("repo", "required")
	}
	if s.Revision == "" {
		p.Field("revision", "required")
	}
	return p
}

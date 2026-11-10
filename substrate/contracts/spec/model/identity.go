package model

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

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
// the same filename-identity direction the other loaders take. One consequence: unlike Chip and
// Fabric, Identity does not strip the catalog's underscore-prefixed _comment* keys (workload
// Shape has the same gap), so such a key would be rejected rather than ignored; no
// committed model.yaml carries one today.
type Identity struct {
	Name   string  `yaml:"name"`
	Source *Source `yaml:"source"`
	// Licence is the licence the model's vendor config.json is published under (X12). The
	// loader decodes it and does not require it; ValidateLicence, run in CI, does.
	Licence *Licence `yaml:"licence"`
	// Config pins the vendor config.json by hash, for a model whose licence is not on the
	// allow-list: such a model commits only this hash and the source revision, and the
	// deriver fetches the file in a nightly job that holds the access token (X12).
	Config *ConfigPin `yaml:"config,omitempty"`
	// Drafters lists what can draft tokens for this target (design D8): the word
	// "speculator" for the in-graph speculator its graph.yaml states (its MTP layers, say), and
	// the catalog name of each separate drafter checkpoint. The catalog check requires every
	// graph with a speculator to list it, and every named drafter to name this model as its
	// Target.
	Drafters []string `yaml:"drafters,omitempty"`
	// Targets is set on a separate drafter's entry: the catalog names of the models it drafts
	// for. Each of them lists this entry in its Drafters. A drafter checkpoint drafts for every
	// model whose hidden states it was trained to read, and one architecture ships in several
	// quantisations that are separate catalog entries, so a drafter may have several targets;
	// acceptance is keyed by the (target, drafter) pair (design D8), so each pair is its own
	// key.
	Targets []string `yaml:"targets,omitempty"`
}

// InGraphSpeculator is the word a target's Drafters list uses for the speculator its own
// graph states.
const InGraphSpeculator = "speculator"

// Licence records a model's licence and where it was read.
type Licence struct {
	// ID is the licence's SPDX identifier, or LicenseRef-<name> for a licence SPDX does not
	// list, written as the model card at the pinned revision names it. When the card names
	// no licence, or names only "other", the ID names the licence file read at the pinned
	// revision instead (LicenseRef-deepseek for DeepSeek-V3's LICENSE-MODEL), never a
	// placeholder such as LicenseRef-other or LicenseRef-none-stated, which names no text.
	ID string `yaml:"id"`
	// Evidence is the URL the licence was read from, at the pinned revision: the model
	// card's front matter, the provider's API record of that revision, or, when the ID
	// names a licence file because the card does not, that file.
	Evidence string `yaml:"evidence"`
}

// ConfigPin is the SHA-256 of a vendor config.json that is not committed.
type ConfigPin struct {
	SHA256 string `yaml:"sha256"`
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

// Validate performs the structural validation of a model identity manifest, the part the
// loader applies on every load: a name must be present (the directory-name match is checked
// by the caller that knows the directory), the source provenance must name a provider, a repo
// and a revision, and a config pin must be a SHA-256.
//
// It does not look at the licence block. No simulation path reads, requires or enforces a
// licence (X12): the block is decoded so that CI can read it, and ValidateLicence is the CI
// check on it.
func (id *Identity) Validate() *validate.Problems {
	p := &validate.Problems{}
	id.validateStructure(p)
	return p
}

func (id *Identity) validateStructure(p *validate.Problems) {
	if id.Name == "" {
		p.Field("name", "required")
	}
	if id.Source == nil {
		p.Field("source", "required: a model entry records where its config was retrieved from")
		return
	}
	p.Merge("source", id.Source.Validate())
	if id.Config != nil && !validate.IsSHA256(id.Config.SHA256) {
		p.Field("config.sha256", "%q is not a lowercase hex SHA-256", id.Config.SHA256)
	}
	seen := map[string]bool{}
	for i, d := range id.Drafters {
		at := fmt.Sprintf("drafters[%d]", i)
		switch {
		case d == "":
			p.Field(at, "is empty: name %q or a catalog model", InGraphSpeculator)
		case d == id.Name:
			p.Field(at, "%q is this model; a model does not draft for itself", d)
		case seen[d]:
			p.Field(at, "%q is listed twice", d)
		}
		seen[d] = true
	}
	seenTarget := map[string]bool{}
	for i, t := range id.Targets {
		at := fmt.Sprintf("targets[%d]", i)
		switch {
		case t == "":
			p.Field(at, "is empty: name the catalog model this drafter drafts for")
		case t == id.Name:
			p.Field(at, "%q is this model; a drafter's target is another model", t)
		case seenTarget[t]:
			p.Field(at, "%q is listed twice", t)
		}
		seenTarget[t] = true
	}
	if len(id.Targets) > 0 && len(id.Drafters) > 0 {
		p.Field("drafters", "a separate drafter names its targets and lists no drafters of its own")
	}
}

// ValidateLicence is the CI check on the licence block alone (X12): the project's own catalog
// records, for each model, the licence its config.json is published under, and records it
// well. It is run by validate-catalog and by the substrate's own tests, never by the loader,
// so a catalog a user supplies is not refused for lacking a licence record.
//
// A model with no source is reported by Validate; it has no revision to read the licence at,
// so this check has nothing to add.
func (id *Identity) ValidateLicence() *validate.Problems {
	p := &validate.Problems{}
	id.validateLicence(p)
	return p
}

// ValidateWithLicence is Validate followed by ValidateLicence, reported together, as
// validate-catalog reports a model.yaml.
func (id *Identity) ValidateWithLicence() *validate.Problems {
	p := &validate.Problems{}
	id.validateStructure(p)
	id.validateLicence(p)
	return p
}

func (id *Identity) validateLicence(p *validate.Problems) {
	if id.Source == nil {
		return
	}
	if id.Licence == nil {
		p.Field("licence", "required: a model entry records its config.json's licence (X12)")
		return
	}
	p.Merge("licence", id.Licence.Validate(id.Source))
}

var licenceRef = regexp.MustCompile(`^(LicenseRef-[A-Za-z0-9.-]+|[A-Za-z0-9][A-Za-z0-9.+-]*)$`)

// placeholderLicences are LicenseRef names that record what a model card says about its
// licence ("other", nothing) rather than naming a licence text.
var placeholderLicences = map[string]bool{
	"LicenseRef-other":       true,
	"LicenseRef-none":        true,
	"LicenseRef-none-stated": true,
	"LicenseRef-unknown":     true,
}

// Validate checks a licence record (a CI check; see Identity.ValidateLicence): an identifier in SPDX form, and evidence at the
// source's pinned revision.
func (l *Licence) Validate(src *Source) *validate.Problems {
	p := &validate.Problems{}
	switch {
	case !licenceRef.MatchString(l.ID):
		p.Field("id", "%q is not an SPDX identifier or LicenseRef-<name>", l.ID)
	case placeholderLicences[l.ID]:
		p.Field("id", "%q names no licence text; name the licence file read at the pinned revision, and cite that file as the evidence", l.ID)
	}
	if !strings.HasPrefix(l.Evidence, "https://") {
		p.Field("evidence", "%q is not an https URL", l.Evidence)
	} else if src != nil && src.Revision != "" && !strings.Contains(l.Evidence, src.Revision) {
		p.Field("evidence", "the URL does not name the pinned revision %s; read the licence at that revision", src.Revision)
	}
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

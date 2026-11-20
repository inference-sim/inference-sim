package measurement

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Registry is the catalog's licence registry, measurements/sources.yaml: one entry per
// source a run may cite, with the evidence for what its terms permit.
type Registry struct {
	Kind    string    `yaml:"kind"` // "MeasurementSources"
	Sources []License `yaml:"sources"`
}

// License is one source's terms. Only a verified entry permits anything: an unverified or
// refused source cannot be scored, fitted to, or committed.
type License struct {
	ID    string `yaml:"id"`
	Title string `yaml:"title"`
	// License is the SPDX identifier the evidence states, empty when none was found.
	License  string        `yaml:"license,omitempty"`
	Status   LicenseStatus `yaml:"status"`
	Permits  []Use         `yaml:"permits,omitempty"`
	Evidence []Evidence    `yaml:"evidence,omitempty"`
	// Attribution is what a use must carry, when the terms require it.
	Attribution string `yaml:"attribution,omitempty"`
	Note        string `yaml:"note,omitempty"`
}

// Evidence is one quoted statement of terms and where it is.
type Evidence struct {
	URL      string `yaml:"url"`
	Quote    string `yaml:"quote"`
	Location string `yaml:"location"`
}

var slug = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)

// Validate is the CI check on the registry (X12, D6): the structural checks of ValidateStructure
// and every rule about the terms themselves, namely what a verified source must state, the
// allow-list for redistribution, and the notes an unverified or refused source owes. The
// validators run it on the project's own catalog.
func (r *Registry) Validate() *validate.Problems { return r.validate(true) }

// ValidateStructure is the check the loader applies: the document is well formed. It does
// not judge the terms. No simulation path reads, requires or enforces a licence (X12), so a
// catalog a user supplies is not refused for what its registry says, or fails to say, about
// a source's licence.
func (r *Registry) ValidateStructure() *validate.Problems { return r.validate(false) }

func (r *Registry) validate(terms bool) *validate.Problems {
	p := &validate.Problems{}
	if r.Kind != "MeasurementSources" {
		p.Field("kind", "must be %q, got %q", "MeasurementSources", r.Kind)
	}
	if terms && len(r.Sources) == 0 {
		p.Field("sources", "a registry with no sources permits nothing")
	}
	seen := map[string]bool{}
	for i, l := range r.Sources {
		at := fmt.Sprintf("sources[%d]", i)
		switch {
		case !slug.MatchString(l.ID):
			p.Field(at+".id", "must be a lowercase slug, got %q", l.ID)
		case seen[l.ID]:
			p.Field(at+".id", "duplicate source %q", l.ID)
		}
		seen[l.ID] = true
		l.validate(p, at, terms)
	}
	return p
}

func (l License) validate(p *validate.Problems, at string, terms bool) {
	if l.Title == "" {
		p.Field(at+".title", "required")
	}
	if !l.Status.Valid() {
		p.Field(at+".status", "%q is not one of %v", l.Status, AllLicenseStatuses())
	}
	permitted := map[Use]bool{}
	for j, u := range l.Permits {
		pat := fmt.Sprintf("%s.permits[%d]", at, j)
		switch {
		case !u.Valid():
			p.Field(pat, "%q is not one of %v", u, AllUses())
		case permitted[u]:
			p.Field(pat, "duplicate use %q", u)
		}
		permitted[u] = true
	}
	if !terms {
		return
	}
	switch l.Status {
	case LicenseVerified:
		if l.License == "" {
			p.Field(at+".license", "required for a verified source: the licence the evidence states")
		}
		if permitted[UseRedistribute] && !vocab.LicenceRedistributable(l.License) {
			p.Field(at+".permits", "%q is not on the licence allow-list %v, so the source cannot permit redistribute (X12, D6)",
				l.License, vocab.RedistributableLicences())
		}
		if len(l.Permits) == 0 {
			p.Field(at+".permits", "a verified source states at least one permitted use; one that permits none is refused")
		}
		if len(l.Evidence) == 0 {
			p.Field(at+".evidence", "a verified source quotes its terms")
		}
	case LicenseUnverified, LicenseRefused:
		if len(l.Permits) > 0 {
			p.Field(at+".permits", "a %s source permits nothing", l.Status)
		}
		if l.Note == "" {
			p.Field(at+".note", "a %s source says why", l.Status)
		}
	}
	for j, e := range l.Evidence {
		eat := fmt.Sprintf("%s.evidence[%d]", at, j)
		if !strings.HasPrefix(e.URL, "https://") && !strings.HasPrefix(e.URL, "http://") {
			p.Field(eat+".url", "must be an http(s) URL, got %q", e.URL)
		}
		if e.Quote == "" {
			p.Field(eat+".quote", "required: the text that states the terms")
		}
		if e.Location == "" {
			p.Field(eat+".location", "required: where the quoted text is")
		}
	}
}

// Lookup returns the entry with id, or nil.
func (r *Registry) Lookup(id string) *License {
	if r == nil {
		return nil
	}
	for i := range r.Sources {
		if r.Sources[i].ID == id {
			return &r.Sources[i]
		}
	}
	return nil
}

// Permits reports whether the source id is verified and its terms permit u. An unknown
// id permits nothing. Redistribution also needs a licence on the allow-list of
// contracts/vocab, the one list the catalog, the data ledger and this registry share (X12, D6).
func (r *Registry) Permits(id string, u Use) bool {
	l := r.Lookup(id)
	if l == nil || l.Status != LicenseVerified {
		return false
	}
	if u == UseRedistribute && !vocab.LicenceRedistributable(l.License) {
		return false
	}
	for _, have := range l.Permits {
		if have == u {
			return true
		}
	}
	return false
}

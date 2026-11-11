package routertable

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
)

var (
	token      = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
	repository = regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`)
	commit     = regexp.MustCompile(`^[0-9a-f]{40}$`)
	lines      = regexp.MustCompile(`^([1-9][0-9]*)(?:-([1-9][0-9]*))?$`)
)

// Validate checks an entry on its own: it is complete, it names a policy llm-d's router
// offers, and it carries the evidence for each field it states. It returns every problem it
// finds, or nil. Whether two entries share a key is a check across files (Duplicates).
func (e *Entry) Validate() *validate.Problems {
	p := &validate.Problems{}
	if e.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, e.Kind)
	}
	if e.Name != "" && !token.MatchString(e.Name) {
		p.Field("name", "the file name must be a lowercase slug, got %q", e.Name)
	}
	if e.Name != "" && e.Router != "" && e.Mode != "" && e.Name != e.WantName() {
		p.Field("name", "the file is named %q, but the entry's router, mode and affinity make its name %q", e.Name, e.WantName())
	}
	switch {
	case e.Router == "":
		p.Field("router", "required: the router the run was measured behind, in lower case")
	case !token.MatchString(e.Router):
		p.Field("router", "must be lower case letters, digits, '.', '_' or '-', got %q", e.Router)
	case e.Router == "llm-d":
		p.Field("router", "BLIS hosts llm-d's router, so it has no entry; the table is for the routers it does not host")
	}
	switch {
	case e.Mode == "":
		p.Field("mode", "required: the router's mode or algorithm as the run's configuration names it")
	case !token.MatchString(e.Mode):
		p.Field("mode", "must be lower case letters, digits, '.', '_' or '-', got %q", e.Mode)
	}
	if e.SessionAffinityTTLS != nil && *e.SessionAffinityTTLS < 1 {
		p.Field("session_affinity_ttl_s", "must be at least 1 second when set, got %d", *e.SessionAffinityTTLS)
	}
	switch {
	case e.Policy == "":
		p.Field("policy", "required: a policy llm-d's router offers, or %q", PolicyNone)
	case !e.Policy.Offered():
		p.Field("policy", "%q is not a policy llm-d's router offers: %v, or %q", e.Policy, policyNames(), PolicyNone)
	}
	if strings.TrimSpace(e.Reason) == "" {
		if e.Policy == PolicyNone {
			p.Field("reason", "required: why no policy of llm-d's router is close enough")
		} else {
			p.Field("reason", "required: what the router's mode and the policy do alike and differently")
		}
	}
	e.validateRevisions(p)
	e.validateEvidence(p)
	return p
}

func policyNames() []string {
	return PolicyNames()
}

func (e *Entry) validateEvidence(p *validate.Problems) {
	if len(e.Evidence) == 0 {
		p.Field("evidence", "required: a citation of each router's code or documentation at a pinned revision. An entry is added only with evidence")
		return
	}
	// supported[side][field] counts the citations of one side for one field.
	supported := map[Side]map[Field]int{SideRouter: {}, SideLLMD: {}}
	seen := map[string]bool{}
	for i, c := range e.Evidence {
		at := fmt.Sprintf("evidence[%d]", i)
		c.validate(p, at)
		c.validateSide(p, at, e.Policy)
		if m, ok := supported[c.Side]; ok {
			m[c.Field]++
		}
		key := c.Repository + "@" + c.Commit + ":" + c.Path + ":" + c.Lines + ":" + string(c.Field)
		if seen[key] {
			p.Field(at, "repeats a citation of %s", c.Path)
		}
		seen[key] = true
	}
	// Each field the entry states is supported by a passage of the router that has it, and the
	// policy by both routers' code (SUB-18).
	if supported[SideRouter][FieldMode] == 0 {
		p.Field("evidence", "no citation of the router's own code or documentation supports the mode %q (side router, field mode)", e.Mode)
	}
	if e.SessionAffinityTTLS != nil && supported[SideRouter][FieldSessionAffinityTTLS] == 0 {
		p.Field("evidence", "no citation of the router supports its session-affinity setting (side router, field session_affinity_ttl_s)")
	}
	if e.SessionAffinityTTLS == nil && supported[SideRouter][FieldSessionAffinityTTLS]+supported[SideLLMD][FieldSessionAffinityTTLS] > 0 {
		p.Field("evidence", "a citation supports a session-affinity setting the entry does not state")
	}
	if supported[SideRouter][FieldPolicy] == 0 {
		p.Field("evidence", "no citation of the router's code shows the behaviour that the policy is matched to (side router, field policy)")
	}
	if e.Policy != "" && e.Policy.Offered() && e.Policy != PolicyNone {
		if !e.citesPolicySource() {
			p.Field("evidence", "no llm-d citation is of a file that defines the policy %q; cite one of %v", e.Policy, e.Policy.sources())
		}
	}
	if supported[SideLLMD][FieldPolicy] == 0 {
		p.Field("evidence", "no citation of llm-d's router code or documentation supports the policy %q (side llm-d, field policy)", e.Policy)
	}
}

// sources lists the files of llm-d's router that define a policy.
func (p Policy) sources() []string {
	for _, i := range Policies() {
		if i.Policy == p {
			return i.Sources
		}
	}
	return nil
}

func (e *Entry) citesPolicySource() bool {
	for _, c := range e.Evidence {
		if c.Side != SideLLMD || c.Field != FieldPolicy {
			continue
		}
		for _, f := range e.Policy.sources() {
			if c.Path == f {
				return true
			}
		}
	}
	return false
}

// validateSide checks that a citation of llm-d's side is of llm-d's router and that one of the
// router's side is not: a citation is of the code it says it is.
func (c Citation) validateSide(p *validate.Problems, at string, _ Policy) {
	isLLMD := false
	for _, r := range LLMDRepositories {
		if c.Repository == r {
			isLLMD = true
		}
	}
	switch {
	case c.Side == SideLLMD && !isLLMD:
		p.Field(at+".repository", "a citation of llm-d's side is of one of %v, got %q", LLMDRepositories, c.Repository)
	case c.Side == SideRouter && isLLMD:
		p.Field(at+".repository", "a citation of the router the run used cannot be of llm-d's router, got %q", c.Repository)
	}
}

func (c Citation) validate(p *validate.Problems, at string) {
	switch c.Side {
	case SideRouter, SideLLMD:
	default:
		p.Field(at+".side", "must be %q or %q, got %q", SideRouter, SideLLMD, c.Side)
	}
	switch c.Field {
	case FieldMode, FieldSessionAffinityTTLS, FieldPolicy:
	default:
		p.Field(at+".field", "must be %q, %q or %q, got %q", FieldMode, FieldSessionAffinityTTLS, FieldPolicy, c.Field)
	}
	if c.Side == SideLLMD && c.Field != FieldPolicy {
		p.Field(at+".field", "llm-d's code supports the policy only; the router's mode and affinity are the router's own")
	}
	if !repository.MatchString(c.Repository) {
		p.Field(at+".repository", "must be <owner>/<name>, got %q", c.Repository)
	}
	if !commit.MatchString(c.Commit) {
		p.Field(at+".commit", "must be a full 40-character lowercase commit, a pinned revision, got %q", c.Commit)
	}
	switch {
	case c.Path == "":
		p.Field(at+".path", "required: the file, from the repository's root")
	case strings.HasPrefix(c.Path, "/") || strings.Contains(c.Path, ".."):
		p.Field(at+".path", "is a path from the repository's root, got %q", c.Path)
	}
	m := lines.FindStringSubmatch(c.Lines)
	if m == nil {
		p.Field(at+".lines", "must be a line number or a range <first>-<last>, got %q", c.Lines)
	} else if m[2] != "" {
		a, _ := strconv.Atoi(m[1])
		b, _ := strconv.Atoi(m[2])
		if a >= b {
			p.Field(at+".lines", "the range must run forward, got %q", c.Lines)
		}
	}
	if strings.TrimSpace(c.Claim) == "" {
		p.Field(at+".claim", "required: what the passage shows")
	}
}

// validateRevisions checks the builds an entry is established for, and that the evidence is at
// them: a passage of the router's code shows the behaviour of the build it is read at, so a
// router-side citation at a commit the entry does not list proves nothing about the builds it
// does, and a listed build with no citation has none of its own.
func (e *Entry) validateRevisions(p *validate.Problems) {
	if len(e.Revisions) == 0 {
		p.Field("revisions", "required: the builds of the router the entry is established for, each a full commit. A mapping shown for one build is not inherited by another")
		return
	}
	listed := map[string]bool{}
	for i, r := range e.Revisions {
		at := fmt.Sprintf("revisions[%d]", i)
		switch {
		case !commit.MatchString(r):
			p.Field(at, "must be a full 40-character lowercase commit, got %q", r)
		case listed[r]:
			p.Field(at, "repeats %s", r)
		case i > 0 && r < e.Revisions[i-1]:
			p.Field(at, "revisions are sorted")
		}
		listed[r] = true
	}
	mode, policy := map[string]bool{}, map[string]bool{}
	for i, c := range e.Evidence {
		if c.Side != SideRouter {
			continue
		}
		if !listed[c.Commit] {
			p.Field(fmt.Sprintf("evidence[%d].commit", i), "the router-side citation is of %s, which is not among the entry's revisions: it shows another build than the ones the entry covers", c.Commit)
			continue
		}
		switch c.Field {
		case FieldMode:
			mode[c.Commit] = true
		case FieldPolicy:
			policy[c.Commit] = true
		}
	}
	for _, r := range e.Revisions {
		if listed[r] && commit.MatchString(r) && (!mode[r] || !policy[r]) {
			p.Field("evidence", "revision %s has no router-side citation for %s: each build the entry covers needs its own", r, missing(mode[r], policy[r]))
		}
	}
}

func missing(mode, policy bool) string {
	switch {
	case !mode && !policy:
		return "its mode and its behaviour"
	case !mode:
		return "its mode"
	}
	return "its behaviour"
}

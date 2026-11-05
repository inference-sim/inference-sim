package model

import (
	"strings"
	"testing"
)

// A valid identity carries a non-empty name and a source whose provider, repo and
// revision are all present. retrieved is optional, so a card that omits it still
// validates — the catalog's own validate_models does not require it, and the Go gate
// must agree rather than diverge.
func TestIdentityValidateAcceptsAWellFormedCard(t *testing.T) {
	id := &Identity{
		Name: "deepseek-v4-pro",
		Source: &Source{
			Provider:  "huggingface",
			Repo:      "deepseek-ai/DeepSeek-V4-Pro",
			Revision:  "b5968e9190ef611bbf34a7229255be88a0e937c1",
			Retrieved: "2026-10-02",
		},
	}
	if p := id.Validate(); !p.OK() {
		t.Fatalf("a well-formed identity failed validation:\n%s", p.Error())
	}
}

// retrieved is optional: a card without it is still valid, because the catalog's
// Python gate does not check it and the two gates must not disagree.
func TestIdentityValidateAcceptsAMissingRetrieved(t *testing.T) {
	id := &Identity{
		Name: "qwen3-14b",
		Source: &Source{
			Provider: "huggingface",
			Repo:     "Qwen/Qwen3-14B",
			Revision: "abc123",
		},
	}
	if p := id.Validate(); !p.OK() {
		t.Fatalf("a card omitting retrieved failed validation:\n%s", p.Error())
	}
}

// Each required field, when missing, must be reported at its own path so a reviewer
// fixes every one in a single pass rather than one run at a time.
func TestIdentityValidateReportsEveryMissingField(t *testing.T) {
	cases := []struct {
		name string
		id   *Identity
		want string // a path the error list must mention
	}{
		{"missing name", &Identity{Source: &Source{Provider: "p", Repo: "r", Revision: "v"}}, "name"},
		{"missing source entirely", &Identity{Name: "m"}, "source"},
		{"missing provider", &Identity{Name: "m", Source: &Source{Repo: "r", Revision: "v"}}, "source.provider"},
		{"missing repo", &Identity{Name: "m", Source: &Source{Provider: "p", Revision: "v"}}, "source.repo"},
		{"missing revision", &Identity{Name: "m", Source: &Source{Provider: "p", Repo: "r"}}, "source.revision"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := c.id.Validate()
			if p.OK() {
				t.Fatalf("%s validated clean; a required field was not enforced", c.name)
			}
			found := false
			for _, pr := range p.All() {
				if pr.Path == c.want {
					found = true
				}
			}
			if !found {
				t.Errorf("no problem reported at path %q; got:\n%s", c.want, p.Error())
			}
		})
	}
}

// A source with all three required strings but an empty one is as unusable as an
// absent field: an empty provider or revision names nothing.
func TestIdentityValidateRejectsEmptyRequiredStrings(t *testing.T) {
	id := &Identity{Name: "m", Source: &Source{Provider: "huggingface", Repo: "", Revision: "v"}}
	p := id.Validate()
	if p.OK() {
		t.Fatal("an empty repo validated clean")
	}
	if !strings.Contains(p.Error(), "source.repo") {
		t.Errorf("the empty repo was not reported at source.repo:\n%s", p.Error())
	}
}

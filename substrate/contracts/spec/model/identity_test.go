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
		Licence: &Licence{ID: "MIT", Evidence: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/b5968e9190ef611bbf34a7229255be88a0e937c1/README.md"},
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
		Licence: &Licence{ID: "Apache-2.0", Evidence: "https://huggingface.co/Qwen/Qwen3-14B/raw/abc123/README.md"},
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

// X12: the structural check does not look at the licence block; the licence check does.
func TestIdentityValidateLeavesTheLicenceToCI(t *testing.T) {
	id := &Identity{Name: "m", Source: &Source{Provider: "p", Repo: "r", Revision: "v"}}
	if p := id.Validate(); !p.OK() {
		t.Fatalf("an identity with no licence failed the structural check:\n%s", p.Error())
	}
	p := id.ValidateLicence()
	if p.OK() || !strings.Contains(p.Error(), "licence") || !strings.Contains(p.Error(), "X12") {
		t.Errorf("the licence check did not require a licence naming X12: %v", p.Error())
	}
	id.Licence = &Licence{ID: "LicenseRef-other", Evidence: "http://x"}
	if p := id.Validate(); !p.OK() {
		t.Errorf("an invalid licence failed the structural check:\n%s", p.Error())
	}
	if id.ValidateLicence().OK() {
		t.Error("an invalid licence passed the licence check")
	}
}

// A source may name the config's path inside the repository, for a model shipped in a
// directory of its own. The path is a relative .json file; nothing that leaves the repository.
func TestSourcePathNamesAFileInsideTheRepository(t *testing.T) {
	src := func(path string) *Identity {
		return &Identity{Name: "m", Source: &Source{Provider: "huggingface", Repo: "o/m", Revision: "r", Path: path}}
	}
	for _, ok := range []string{"", "config.json", "dspark/config.json", "a/b/c.json"} {
		if p := src(ok).Validate(); !p.OK() {
			t.Errorf("path %q was refused:\n%s", ok, p.Error())
		}
	}
	for _, bad := range []string{"/config.json", "../config.json", "a/../config.json", "a//b.json", "./config.json",
		"dspark/", "dspark/config.txt", "dspark\\config.json",
		"a?x=1/config.json", "a#b/config.json", "a%2e/config.json", "a b/config.json", "a\x00/config.json"} {
		p := src(bad).Validate()
		if p.OK() || !strings.Contains(p.Error(), "source.path") {
			t.Errorf("path %q was accepted: %v", bad, p)
		}
	}
	if got := (&Source{}).ConfigPath(); got != "config.json" {
		t.Errorf("the default path is %q, not config.json", got)
	}
	if got := (&Source{Path: "dspark/config.json"}).ConfigPath(); got != "dspark/config.json" {
		t.Errorf("a stated path became %q", got)
	}
}

// A drafter list and a target are checked for what a file can know about itself: no empty or
// repeated name, no model that drafts for itself, and a separate drafter lists none of its own.
// The links between files are the catalog check's (validate/dataset).
func TestIdentityValidateChecksDrafterNames(t *testing.T) {
	base := func() *Identity {
		return &Identity{Name: "m", Source: &Source{Provider: "huggingface", Repo: "o/m", Revision: "r"}}
	}
	good := base()
	good.Drafters = []string{InGraphSpeculator, "m-eagle3"}
	if p := good.Validate(); !p.OK() {
		t.Fatalf("a target listing its speculator and a drafter was refused:\n%s", p.Error())
	}
	drafter := base()
	drafter.Name, drafter.Targets = "m-eagle3", []string{"m", "m-fp8"}
	if p := drafter.Validate(); !p.OK() {
		t.Fatalf("a drafter naming its target was refused:\n%s", p.Error())
	}
	for name, mut := range map[string]func(*Identity){
		"an empty drafter":          func(i *Identity) { i.Drafters = []string{""} },
		"a model drafting itself":   func(i *Identity) { i.Drafters = []string{"m"} },
		"a drafter listed twice":    func(i *Identity) { i.Drafters = []string{"a", "a"} },
		"a model as its own target": func(i *Identity) { i.Targets = []string{"t", "m"} },
		"an empty target":           func(i *Identity) { i.Targets = []string{""} },
		"a target listed twice":     func(i *Identity) { i.Targets = []string{"t", "t"} },
		"a drafter with drafters":   func(i *Identity) { i.Targets, i.Drafters = []string{"t"}, []string{"a"} },
	} {
		id := base()
		mut(id)
		if p := id.Validate(); p.OK() {
			t.Errorf("%s was accepted", name)
		}
	}
}

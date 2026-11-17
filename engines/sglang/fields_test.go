// Original to blis-sglang.

package blissglang

import (
	"bytes"
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// fields.json is the field list SGL-3 asks for: every configuration field and environment
// variable SGLang v0.5.23 reads, each with a status, its evidence and an estimation rule. The
// names come from SGLang's source (fields/gen_fields.py, checked against the pinned commit in
// CI); this test holds the file to the document type blis-substrate defines for it.
//
//   - It loads through the substrate's strict loader (fields.Load), which refuses unknown keys,
//     a status without its evidence, an issue that is not <owner>/<repo>#<n>, and an unsorted or
//     repeated entry.
//   - It is for the engine and release this build models (Kind, Upstream).
//   - It is in canonical form: the substrate's Marshal writes the same bytes.
//   - Every trace a modelled field cites is a file in this repository.
func TestTheFieldListLoadsAndNamesThisRelease(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	if l.Engine.Kind != Kind || l.Engine.Version != Upstream.Version {
		t.Errorf("fields.json is for %s %s; this build models %s %s", l.Engine.Kind, l.Engine.Version, Kind, Upstream.Version)
	}
	raw, err := os.ReadFile("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, canonical) {
		t.Error("fields.json is not in the substrate's canonical form; regenerate it with fields/gen_fields.py")
	}
	// A floor, so a generator that emits almost nothing cannot pass: SGLang v0.5.23 declares
	// more than 540 ServerArgs fields and more than 770 Envs fields.
	var config, env int
	for _, f := range l.Fields {
		switch f.Source {
		case fields.SourceConfig:
			config++
		case fields.SourceEnv:
			env++
		}
		for _, tr := range f.Evidence.Traces {
			if strings.HasPrefix(tr, "/") || strings.Contains(tr, "..") {
				t.Errorf("%s: trace %q is not a path inside this repository", f.Name, tr)
				continue
			}
			if _, err := os.Stat(tr); err != nil {
				t.Errorf("%s: trace %q: %v", f.Name, tr, err)
			}
		}
	}
	if config < 540 || env < 770 {
		t.Errorf("fields.json lists %d configuration fields and %d environment variables; SGLang v0.5.23 has more than 540 and 770", config, env)
	}
}

// A not_modelled field is refused when a run sets it away from its default, and the refusal
// names the issue that will model it. X28 keeps that issue open while it is cited, so the issue
// must be one of this repository's: an issue elsewhere cannot be held open by this repository's
// checks.
func TestEveryNotModelledFieldCitesAnIssueOfThisRepository(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	own := regexp.MustCompile(`^inference-sim/blis-sglang#[1-9][0-9]*$`)
	for _, f := range l.Fields {
		if f.Status == fields.StatusNotModelled && !own.MatchString(f.Evidence.Issue) {
			t.Errorf("%s:%s is not_modelled and cites %q, not an inference-sim/blis-sglang issue", f.Source, f.Name, f.Evidence.Issue)
		}
	}
}

// fields/README.md names every not_modelled field whose estimation rule is leave_unset, and only
// those. The list drifts the moment a field gains or loses the rule, so this test fails either way.
func TestReadmeNamesEveryLeaveUnsetField(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	section := leaveUnsetSection(t)
	want := map[string]bool{}
	for _, f := range l.Fields {
		if f.Status == fields.StatusNotModelled && f.Estimation.Rule == fields.RuleLeaveUnset {
			want[f.Name] = true
			if !strings.Contains(section, "`"+f.Name+"`") {
				t.Errorf("fields.json lists %s as not_modelled with leave_unset, and fields/README.md's list does not name it", f.Name)
			}
		}
	}
	for _, m := range regexp.MustCompile("`([A-Za-z_][A-Za-z0-9_.]*)`").FindAllStringSubmatch(section, -1) {
		if !want[m[1]] {
			t.Errorf("fields/README.md's leave_unset list names %s, which fields.json does not list as not_modelled with leave_unset", m[1])
		}
	}
}

func leaveUnsetSection(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile("fields/README.md")
	if err != nil {
		t.Fatal(err)
	}
	readme := string(raw)
	start := strings.Index(readme, "Every `not_modelled` field with the rule `leave_unset` is named below")
	if start < 0 {
		t.Fatal("fields/README.md has no list of the not_modelled leave_unset fields")
	}
	end := strings.Index(readme[start:], "End of the leave_unset list.")
	if end < 0 {
		t.Fatal("fields/README.md: the leave_unset list has no end marker")
	}
	// The sentence that opens the list names the rule itself, not a field.
	body := readme[start : start+end]
	return body[strings.Index(body, "\n"):]
}

// declarations is fields/declarations.json, the part config/ reads.
type declarations struct {
	ServerArgs map[string]struct {
		Flags []string `json:"flags"`
		NoCLI bool     `json:"no_cli"`
		NoArg bool     `json:"no_arg"`
	} `json:"server_args"`
	CLIOnly map[string]json.RawMessage `json:"cli_only"`
	Env     map[string]json.RawMessage `json:"env"`
}

// fields/declarations.json and fields.json are written from the same source by the same
// generator; config/ builds ServerArgs from the first and refuses by the second. A configuration
// field one has and the other lacks would be accepted without a status, or refused without a
// declaration. The no_cli rule also holds: a field without a flag has no flag names.
func TestTheDeclarationsCoverExactlyTheConfigurationFields(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("fields/declarations.json")
	if err != nil {
		t.Fatal(err)
	}
	var d declarations
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatal(err)
	}
	listed := map[string]bool{}
	for _, f := range l.Fields {
		if f.Source != fields.SourceConfig {
			continue
		}
		listed[f.Name] = true
		_, sa := d.ServerArgs[f.Name]
		_, cli := d.CLIOnly[f.Name]
		if !sa && !cli {
			t.Errorf("config:%s is in fields.json and has no declaration", f.Name)
		}
	}
	for n, decl := range d.ServerArgs {
		if !listed[n] {
			t.Errorf("ServerArgs.%s is declared and fields.json has no status for it", n)
		}
		if (decl.NoCLI || decl.NoArg) && len(decl.Flags) != 0 {
			t.Errorf("ServerArgs.%s has no flag (no_cli or no Arg) and lists flags %v", n, decl.Flags)
		}
		if !decl.NoCLI && !decl.NoArg && len(decl.Flags) == 0 {
			t.Errorf("ServerArgs.%s has an Arg and no flag", n)
		}
	}
	for n := range d.CLIOnly {
		if !listed[n] {
			t.Errorf("the flag for %s is registered by hand and fields.json has no status for it", n)
		}
	}
}

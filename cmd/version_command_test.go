package cmd

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	latencykernel "github.com/inference-sim/inference-sim/substrate/kernel"
)

func withBundledStack(t *testing.T) {
	t.Helper()
	b, err := os.ReadFile("../stack.json")
	if err != nil {
		t.Fatal(err)
	}
	prev := bundledStack
	SetBundledStack(b)
	t.Cleanup(func() { SetBundledStack(prev) })
}

// runVersion writes `blis version`'s output for the given --json flag and returns it.
func runVersion(t *testing.T, asJSON bool) string {
	t.Helper()
	prev := versionJSON
	versionJSON = asJSON
	t.Cleanup(func() { versionJSON = prev })
	var out bytes.Buffer
	versionCmd.SetOut(&out)
	t.Cleanup(func() { versionCmd.SetOut(nil) })
	if err := versionCmd.RunE(versionCmd, nil); err != nil {
		t.Fatal(err)
	}
	return out.String()
}

// byKind is the report's records keyed by engine kind.
func byKind(t *testing.T, models []engineReport) map[string]engineReport {
	t.Helper()
	out := map[string]engineReport{}
	for _, m := range models {
		if _, dup := out[m.Kind]; dup {
			t.Fatalf("%q is reported twice", m.Kind)
		}
		out[m.Kind] = m
	}
	return out
}

// The version command prints the commit of every engine this build runs, the llm-d commit and
// the substrate, which a scenario no longer states (LIF-6).
func TestVersionCommandPrintsTheCommitsTheBuildRuns(t *testing.T) {
	withBundledStack(t)
	out := runVersion(t, false)
	for _, e := range engineModels() {
		want := "engine " + e.Kind + " " + e.ModeledVersion + " " + e.ModeledCommit +
			" (rules from " + e.BuildVersion + " " + e.BuildCommit + "; roles "
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
	if !regexp.MustCompile(`(?m)^llm-d \S+ [0-9a-f]{40} \(router fork `).MatchString(out) ||
		!strings.Contains(out, "substrate ") || !strings.HasPrefix(out, "inference-sim ") {
		t.Errorf("output lacks the simulator, substrate or llm-d line:\n%s", out)
	}
}

// `blis version` reports every kind this build runs, the release each models, and the upstream
// tree its rules came from (LIF-6). The modelled release is the one a price belongs to; the tree
// is provenance and may be a commit with no tag, so the two are not interchangeable.
func TestVersionJSONReportsTheReleaseEachKindModels(t *testing.T) {
	withBundledStack(t)
	var doc versionDocument
	if err := json.Unmarshal([]byte(runVersion(t, true)), &doc); err != nil {
		t.Fatalf("the output is not JSON: %v", err)
	}
	built := builtEngines()
	if len(doc.Engines) != len(built) {
		t.Fatalf("reported %d kinds, this build runs %d: %+v", len(doc.Engines), len(built), doc.Engines)
	}
	if doc.InferenceSim == "" || doc.Substrate == "" || doc.LLMD == nil {
		t.Errorf("the document leaves the build's identity incomplete: %+v", doc)
	}
	if doc.LLMD.Commit == "" || doc.LLMD.RouterFork == "" {
		t.Errorf("the llm-d block leaves the release or router fork empty: %+v", doc.LLMD)
	}
	got := byKind(t, doc.Engines)

	vllm, ok := got[engineKindVLLM]
	if !ok {
		t.Fatalf("no vLLM record: %+v", doc.Engines)
	}
	// The kernel's behaviour release is what a Result's assumptions cite, so it is what the
	// report must name: a user matches benchmarked figures to it.
	if vllm.ModeledVersion != latencykernel.EngineBehaviourVersion ||
		vllm.ModeledCommit != latencykernel.EngineBehaviourCommit {
		t.Errorf("vLLM models %s (%s), want the kernel's behaviour release %s (%s)",
			vllm.ModeledVersion, vllm.ModeledCommit,
			latencykernel.EngineBehaviourVersion, latencykernel.EngineBehaviourCommit)
	}
	if b := built[engineKindVLLM]; vllm.BuildVersion != b.Version || vllm.BuildCommit != b.Commit {
		t.Errorf("vLLM rules came from %s (%s), want the build's upstream %s (%s)",
			vllm.BuildVersion, vllm.BuildCommit, b.Version, b.Commit)
	}
	if vllm.ModeledVersion == "" || vllm.ModeledCommit == "" {
		t.Errorf("the vLLM record leaves the modelled release unreported: %+v", vllm)
	}
	// Non-vacuity: the build's engine snapshot is not the release the engine models, so a
	// report that printed one for the other would be caught above.
	if vllm.ModeledVersion == vllm.BuildVersion {
		t.Errorf("this build's modelled release and its engine snapshot are both %q: the record "+
			"cannot show the two are reported separately", vllm.ModeledVersion)
	}
	if want := []deployment.Role{deployment.RoleColocated, deployment.RolePrefill, deployment.RoleDecode}; !reflect.DeepEqual(vllm.Roles, want) {
		t.Errorf("vLLM roles = %v, want %v", vllm.Roles, want)
	}

	sglang, ok := got[engineKindSGLang]
	if !ok {
		t.Fatalf("no SGLang record: %+v", doc.Engines)
	}
	// SGLang has one upstream release, so its modelled release is the tree it ports; the
	// record must still report both members and both commits.
	if b := built[engineKindSGLang]; sglang.ModeledVersion != b.Version ||
		sglang.ModeledCommit != b.Commit || sglang.BuildVersion != b.Version ||
		sglang.BuildCommit != b.Commit {
		t.Errorf("SGLang record %+v does not name the one release the build runs (%s, %s)",
			sglang, b.Version, b.Commit)
	}
	// The engine refuses prefill/decode disaggregation (inference-sim/inference-sim#2351), so a
	// report naming a P/D role would offer a mode the build cannot run.
	if want := []deployment.Role{deployment.RoleColocated}; !reflect.DeepEqual(sglang.Roles, want) {
		t.Errorf("SGLang roles = %v, want %v: the engine models no disaggregated pool", sglang.Roles, want)
	}

	for _, m := range doc.Engines {
		if m.Kind == "" || m.ModeledVersion == "" || m.ModeledCommit == "" || len(m.Roles) == 0 {
			t.Errorf("record %+v leaves a field a caller needs empty", m)
		}
		for role := range m.Roles {
			if !m.Roles[role].Valid() {
				t.Errorf("%s reports the role %q, which the deployment contract does not define", m.Kind, m.Roles[role])
			}
		}
	}

	// Sorted by kind, so two runs of the same build are byte-identical (X6).
	if kinds := []string{doc.Engines[0].Kind, doc.Engines[1].Kind}; !reflect.DeepEqual(kinds, []string{"sglang", "vllm"}) {
		t.Errorf("kinds are ordered %v, want sorted", kinds)
	}
}

// A build with no compiled-in stack still reports its engines and omits the llm-d block rather
// than inventing one.
func TestVersionJSONOmitsLLMDWithoutAStack(t *testing.T) {
	prev := bundledStack
	SetBundledStack(nil)
	t.Cleanup(func() { SetBundledStack(prev) })
	doc := versionDocumentOf(revisionsFrom(nil), engineModels())
	if doc.LLMD != nil {
		t.Errorf("a build with no stack reports llm-d %+v", doc.LLMD)
	}
	if len(doc.Engines) == 0 {
		t.Error("a build with no stack reports no engines")
	}
}

// The text output names the modelled release, the tree the rules came from and the roles.
func TestVersionTextNamesTheReleaseTheRolesAndTheTree(t *testing.T) {
	withBundledStack(t)
	out := runVersion(t, false)
	built := builtEngines()[engineKindVLLM]
	for _, want := range []string{
		"engine vllm " + latencykernel.EngineBehaviourVersion + " ",
		"rules from " + built.Version,
		"roles colocated,prefill,decode",
		"engine sglang " + builtEngines()[engineKindSGLang].Version + " ",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
	// A caller reading the text must not see the engine snapshot where the modelled release
	// belongs: the snapshot may be a commit with no release tag.
	if strings.Contains(out, "engine vllm "+built.Version+" ") {
		t.Errorf("the text reports the engine snapshot %q as the modelled release:\n%s", built.Version, out)
	}
}

// `version` is the one command that reports the build and its engines, and it takes --json; the
// separate `engines` command the facts used to live in is gone.
func TestVersionIsTheOnlyReportAndTakesJSON(t *testing.T) {
	var version, engines bool
	for _, c := range rootCmd.Commands() {
		switch c.Name() {
		case "version":
			version = true
		case "engines":
			engines = true
		}
	}
	if !version {
		t.Fatal("`blis version` is not registered")
	}
	if engines {
		t.Fatal("`blis engines` is still registered: the build's engine report belongs to `version`")
	}
	if versionCmd.Flags().Lookup("json") == nil {
		t.Fatal("`blis version` takes no --json")
	}
}

// Two reports of the same build are byte-identical: the command reads no clock, no environment
// and no unsorted map (X6).
func TestVersionIsDeterministic(t *testing.T) {
	first, second := runVersion(t, true), runVersion(t, true)
	if first != second {
		t.Errorf("two reports of one build differ:\n--- first ---\n%s\n--- second ---\n%s", first, second)
	}
}

// Every Result records the llm-d commit beside the engine commits.
func TestRevisionsRecordTheLLMDCommit(t *testing.T) {
	withBundledStack(t)
	if got := revisionsFrom(nil).LLMD; !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(got) {
		t.Errorf("revisions.llm_d = %q, want the 40-hex llm-d commit from stack.json", got)
	}
}

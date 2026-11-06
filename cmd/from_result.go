package cmd

// run --from-result and replay --from-result (#2054, SIM-10, DEV-7): reproduce a Result from
// the input it embeds. The Result records its scenario, its deployment, its policy bundle and
// every command-line option that set the run (input.options), so the command rebuilds the
// invocation from the document. The files the options name are not in the document, only their
// sha256: the caller supplies each again and it is checked against the digest. The new Result
// names the one it reproduces (rerun_of) and each component whose revision differs
// (revision_changes), and is otherwise the same document outside runtime (SIM-10).

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// fromResultPath is --from-result: a Result file to reproduce.
var fromResultPath string

// reproduction is the original Result a command was asked to reproduce.
type reproduction struct {
	doc    *result.Result
	ref    result.Reference
	tmpDir string
}

// fromResult is the reproduction in progress, nil without --from-result.
var fromResult *reproduction

// registerFromResultFlag adds --from-result to cmd.
func registerFromResultFlag(cmd *cobra.Command) {
	cmd.Flags().StringVar(&fromResultPath, "from-result", "",
		"Reproduce the Result in this file: rebuild the run from the scenario, deployment, policy and options it records. "+
			"Pass again each file an option named (the Result holds only its sha256) and, for a router that names files, --scenarios. "+
			"The new Result names this one in rerun_of and lists the revisions that changed")
}

// applyFromResult prepares command ("run" or "replay") to reproduce --from-result, before the
// command reads any flag. It returns a cleanup for the scenario bundle it writes. Without
// --from-result it does nothing.
func applyFromResult(command string, cmd *cobra.Command) func() {
	fromResult = nil
	if fromResultPath == "" {
		return func() {}
	}
	raw, err := os.ReadFile(fromResultPath)
	if err != nil {
		logrus.Fatalf("--from-result %s: %v", fromResultPath, err)
	}
	var doc result.Result
	if err := result.Unmarshal(raw, &doc); err != nil {
		logrus.Fatalf("--from-result %s: not a Result this build reads: %v", fromResultPath, err)
	}
	if p := doc.Validate(); !p.OK() {
		logrus.Fatalf("--from-result %s: the Result does not validate: %v", fromResultPath, p)
	}
	isTrace := replaysATrace(&doc)
	if (command == "replay") != isTrace {
		other := map[string]string{"run": "replay", "replay": "run"}[command]
		logrus.Fatalf("--from-result %s: this Result came from `blis %s` (its frame's generator), not `blis %s`; use `blis %s --from-result`",
			fromResultPath, other, command, other)
	}
	for _, name := range []string{"scenario", "policy-config"} {
		if cmd.Flags().Changed(name) {
			logrus.Fatalf("--%s cannot be combined with --from-result: the Result records its own %s", name, name)
		}
	}
	tmp, err := os.MkdirTemp("", "blis-from-result-")
	if err != nil {
		logrus.Fatalf("--from-result: %v", err)
	}
	cleanup := func() { _ = os.RemoveAll(tmp) }
	fromResult = &reproduction{doc: &doc, tmpDir: tmp,
		ref: result.Reference{Name: doc.Name, SHA256: bytesDigest(raw)}}

	// The scenario and deployment, as the two documents of a bundle file.
	srcDir := kernelScenarioDir // router files are read from the caller's --scenarios, when given
	const bundle = "from-result.yaml"
	var b strings.Builder
	for _, v := range []any{doc.Input.Scenario, doc.Input.Deployment} {
		d, err := yaml.Marshal(v)
		if err != nil {
			logrus.Fatalf("--from-result: %v", err)
		}
		b.WriteString("---\n" + string(d))
	}
	if err := os.WriteFile(filepath.Join(tmp, bundle), []byte(b.String()), 0o644); err != nil {
		logrus.Fatalf("--from-result: %v", err)
	}
	if r := doc.Input.Deployment.Router; r != nil {
		files := append(append([]string{}, r.Manifests...), r.EndpointPickerConfig)
		for _, f := range files {
			if f == "" {
				continue
			}
			if srcDir == "" {
				logrus.Fatalf("--from-result: the deployment's router names %s; pass --scenarios, the directory it is read from", f)
			}
			data, err := os.ReadFile(filepath.Join(srcDir, f))
			if err != nil {
				logrus.Fatalf("--from-result: router file %s: %v", f, err)
			}
			if err := os.MkdirAll(filepath.Dir(filepath.Join(tmp, f)), 0o755); err != nil {
				logrus.Fatalf("--from-result: %v", err)
			}
			if err := os.WriteFile(filepath.Join(tmp, f), data, 0o644); err != nil {
				logrus.Fatalf("--from-result: %v", err)
			}
		}
	}
	setFlag(cmd, "scenarios", tmp)
	setFlag(cmd, "scenario", bundle)
	if doc.Input.Policy != nil {
		d, err := yaml.Marshal(doc.Input.Policy)
		if err != nil {
			logrus.Fatalf("--from-result: %v", err)
		}
		p := filepath.Join(tmp, "policy.yaml")
		if err := os.WriteFile(p, d, 0o644); err != nil {
			logrus.Fatalf("--from-result: %v", err)
		}
		setFlag(cmd, "policy-config", p)
	}
	// The options: each is set to the recorded value, or checked against it when the caller
	// supplied it again. A file option is checked by digest.
	for _, o := range doc.Input.Options {
		f := cmd.Flags().Lookup(o.Name)
		if f == nil {
			logrus.Fatalf("--from-result: the Result records the option --%s, which `blis %s` of this build does not have", o.Name, command)
		}
		if optionFileFlags[o.Name] {
			if !cmd.Flags().Changed(o.Name) {
				logrus.Fatalf("--from-result: the Result was produced with --%s (%s); pass the file again", o.Name, o.Value)
			}
			d, err := fileDigest(f.Value.String())
			if err != nil || "sha256:"+d != o.Value {
				logrus.Fatalf("--from-result: --%s %s is not the file the Result was produced with (recorded %s, got sha256:%s)",
					o.Name, f.Value.String(), o.Value, d)
			}
			continue
		}
		if cmd.Flags().Changed(o.Name) {
			if f.Value.String() != o.Value {
				logrus.Fatalf("--from-result: --%s %s differs from the Result's %s; a reproduction takes the recorded options", o.Name, f.Value.String(), o.Value)
			}
			continue
		}
		setFlag(cmd, o.Name, o.Value)
	}
	return cleanup
}

// setFlag sets a flag as if the caller had passed it.
func setFlag(cmd *cobra.Command, name, value string) {
	if err := cmd.Flags().Set(name, value); err != nil {
		logrus.Fatalf("--from-result: setting --%s %s: %v", name, value, err)
	}
}

// markRerun fills r.RerunOf and r.RevisionChanges from the reproduction, when there is one.
func markRerun(r *result.Result) {
	if fromResult == nil {
		return
	}
	ref := fromResult.ref
	r.RerunOf = &ref
	old := fromResult.doc
	pairs := []struct{ component, from, to string }{
		{"simulator", old.Revisions.Simulator, r.Revisions.Simulator},
		{"substrate", old.Revisions.Substrate, r.Revisions.Substrate},
		{"router", old.Revisions.Router, r.Revisions.Router},
		{"engine_module", old.Revisions.EngineModule, r.Revisions.EngineModule},
		{"catalog", old.Data.Catalog.CombinedHash, r.Data.Catalog.CombinedHash},
		{"registry", old.Data.Registry.CombinedHash, r.Data.Registry.CombinedHash},
	}
	for _, p := range pairs {
		if p.from != p.to && p.to != "" {
			r.RevisionChanges = append(r.RevisionChanges, result.RevisionChange{Component: p.component, From: p.from, To: p.to})
		}
	}
}

// replaysATrace reports whether the Result is a replay's: the generator of its frame binds a
// trace or an agentic replay, rather than a generated stream. It reads the frame as the
// document's generic form, so it names no field of the binding.
func replaysATrace(r *result.Result) bool {
	data, err := yaml.Marshal(r.Frame.Generator)
	if err != nil {
		return false
	}
	var g map[string]any
	if yaml.Unmarshal(data, &g) != nil {
		return false
	}
	for _, v := range g {
		if m, ok := v.(map[string]any); ok {
			if _, trace := m["trace"]; trace {
				return true
			}
			if _, replay := m["replay"]; replay {
				return true
			}
		}
	}
	return false
}

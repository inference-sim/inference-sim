package cmd

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The scenario files a new user is sent to, and the commands the README gives for them, run
// as written. A rename of an engine key in the pinned engine makes every file that still
// spells the old key fail at start, so these are the files that must follow it. Before this
// test a rename left the five files of examplescenarios/ on the old keys, and every README
// quickstart command exited 1, with all other tests green.

// exampleScenarioDirs are the directories of ready-to-run scenario files, relative to cmd/.
var exampleScenarioDirs = []string{"../examplescenarios", "../examples/llm-d/scenarios"}

// exampleCommandDocs are the documents whose `blis run` and `blis replay` commands a reader
// copies to get started.
var exampleCommandDocs = []string{"../README.md", "../examplescenarios/README.md"}

// exampleShapeFlags returns the flags a scenario file needs to start: the instance counts of a
// prefill/decode scenario, one instance per pool otherwise (none).
func exampleShapeFlags(t *testing.T, path string) []string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// A scenario file is a Scenario and a Deployment document; the pools are in the Deployment.
	var roles []string
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	for {
		var doc struct {
			Pools []struct {
				Role string `yaml:"role"`
			} `yaml:"pools"`
		}
		if err := dec.Decode(&doc); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("%s: %v", path, err)
		}
		for _, p := range doc.Pools {
			roles = append(roles, p.Role)
		}
	}
	if slices.Contains(roles, "prefill") && slices.Contains(roles, "decode") {
		return []string{"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1"}
	}
	return nil
}

// Every scenario file the repository ships for a reader to run starts and serves requests.
func TestExampleScenariosEachRun(t *testing.T) {
	ran := 0
	for _, dir := range exampleScenarioDirs {
		files, err := filepath.Glob(filepath.Join(dir, "*.yaml"))
		if err != nil || len(files) == 0 {
			t.Fatalf("non-vacuity: %s holds no scenario file (%v)", dir, err)
		}
		for _, f := range files {
			t.Run(strings.TrimPrefix(f, "../"), func(t *testing.T) {
				args := append([]string{"run", "--scenarios", dir, "--scenario", filepath.Base(f),
					"--rate", "5", "--num-requests", "5", "--seed", "3"}, exampleShapeFlags(t, f)...)
				o := capRun(t, args...)
				if o.num("completed_requests") <= 0 {
					t.Errorf("%s started but served no request", f)
				}
			})
			ran++
		}
	}
	if ran < 10 {
		t.Fatalf("non-vacuity: ran %d example scenarios, the repository ships at least 10", ran)
	}
}

// docCommands returns each `blis run` / `blis replay` command of a document, its `\`
// continuations joined, as the fields a shell would pass.
func docCommands(t *testing.T, file string) [][]string {
	t.Helper()
	raw, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(string(raw), "\n")
	var out [][]string
	for i := 0; i < len(lines); i++ {
		if !docCommandStart.MatchString(lines[i]) || strings.Contains(lines[i], "<") {
			continue
		}
		command := lines[i]
		for strings.HasSuffix(strings.TrimRight(lines[i], " \t"), `\`) && i+1 < len(lines) {
			i++
			command += " " + lines[i]
		}
		fields := strings.Fields(strings.ReplaceAll(command, `\`, " "))
		out = append(out, fields[1:]) // drop ./blis
	}
	return out
}

// runnableForm turns a documented command into one the test can run from cmd/: paths that name
// a file of this repository are made relative to cmd/, the catalog and registry flags (which
// point at checkouts the reader makes) are dropped because runKernelCLI supplies the vendored
// ones, and the request count is cut to five so the test costs seconds, not minutes.
func runnableForm(args []string) []string {
	var out []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch a {
		case "--catalog", "--registry":
			i++
			continue
		case "--num-requests":
			if i+1 < len(args) {
				n, _ := strconv.Atoi(args[i+1])
				out = append(out, a, strconv.Itoa(min(max(n, 1), 5)))
				i++
				continue
			}
		}
		if _, err := os.Stat(filepath.Join("..", a)); err == nil && !strings.HasPrefix(a, "-") {
			a = filepath.Join("..", a)
		}
		out = append(out, a)
	}
	return out
}

// flagValue returns the value after flag in args, or "".
func flagValue(args []string, flag string) string {
	if i := slices.Index(args, flag); i >= 0 && i+1 < len(args) {
		return args[i+1]
	}
	return ""
}

// Every `blis run` and `blis replay` command in the README and the example scenarios' README
// runs. A replay command names a trace the reader exported earlier; the test exports one from
// the same scenario first.
func TestReadmeQuickstartCommandsRun(t *testing.T) {
	ran := 0
	for _, doc := range exampleCommandDocs {
		for _, cmdline := range docCommands(t, doc) {
			args := runnableForm(cmdline)
			scenarios, scenario := flagValue(args, "--scenarios"), flagValue(args, "--scenario")
			if scenarios == "" || scenario == "" {
				t.Errorf("%s: %q names no --scenarios directory and --scenario file", doc, strings.Join(cmdline, " "))
				continue
			}
			t.Run(strings.TrimPrefix(doc, "../")+"/"+strings.Join(cmdline[:min(len(cmdline), 6)], "_"), func(t *testing.T) {
				if args[0] == "replay" {
					prefix := filepath.Join(t.TempDir(), "trace")
					shape := exampleShapeFlags(t, filepath.Join(scenarios, scenario))
					exp := append([]string{"run", "--scenarios", scenarios, "--scenario", scenario,
						"--rate", "5", "--num-requests", "5", "--seed", "3", "--trace-output", prefix}, shape...)
					capRun(t, exp...)
					for i, a := range args {
						switch a {
						case "--trace-header":
							args[i+1] = prefix + ".yaml"
						case "--trace-data":
							args[i+1] = prefix + ".csv"
						}
					}
					args = append(args, shape...)
				}
				capRun(t, args...)
			})
			ran++
		}
	}
	if ran < 8 {
		t.Fatalf("non-vacuity: ran %d documented commands, the READMEs hold at least 8", ran)
	}
}

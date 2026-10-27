package cmd

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// docGoRunCommand matches a documented `go run ./tools/<name>` invocation, or one of the old
// `go run ./cmd/<name>`: the scoring tools moved to tools/, outside the simulator's sim/ and
// cmd/ (phase 2c of #2021), so a page still naming cmd/ documents a command that is gone.
var docGoRunCommand = regexp.MustCompile(`go run \./(cmd|tools)/([A-Za-z0-9_-]+)(.*)`)

// docCdLine matches a shell `cd <dir>` line inside a code block.
var docCdLine = regexp.MustCompile(`^\s*cd\s+(\S+)`)

// goFlagDefinition matches a standard-library flag definition: flag.Int("name", ...) or
// flag.IntVar(&v, "name", ...), for every flag type.
var goFlagDefinition = regexp.MustCompile(`flag\.[A-Z][A-Za-z0-9]*\((?:&[^,]+,\s*)?"([^"]+)"`)

// definedGoFlags returns the flags the main package in dir defines.
func definedGoFlags(t *testing.T, dir string) map[string]bool {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	flags := map[string]bool{"h": true, "help": true}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range goFlagDefinition.FindAllStringSubmatch(string(src), -1) {
			flags[m[1]] = true
		}
	}
	return flags
}

// documentedFlags returns the flag names an example passes, up to the first shell operator
// or comment.
func documentedFlags(args string) []string {
	var out []string
	for _, field := range strings.Fields(args) {
		if field == "#" || strings.HasPrefix(field, "#") || strings.ContainsAny(field[:1], "|>&;") {
			break
		}
		if !strings.HasPrefix(field, "-") || field == "-" {
			continue
		}
		name := strings.TrimLeft(field, "-")
		if i := strings.Index(name, "="); i >= 0 {
			name = name[:i]
		}
		out = append(out, name)
	}
	return out
}

// Every documented scorer command -- `go run ./tools/<name> ...` -- names a command this
// repository has and passes only flags that command defines, so a reader can run it as
// written. A command in a code block that first `cd`s into another repository (the latency
// kernel, vLLM) belongs to that repository and is not checked here. Behavioural: a removed
// command or flag (estimatorscore, -length-range-ratio) fails it, as either would abort.
func TestDocScorerCommandsUseCommandsAndFlagsThatExist(t *testing.T) {
	files := collectDocFiles(t)
	checked := 0
	for _, file := range files {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		inBlock, elsewhere := false, false
		lines := strings.Split(string(src), "\n")
		for i := 0; i < len(lines); i++ {
			line := lines[i]
			if strings.HasPrefix(strings.TrimSpace(line), "```") {
				inBlock, elsewhere = !inBlock, false
				continue
			}
			if m := docCdLine.FindStringSubmatch(line); inBlock && m != nil {
				dir := filepath.Base(strings.TrimRight(m[1], "/"))
				elsewhere = dir != "inference-sim" && dir != "." && !strings.Contains(m[1], "inference-sim")
				continue
			}
			m := docGoRunCommand.FindStringSubmatch(line)
			if m == nil || elsewhere {
				continue
			}
			command := line
			for strings.HasSuffix(strings.TrimRight(lines[i], " \t"), `\`) && i+1 < len(lines) {
				i++
				command += " " + lines[i]
			}
			m = docGoRunCommand.FindStringSubmatch(command)
			checked++
			name := m[1] + "/" + m[2]
			dir := filepath.Join("..", m[1], m[2])
			if info, err := os.Stat(dir); err != nil || !info.IsDir() {
				t.Errorf("%s:%d: documents `go run ./%s`, which this repository does not have", file, i+1, name)
				continue
			}
			defined := definedGoFlags(t, dir)
			for _, f := range documentedFlags(strings.ReplaceAll(m[3], `\`, " ")) {
				if !defined[f] {
					t.Errorf("%s:%d: `go run ./%s` passes -%s, which it does not define, so it aborts "+
						"with \"flag provided but not defined\"", file, i+1, name, f)
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("non-vacuity: found no documented scorer command to check")
	}
}

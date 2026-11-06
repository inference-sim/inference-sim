// Command modlint is the selected-version lint (X11, LIF-2). Run from a BLIS module's root:
//
//	go run github.com/inference-sim/blis-substrate/tools/modlint@<substrate version>
//
// It reads the versions Go selects (go list -m -json all) and the module graph (go mod
// graph), not only the text of go.mod, and fails on any of these:
//
//   - a replace directive outside inference-sim; in inference-sim, anything but exactly
//     one, from github.com/llm-d/llm-d-router to github.com/inference-sim/llm-d-router
//     at a -blis.N tag;
//   - a module other than inference-sim that requires the router;
//   - an exclude directive outside inference-sim, or one that names anything but a plain
//     tag of an engine module;
//   - a committed go.work or go.work.sum;
//   - a BLIS module selected at a pseudo-version or through a replace;
//   - a selected module whose go directive is above contracts/toolchain;
//   - a go command whose version differs from contracts/toolchain.
//
// contracts/toolchain is read from this repository in blis-substrate, and otherwise from
// the blis-substrate version the build selects; -toolchain names it for a module that
// does not yet select the substrate. Exit status: 0 clean, 1 violations,
// 2 the lint could not run.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const substrate = "github.com/inference-sim/blis-substrate"

func main() {
	dir := flag.String("C", ".", "the module root")
	tcFlag := flag.String("toolchain", "", "the toolchain version to check against, instead of reading contracts/toolchain")
	deferral := flag.String("deferral", "", "the pin-by-commit deferral file; while it holds, a BLIS module pinned by commit is reported as deferred")
	today := flag.String("today", "", "the date to judge the deferral's expiry on (YYYY-MM-DD); default today, UTC")
	flag.Parse()
	bad, err := lintDir(*dir, *tcFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "modlint:", err)
		os.Exit(2)
	}
	if *deferral != "" {
		src, rerr := os.ReadFile(*deferral)
		if rerr != nil && !errors.Is(rerr, fs.ErrNotExist) {
			fmt.Fprintln(os.Stderr, "modlint:", rerr)
			os.Exit(2)
		}
		day := time.Now().UTC()
		if *today != "" {
			if day, rerr = time.Parse("2006-01-02", *today); rerr != nil {
				fmt.Fprintln(os.Stderr, "modlint: -today:", rerr)
				os.Exit(2)
			}
		}
		var deferred []string
		bad, deferred = ApplyDeferral(src, day, bad)
		if len(deferred) > 0 {
			fmt.Printf("modlint: %d finding(s) deferred (see %s)\n", len(deferred), *deferral)
		}
	}
	for _, b := range bad {
		fmt.Println(b)
	}
	if len(bad) > 0 {
		os.Exit(1)
	}
	fmt.Println("modlint: clean")
}

func lintDir(dir, toolchain string) ([]string, error) {
	gomod, err := os.ReadFile(filepath.Join(dir, "go.mod"))
	if err != nil {
		return nil, err
	}
	in := Input{GoMod: gomod}
	out, err := gocmd(dir, "list", "-m", "-json", "all")
	if err != nil {
		return nil, err
	}
	if in.Modules, err = decodeModules(out); err != nil {
		return nil, err
	}
	graph, err := gocmd(dir, "mod", "graph")
	if err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(bytes.NewReader(graph))
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) == 2 {
			in.Graph = append(in.Graph, [2]string{f[0], f[1]})
		}
	}
	files, err := exec.Command("git", "-C", dir, "ls-files", "--", "go.work", "go.work.sum", "**/go.work").Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files: %w", err)
	}
	in.GoWork = strings.Fields(string(files))
	in.Toolchain = toolchain
	if toolchain == "" {
		if in.Toolchain, err = toolchainFile(dir, in.Modules); err != nil {
			return nil, err
		}
	}
	v, err := gocmd(dir, "env", "GOVERSION")
	if err != nil {
		return nil, err
	}
	in.GoVersion = strings.TrimSpace(string(v))
	return Lint(in)
}

func gocmd(dir string, args ...string) ([]byte, error) {
	cmd := exec.Command("go", args...)
	cmd.Dir = dir
	// The lint reads the module graph as committed; a workspace would change it.
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=readonly")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("go %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}

func decodeModules(raw []byte) ([]Module, error) {
	var mods []Module
	dec := json.NewDecoder(bytes.NewReader(raw))
	for {
		var m struct {
			Module
			Dir string
		}
		if err := dec.Decode(&m); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}
		mods = append(mods, m.Module)
	}
	return mods, nil
}

// toolchainFile returns the version contracts/toolchain names: this repository's file in
// blis-substrate, and the selected substrate version's file elsewhere.
func toolchainFile(dir string, mods []Module) (string, error) {
	for _, m := range mods {
		if m.Main && m.Path == substrate {
			raw, err := os.ReadFile(filepath.Join(dir, "contracts", "toolchain"))
			return strings.TrimSpace(string(raw)), err
		}
	}
	out, err := gocmd(dir, "list", "-m", "-f", "{{.Dir}}", substrate)
	if err != nil {
		return "", fmt.Errorf("this module does not select %s, so contracts/toolchain cannot be read: %w", substrate, err)
	}
	d := strings.TrimSpace(string(out))
	if d == "" {
		return "", fmt.Errorf("%s is not downloaded; run go mod download %s", substrate, substrate)
	}
	raw, err := os.ReadFile(filepath.Join(d, "contracts", "toolchain"))
	if err != nil {
		return "", fmt.Errorf("the selected %s has no contracts/toolchain: %w", substrate, err)
	}
	return strings.TrimSpace(string(raw)), nil
}

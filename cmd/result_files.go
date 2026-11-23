package cmd

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/spf13/pflag"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
)

// The small input files a Result holds whole, so --from-result needs none of them passed again
// (inference-sim/inference-sim#2467, DEV-7, SIM-10): the router files the deployment names, and each
// configuration file an option names. A trace or a corpus is bulk data: its option records its
// sha256 only, and a reproduction takes it from the command line again (Principle 6).

// bulkFileOptions are the file options whose file is bulk data, held by sha256 only.
var bulkFileOptions = map[string]bool{"trace-header": true, "trace-data": true, "corpus": true}

// resultOptionFiles are the configuration files the command line named, by option name,
// recorded when the command starts (snapshotResultFlags). Nil before a command runs.
var resultOptionFiles []result.File

// optionFiles reads each non-bulk file option fs set.
func optionFiles(fs *pflag.FlagSet) ([]result.File, error) {
	var out []result.File
	var errs []string
	fs.Visit(func(f *pflag.Flag) {
		if !optionFileFlags[f.Name] || bulkFileOptions[f.Name] || f.Value.String() == "" {
			return
		}
		b, err := os.ReadFile(f.Value.String())
		if err != nil {
			errs = append(errs, fmt.Sprintf("--%s %s: %v", f.Name, f.Value.String(), err))
			return
		}
		out = append(out, fileOf(f.Name, b))
	})
	if len(errs) > 0 {
		return nil, fmt.Errorf("recording the run's input files: %s", strings.Join(errs, "; "))
	}
	return out, nil
}

// routerFiles reads the files the deployment's router names relative to the scenario directory
// dir: its endpoint-picker configuration, its manifests and a --config-file EPP flag. A path
// stated as absolute is a fact of the host, not of the input, and is not held.
func routerFiles(dep *deployment.Deployment, dir string) ([]result.File, error) {
	if dep == nil || dep.Router == nil {
		return nil, nil
	}
	r := dep.Router
	paths := append([]string{r.EndpointPickerConfig}, r.Manifests...)
	for _, f := range r.EPPFlags {
		if p, ok := strings.CutPrefix(f, "--config-file="); ok {
			paths = append(paths, p)
		}
	}
	var out []result.File
	seen := map[string]bool{}
	for _, p := range paths {
		if p == "" || filepath.IsAbs(p) || seen[p] {
			continue
		}
		seen[p] = true
		b, err := os.ReadFile(filepath.Join(dir, p))
		if err != nil {
			return nil, fmt.Errorf("recording the router file %s: %w", p, err)
		}
		out = append(out, fileOf(p, b))
	}
	return out, nil
}

// resultFiles are the files a Result holds: the router's and the options', sorted by path.
func resultFiles(dep *deployment.Deployment) ([]result.File, error) {
	out, err := routerFiles(dep, kernelScenarioDir)
	if err != nil {
		return nil, err
	}
	out = append(out, resultOptionFiles...)
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// fileOf is the record of one file.
func fileOf(path string, content []byte) result.File {
	sum := sha256.Sum256(content)
	return result.File{Path: path, SHA256: hex.EncodeToString(sum[:]), Content: string(content)}
}

// heldFile is the file a Result holds at path, if it holds one.
func heldFile(doc *result.Result, path string) (result.File, bool) {
	for _, f := range doc.Input.Files {
		if f.Path == path {
			return f, true
		}
	}
	return result.File{}, false
}

package cmd

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"runtime/debug"
	"sort"

	"github.com/spf13/cobra"

	referenceconfigs "github.com/llm-d/llm-d-router/reference-configs"
)

// The reference EndpointPicker configurations the pinned router fork ships, with their index
// (policies.json). The files are read from the fork's own embed, so the command holds no list
// of its own and never reads the network or the disk.

const routerModule = "github.com/llm-d/llm-d-router"

var (
	routerConfigsJSON bool
	routerConfigsFile string
)

var routerConfigsCmd = &cobra.Command{
	Use:   "router-configs",
	Short: "Print the reference router configurations of the router fork this build pins",
	Long: "Print the reference EndpointPicker configurations embedded in the llm-d router fork this " +
		"build pins, with the index (policies.json) that describes them. With --json the output is " +
		"{\"router_module\", \"version\", \"index\": <policies.json as published>, \"files\": " +
		"{<name>: {\"sha256\", \"content\"}}}. With --file NAME the bytes of that one file are " +
		"written as they are; a name the fork does not ship is an error. Without either, a summary " +
		"of each policy and the files it names.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, _ []string) error {
		if routerConfigsJSON && routerConfigsFile != "" {
			return fmt.Errorf("router-configs: give at most one of --json and --file")
		}
		return printRouterConfigs(cmd.OutOrStdout(), referenceconfigs.FS, routerVersion(), routerConfigsJSON, routerConfigsFile)
	},
}

// routerVersion is the version of the router module this build was built with: its
// replacement's when replaced by a module version, else its own; "unknown" when the build
// information lists none (a test binary).
func routerVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	for _, dep := range info.Deps {
		if dep.Path == routerModule {
			if v := depVersion(dep); v != "" {
				return v
			}
		}
	}
	return "unknown"
}

type routerConfigFile struct {
	SHA256  string `json:"sha256"`
	Content string `json:"content"`
}

type routerConfigsIndex struct {
	Policies []struct {
		Name  string            `json:"name"`
		Files map[string]string `json:"files"`
	} `json:"policies"`
}

func printRouterConfigs(w io.Writer, fsys fs.FS, version string, asJSON bool, file string) error {
	if file != "" {
		b, err := fs.ReadFile(fsys, file)
		if err != nil {
			return fmt.Errorf("router-configs: the router fork ships no file %q", file)
		}
		_, err = w.Write(b)
		return err
	}
	rawIndex, err := fs.ReadFile(fsys, "policies.json")
	if err != nil {
		return fmt.Errorf("router-configs: the router fork's index: %w", err)
	}
	var idx routerConfigsIndex
	if err := json.Unmarshal(rawIndex, &idx); err != nil {
		return fmt.Errorf("router-configs: the router fork's index: %w", err)
	}
	// Every file the index names must be present: an index that points at nothing is a broken pin.
	for _, p := range idx.Policies {
		for _, name := range p.Files {
			if name == "" {
				continue
			}
			if _, err := fs.Stat(fsys, name); err != nil {
				return fmt.Errorf("router-configs: the index names %q for policy %q and the fork does not ship it", name, p.Name)
			}
		}
	}
	entries, err := fs.ReadDir(fsys, ".")
	if err != nil {
		return err
	}
	files := map[string]routerConfigFile{}
	for _, e := range entries {
		if e.IsDir() || e.Name() == "policies.json" {
			continue
		}
		b, err := fs.ReadFile(fsys, e.Name())
		if err != nil {
			return err
		}
		sum := sha256.Sum256(b)
		files[e.Name()] = routerConfigFile{SHA256: hex.EncodeToString(sum[:]), Content: string(b)}
	}
	if asJSON {
		return writeJSON(w, struct {
			RouterModule string                      `json:"router_module"`
			Version      string                      `json:"version"`
			Index        json.RawMessage             `json:"index"`
			Files        map[string]routerConfigFile `json:"files"`
		}{routerModule, version, rawIndex, files})
	}
	if _, err := fmt.Fprintf(w, "%s %s\n", routerModule, version); err != nil {
		return err
	}
	for _, p := range idx.Policies {
		kinds := make([]string, 0, len(p.Files))
		for k := range p.Files {
			kinds = append(kinds, k)
		}
		sort.Strings(kinds)
		for _, k := range kinds {
			if p.Files[k] == "" {
				continue
			}
			f := files[p.Files[k]]
			if _, err := fmt.Fprintf(w, "%-24s %-10s %-32s %.12s\n", p.Name, k, p.Files[k], f.SHA256); err != nil {
				return err
			}
		}
	}
	return nil
}

func init() {
	routerConfigsCmd.Flags().BoolVar(&routerConfigsJSON, "json", false, "print JSON")
	routerConfigsCmd.Flags().StringVar(&routerConfigsFile, "file", "", "print the bytes of one reference file, such as default.yaml")
	rootCmd.AddCommand(routerConfigsCmd)
}

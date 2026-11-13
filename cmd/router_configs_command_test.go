package cmd

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io/fs"
	"strings"
	"testing"
	"testing/fstest"

	referenceconfigs "github.com/llm-d/llm-d-router/reference-configs"
)

type routerConfigsOutput struct {
	RouterModule string          `json:"router_module"`
	Version      string          `json:"version"`
	Index        json.RawMessage `json:"index"`
	Files        map[string]struct {
		SHA256  string `json:"sha256"`
		Content string `json:"content"`
	} `json:"files"`
}

// Every file the index names appears in the --json output, with the sha256 of its bytes, and
// the index is carried as published.
func TestRouterConfigsJSONCarriesEveryFileTheIndexNames(t *testing.T) {
	var out bytes.Buffer
	if err := printRouterConfigs(&out, referenceconfigs.FS, "vX", true, ""); err != nil {
		t.Fatal(err)
	}
	var got routerConfigsOutput
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.RouterModule != "github.com/llm-d/llm-d-router" || got.Version != "vX" {
		t.Fatalf("module %q version %q", got.RouterModule, got.Version)
	}
	raw, err := fs.ReadFile(referenceconfigs.FS, "policies.json")
	if err != nil {
		t.Fatal(err)
	}
	var a, b any
	if err := json.Unmarshal(raw, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(got.Index, &b); err != nil {
		t.Fatal(err)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Fatal("the index in the output is not policies.json")
	}
	var idx routerConfigsIndex
	if err := json.Unmarshal(raw, &idx); err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, p := range idx.Policies {
		for _, name := range p.Files {
			if name == "" {
				continue
			}
			f, ok := got.Files[name]
			if !ok {
				t.Fatalf("policy %s names %s; the output has no such file", p.Name, name)
			}
			want, _ := fs.ReadFile(referenceconfigs.FS, name)
			sum := sha256.Sum256(want)
			if f.SHA256 != hex.EncodeToString(sum[:]) || f.Content != string(want) {
				t.Fatalf("%s: sha256/content differ from the embedded bytes", name)
			}
			seen++
		}
	}
	if seen == 0 {
		t.Fatal("the index names no file")
	}
}

// --file writes the exact bytes of one file and nothing else; an unknown name is an error.
func TestRouterConfigsFileReturnsExactBytesAndRefusesUnknown(t *testing.T) {
	want, err := fs.ReadFile(referenceconfigs.FS, "default.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := printRouterConfigs(&out, referenceconfigs.FS, "vX", false, "default.yaml"); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out.Bytes(), want) {
		t.Fatal("--file default.yaml did not write the file's exact bytes")
	}
	out.Reset()
	if err := printRouterConfigs(&out, referenceconfigs.FS, "vX", false, "no-such.yaml"); err == nil || out.Len() != 0 {
		t.Fatalf("unknown file: err %v, wrote %d bytes", err, out.Len())
	}
	if err := printRouterConfigs(&out, referenceconfigs.FS, "vX", true, "../policies.json"); err == nil {
		t.Fatal("a path outside the embed was accepted")
	}
}

// An index that names a file the fork does not ship is refused, not printed with a gap.
func TestRouterConfigsRefusesAnIndexNamingAMissingFile(t *testing.T) {
	fsys := fstest.MapFS{
		"policies.json": {Data: []byte(`{"policies":[{"name":"p","files":{"colocated":"p.yaml","pd":null}}]}`)},
	}
	var out bytes.Buffer
	err := printRouterConfigs(&out, fsys, "vX", true, "")
	if err == nil || !strings.Contains(err.Error(), "p.yaml") {
		t.Fatalf("err = %v", err)
	}
}

// The default summary names each policy with the file it uses.
func TestRouterConfigsSummaryNamesEachFile(t *testing.T) {
	var out bytes.Buffer
	if err := printRouterConfigs(&out, referenceconfigs.FS, "vX", false, ""); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "default.yaml") || !strings.Contains(out.String(), "vX") {
		t.Fatalf("summary:\n%s", out.String())
	}
}

// Files in subdirectories (the guide configurations) are keyed by their relative path and
// --file takes that path.
func TestRouterConfigsCarriesFilesInSubdirectories(t *testing.T) {
	fsys := fstest.MapFS{
		"policies.json": {Data: []byte(`{"policies":[{"name":"g","files":{"colocated":"guides/g.yaml"}}]}`)},
		"guides/g.yaml": {Data: []byte("a: 1\n")},
		"top.yaml":      {Data: []byte("b: 2\n")},
	}
	var out bytes.Buffer
	if err := printRouterConfigs(&out, fsys, "vX", true, ""); err != nil {
		t.Fatal(err)
	}
	var got routerConfigsOutput
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Files["guides/g.yaml"].Content != "a: 1\n" || got.Files["top.yaml"].SHA256 == "" || len(got.Files) != 2 {
		t.Fatalf("files: %v", got.Files)
	}
	out.Reset()
	if err := printRouterConfigs(&out, fsys, "vX", false, "guides/g.yaml"); err != nil || out.String() != "a: 1\n" {
		t.Fatalf("--file: %v %q", err, out.String())
	}
}

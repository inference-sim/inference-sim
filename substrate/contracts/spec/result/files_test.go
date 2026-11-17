package result

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func fileOf(path, content string) File {
	sum := sha256.Sum256([]byte(content))
	return File{Path: path, SHA256: hex.EncodeToString(sum[:]), Content: content}
}

// A Result may hold the small files its run read, each once, sorted by path, with the sha256
// of its content; each departure is refused naming the field, and the closest valid list passes.
func TestInputFiles(t *testing.T) {
	good := []File{fileOf("epp.yaml", "kind: EndpointPickerConfig\n"), fileOf("workload-spec", "version: \"2\"\n")}
	r := valid()
	r.Input.Files = good
	if p := r.Validate(); !p.OK() {
		t.Fatalf("two well-formed files: %v", p)
	}
	tampered := fileOf("epp.yaml", "kind: EndpointPickerConfig\n")
	tampered.Content += "# edited\n"
	for name, c := range map[string]struct {
		files []File
		want  string
	}{
		"a content that is not its sha256": {[]File{tampered}, "input.files[0].sha256"},
		"an empty path":                    {[]File{fileOf("", "x")}, "input.files[0].path"},
		"out of order":                     {[]File{good[1], good[0]}, "input.files[1].path"},
		"listed twice":                     {[]File{good[0], good[0]}, "input.files[1].path"},
	} {
		r := valid()
		r.Input.Files = c.files
		p := r.Validate()
		if p.OK() || !strings.Contains(p.Error(), c.want) {
			t.Errorf("%s: want a problem at %s, got %v", name, c.want, p)
		}
	}
}

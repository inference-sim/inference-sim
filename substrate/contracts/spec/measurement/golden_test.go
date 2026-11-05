package measurement

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

func decodeStrict(t *testing.T, b []byte, dst any) {
	t.Helper()
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	if err := dec.Decode(dst); err != nil {
		t.Fatal(err)
	}
}

func canonical() *MeasuredRun {
	r := exactRun()
	a := Assess(r, true)
	r.Verdict, r.Resolution = a.Verdict, a.Resolution
	return r
}

// TestGoldenRoundTrip pins the wire form: the fixture marshals to the committed bytes, and
// those bytes decode, strictly, back to the fixture and re-marshal to themselves.
func TestGoldenRoundTrip(t *testing.T) {
	cases := map[string]any{"run.yaml": canonical(), "sources.yaml": permitting()}
	for file, want := range cases {
		t.Run(file, func(t *testing.T) {
			path := filepath.Join("testdata", file)
			golden, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			b, err := Marshal(want)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(b, golden) {
				t.Fatalf("%s is not what the fixture marshals to:\n%s", path, b)
			}
			got := reflect.New(reflect.TypeOf(want).Elem()).Interface()
			decodeStrict(t, golden, got)
			// A pool's engine block is held as a YAML node, which records the line it was read
			// at; compare what the run holds, with each block compared by its content.
			if !reflect.DeepEqual(withoutEngineBlocks(got), withoutEngineBlocks(want)) || !sameEngineBlocks(got, want) {
				t.Fatalf("%s does not decode to the fixture", path)
			}
			again, err := Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(again, golden) {
				t.Fatal("a decoded run does not re-marshal to the same bytes")
			}
		})
	}
	var r MeasuredRun
	golden, _ := os.ReadFile(filepath.Join("testdata", "run.yaml"))
	decodeStrict(t, golden, &r)
	if p := r.ValidateWithSources(permitting()); !p.OK() {
		t.Fatalf("the golden run does not validate:\n%s", p.Error())
	}
}

func TestUnknownFieldIsAnError(t *testing.T) {
	golden, err := os.ReadFile(filepath.Join("testdata", "run.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	bad := bytes.Replace(golden, []byte("schema_version:"), []byte("verdict_override: EXACT\nschema_version:"), 1)
	var r MeasuredRun
	dec := yaml.NewDecoder(bytes.NewReader(bad))
	dec.KnownFields(true)
	if err := dec.Decode(&r); err == nil {
		t.Fatal("an unknown field decoded silently")
	}
}

// withoutEngineBlocks is a copy of a measured run with its deployment's engine blocks cleared,
// for a comparison that takes the blocks separately (sameEngineBlocks). Other documents pass
// through.
func withoutEngineBlocks(v any) any {
	r, ok := v.(*MeasuredRun)
	if !ok || r.Deployment == nil {
		return v
	}
	c := *r
	d := *r.Deployment
	d.Pools = append([]deployment.Pool(nil), d.Pools...)
	for i := range d.Pools {
		d.Pools[i].Engine = deployment.EngineBlock{}
	}
	c.Deployment = &d
	return &c
}

// sameEngineBlocks reports whether two measured runs' pools hold the same engine blocks.
func sameEngineBlocks(a, b any) bool {
	ra, okA := a.(*MeasuredRun)
	rb, okB := b.(*MeasuredRun)
	if !okA || !okB || ra.Deployment == nil || rb.Deployment == nil {
		return true
	}
	if len(ra.Deployment.Pools) != len(rb.Deployment.Pools) {
		return false
	}
	for i := range ra.Deployment.Pools {
		if !ra.Deployment.Pools[i].Engine.Equal(rb.Deployment.Pools[i].Engine) {
			return false
		}
	}
	return true
}

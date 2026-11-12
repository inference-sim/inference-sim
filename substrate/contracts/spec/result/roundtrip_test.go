package result

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// Write-then-read of every field a writer can emit (SUB-12, SUB-19, X13, issue 28). The
// Result is filled from its own type by reflection, so a field added later is written and read
// back without anyone listing it. Each leaf gets a distinct non-zero value, so omitempty drops
// nothing and a field read into another's place cannot hide; a float field that opts in to
// +infinity holds it.

// filler sets every field of a value from its type.
type filler struct {
	n       int
	floats  []string // the float fields it set, by path
	infs    []string // those it set to +Inf
	skipped []string // fields it could not set, with the reason
}

func (f *filler) next() int { f.n++; return f.n }

func (f *filler) fill(v reflect.Value, path string, inf bool, depth int) {
	if depth > 14 {
		return
	}
	t := v.Type()
	// A type with its own MarshalYAML writes something the reflection filler cannot predict,
	// so a test supplies its value (customValues). A type that only has an UnmarshalYAML, to
	// restore strict key checking, encodes as the plain struct it is and is filled like one.
	if t.Implements(yamlMarshaler) || reflect.PointerTo(t).Implements(yamlMarshaler) {
		if mk, ok := customValues[t]; ok {
			v.Set(reflect.ValueOf(mk()))
			return
		}
		f.skipped = append(f.skipped, path+" ("+t.String()+": its own MarshalYAML, no value supplied)")
		return
	}
	switch t.Kind() {
	case reflect.Pointer:
		p := reflect.New(t.Elem())
		f.fill(p.Elem(), path, inf, depth+1)
		v.Set(p)
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			sf := t.Field(i)
			if sf.PkgPath != "" || sf.Tag.Get("yaml") == "-" {
				continue
			}
			f.fill(v.Field(i), path+"."+sf.Name, validate.AllowsInfinity(sf), depth+1)
		}
	case reflect.Slice:
		s := reflect.MakeSlice(t, 1, 1)
		f.fill(s.Index(0), path+"[0]", inf, depth+1)
		v.Set(s)
	case reflect.Map:
		m := reflect.MakeMap(t)
		k := reflect.New(t.Key()).Elem()
		f.fill(k, path+"{key}", false, depth+1)
		e := reflect.New(t.Elem()).Elem()
		f.fill(e, path+"{}", inf, depth+1)
		m.SetMapIndex(k, e)
		v.Set(m)
	case reflect.Interface:
		v.Set(reflect.ValueOf(fmt.Sprintf("v%d", f.next())))
	case reflect.String:
		v.SetString(fmt.Sprintf("s%d", f.next()))
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(int64(f.next()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		v.SetUint(uint64(f.next()))
	case reflect.Float32, reflect.Float64:
		f.floats = append(f.floats, path)
		if inf {
			v.SetFloat(math.Inf(1))
			f.infs = append(f.infs, path)
			return
		}
		v.SetFloat(float64(f.next()) + 0.5)
	}
}

var (
	yamlMarshaler = reflect.TypeOf((*yaml.Marshaler)(nil)).Elem()
	// customValues gives a type with its own MarshalYAML a value that round-trips, where the
	// reflection filler cannot build one.
	customValues = map[reflect.Type]func() any{
		reflect.TypeOf(workload.ContextLimit(0)): func() any { return workload.ContextLimit(4096) },
	}
)

func filled(t *testing.T) (*Result, *filler) {
	t.Helper()
	r := &Result{}
	f := &filler{}
	f.fill(reflect.ValueOf(r).Elem(), "Result", false, 0)
	return r, f
}

func TestEveryFieldAWriterCanEmitRoundTripsInBothEncodings(t *testing.T) {
	r, f := filled(t)
	if len(f.floats) < 60 {
		t.Fatalf("the filler reached %d float fields; the Result has about 76. It is not walking the type", len(f.floats))
	}
	for _, f := range []Format{FormatYAML, FormatJSON} {
		t.Run(string(f), func(t *testing.T) {
			b, err := Marshal(r, f)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			var back Result
			if err := Unmarshal(b, &back); err != nil {
				t.Fatalf("the strict loader refused what Marshal wrote: %v\n%.600s", err, b)
			}
			// An engine block holds a parsed node with read positions, which a second read
			// cannot reproduce; the contracts package tests that block's round trip. Compare
			// the rest with it cleared, on copies.
			a, z := *r, back
			a.Input.Deployment.Pools = withoutEngine(r.Input.Deployment.Pools)
			z.Input.Deployment.Pools = withoutEngine(back.Input.Deployment.Pools)
			if !reflect.DeepEqual(a, z) {
				t.Errorf("the document read back differs from the one written:\n%s", diffFields(reflect.ValueOf(a), reflect.ValueOf(z), "Result"))
			}
			again, err := Marshal(&back, f)
			if err != nil || string(again) != string(b) {
				t.Errorf("writing the read-back document gave other bytes (err %v)", err)
			}
		})
	}
}

func withoutEngine(pools []deployment.Pool) []deployment.Pool {
	out := append([]deployment.Pool(nil), pools...)
	for i := range out {
		out[i].Engine = deployment.EngineBlock{}
	}
	return out
}

// diffFields names the first fields that differ, for a failure message.
func diffFields(a, b reflect.Value, path string) string {
	var out []string
	var walk func(a, b reflect.Value, path string)
	walk = func(a, b reflect.Value, path string) {
		if len(out) >= 6 {
			return
		}
		if reflect.DeepEqual(a.Interface(), b.Interface()) {
			return
		}
		switch a.Kind() {
		case reflect.Struct:
			for i := 0; i < a.NumField(); i++ {
				if a.Type().Field(i).PkgPath == "" {
					walk(a.Field(i), b.Field(i), path+"."+a.Type().Field(i).Name)
				}
			}
		case reflect.Pointer:
			if a.IsNil() || b.IsNil() {
				out = append(out, fmt.Sprintf("%s: %v against %v", path, a.Interface(), b.Interface()))
				return
			}
			walk(a.Elem(), b.Elem(), path)
		default:
			out = append(out, fmt.Sprintf("%s: %v against %v", path, a.Interface(), b.Interface()))
		}
	}
	walk(a, b, path)
	return strings.Join(out, "\n")
}

// The fields that hold infinity are exactly the ones a test names. A field that opts in
// without being added here fails this test; so does a name that no longer opts in.
func TestTheFieldsThatHoldInfinityAreNamed(t *testing.T) {
	want := []string{"ReplayDefinition.GracePeriodS"}
	if got := validate.InfinityFields(reflect.TypeOf(Result{})); !reflect.DeepEqual(got, want) {
		t.Errorf("fields that opt in to +infinity: %v; want %v", got, want)
	}
	// The filler puts +Inf in every opted-in field it reaches, and in no other. The Result
	// holds that field in two places, the scenario as given and the workload the run used.
	_, f := filled(t)
	sort.Strings(f.infs)
	if len(f.infs) != 2 {
		t.Fatalf("the round trip put +Inf in %v; want the scenario's and the frame's grace_period_s", f.infs)
	}
	for _, p := range f.infs {
		if !strings.HasSuffix(p, ".ReplayDefinition.GracePeriodS") {
			t.Errorf("+Inf was put in %s, which does not opt in", p)
		}
	}
}

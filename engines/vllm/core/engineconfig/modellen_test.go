// Moved from inference-sim/blis-schemas/spec/deployment/modellen_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package engineconfig

import (
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// max_model_len has three states, and each spelling the engine's flag accepts must land
// on the right one: auto and -1 both mean fit-to-memory, which is not the same request
// as leaving the field out.
func TestMaxModelLenDecodes(t *testing.T) {
	for _, c := range []struct {
		in        string
		want      MaxModelLen
		auto      bool
		stated    int
		statedSet bool
	}{
		{"{}", 0, false, 0, false},
		{"{max_model_len: auto}", MaxModelLenAuto, true, 0, false},
		{"{max_model_len: AUTO}", MaxModelLenAuto, true, 0, false},
		{"{max_model_len: -1}", MaxModelLenAuto, true, 0, false},
		{"{max_model_len: 131072}", 131072, false, 131072, true},
		{"{max_model_len: 128k}", 128000, false, 128000, true},
		{"{max_model_len: 128K}", 131072, false, 131072, true},
		{"{max_model_len: 25.6k}", 25600, false, 25600, true},
		{"{max_model_len: 1M}", 1 << 20, false, 1 << 20, true},
		{"{max_model_len: 131_072}", 131072, false, 131072, true},
		{"{max_model_len: '-1'}", MaxModelLenAuto, true, 0, false},
	} {
		var e Engine
		if err := yaml.Unmarshal([]byte(c.in), &e); err != nil {
			t.Fatalf("%s: %v", c.in, err)
		}
		n, ok := e.MaxModelLen.Stated()
		if e.MaxModelLen != c.want || e.MaxModelLen.Auto() != c.auto || n != c.stated || ok != c.statedSet {
			t.Errorf("%s: got %d (auto %v, stated %d %v)", c.in, e.MaxModelLen, e.MaxModelLen.Auto(), n, ok)
		}
	}
	for _, bad := range []string{"{max_model_len: atuo}", "{max_model_len: 1.5}",
		"{max_model_len: [1]}", "{max_model_len: 1.5K}", "{max_model_len: 12kb}",
		"{max_model_len: 1__0}", "{max_model_len: _10}", "{max_model_len: 10_}"} {
		var e Engine
		if err := yaml.Unmarshal([]byte(bad), &e); err == nil || !strings.Contains(err.Error(), "max_model_len") {
			t.Errorf("%s: want a max_model_len error, got %v (value %d)", bad, err, e.MaxModelLen)
		}
	}
}

func TestMaxModelLenRoundTrips(t *testing.T) {
	out, err := yaml.Marshal(Engine{MaxModelLen: MaxModelLenAuto})
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(out)) != "max_model_len: auto" {
		t.Errorf("auto marshals as %q", out)
	}
	out, _ = yaml.Marshal(Engine{MaxModelLen: 4096})
	if strings.TrimSpace(string(out)) != "max_model_len: 4096" {
		t.Errorf("4096 marshals as %q", out)
	}
}

func TestMaxModelLenValidation(t *testing.T) {
	for _, c := range []struct {
		m    MaxModelLen
		want bool
	}{{0, true}, {MaxModelLenAuto, true}, {8192, true}, {-2, false}} {
		p := engineOnly(Engine{MaxModelLen: c.m}).Validate()
		if p.OK() != c.want {
			t.Errorf("max_model_len %d: ok=%v, want %v\n%s", c.m, p.OK(), c.want, p.Error())
		}
	}
}

// Properties over many values rather than a few: every positive count and auto survive a
// marshal and unmarshal unchanged; a decimal suffix multiplies by a power of 1000 and a
// binary one by a power of 1024, as the engine's parser does; and every spelling of zero
// is refused rather than read as an omitted field.
func TestMaxModelLenProperties(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	decode := func(v string) (MaxModelLen, error) {
		var e Engine
		err := yaml.Unmarshal([]byte("max_model_len: "+v), &e)
		return e.MaxModelLen, err
	}
	for i := 0; i < 2000; i++ {
		m := MaxModelLen(1 + r.Intn(1<<30))
		if i%10 == 0 {
			m = MaxModelLenAuto
		}
		out, err := yaml.Marshal(Engine{MaxModelLen: m})
		if err != nil {
			t.Fatal(err)
		}
		var back Engine
		if err := yaml.Unmarshal(out, &back); err != nil || back.MaxModelLen != m {
			t.Fatalf("%d round-tripped through %q to %d (%v)", m, out, back.MaxModelLen, err)
		}
		n := int64(1 + r.Intn(1<<20))
		for suffix, mult := range map[string]int64{"k": 1000, "m": 1000000, "K": 1 << 10, "M": 1 << 20} {
			want := n * mult // computed in int64, so it is exact on every platform
			got, err := decode(strconv.FormatInt(n, 10) + suffix)
			if want > math.MaxInt {
				if err == nil {
					t.Fatalf("%d%s decoded to %d on a platform whose int cannot hold %d", n, suffix, got, want)
				}
				continue
			}
			if err != nil || int64(got) != want {
				t.Fatalf("%d%s decoded to %d (%v), want %d", n, suffix, got, err, want)
			}
		}
	}
	for _, zero := range []string{"0", "-0", "0k", "0K", "0.0004k", "'0'"} {
		if m, err := decode(zero); err == nil {
			t.Errorf("%s decoded to %d; a stated zero must be refused", zero, m)
		}
	}
	for _, huge := range []string{"9999999999999999999", "99999999999999T", "16777216T", "99999999999t"} {
		if m, err := decode(huge); err == nil {
			t.Errorf("%s decoded to %d; it cannot be held and must be refused", huge, m)
		}
	}
	// A YAML null reads as the field left out, which is what writing nothing means.
	if m, err := decode("~"); err != nil || m != 0 {
		t.Errorf("null decoded to %d (%v), want omitted", m, err)
	}
}

package hardware

import (
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// h200 mirrors the committed catalog entry, so a drift between this repository's
// expectations and the catalog's contents shows up as a test failure rather than a
// silently wrong estimate.
func h200() *Chip {
	return &Chip{
		Name: "h200", Provenance: vocab.ProvenanceVendorSpec,
		BF16Peak: 989.5, FP8Peak: 1979.0, MemoryBandwidthTBs: 4.8,
		MemoryGiB: 141.0, IntraNodeBwGBps: 450, SMCount: 132, GPUsPerNode: 8,
	}
}

func ib400g() *Fabric {
	return &Fabric{Name: "ib-400g", Provenance: vocab.ProvenanceVendorSpec,
		InterNodeBwGBps: 50, RDMA: true}
}

func TestValidChipAndFabricPass(t *testing.T) {
	if p := h200().Validate(); !p.OK() {
		t.Fatalf("h200 rejected:\n%s", p.Error())
	}
	if p := ib400g().Validate(); !p.OK() {
		t.Fatalf("ib-400g rejected:\n%s", p.Error())
	}
}

// TestIntraToInterRatio checks the quantity a cross-node collective is priced
// against. It is a function of the pairing, which is why neither schema owns it.
func TestIntraToInterRatio(t *testing.T) {
	cases := []struct {
		intra, inter, want float64
	}{
		{450, 50, 9},  // NVLink 4 against InfiniBand NDR
		{450, 25, 18}, // the same chip behind RoCE
		{900, 900, 1}, // multi-node NVLink: no step at the boundary
		{450, 0, 1},   // an unstated fabric must not divide by zero
	}
	for _, c := range cases {
		chip := Chip{IntraNodeBwGBps: c.intra}
		fab := Fabric{InterNodeBwGBps: c.inter}
		if got := IntraToInterRatio(chip, fab); got != c.want {
			t.Errorf("ratio(%v, %v) = %v, want %v", c.intra, c.inter, got, c.want)
		}
	}
}

func TestChipRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Chip)
	}{
		{"no name", func(c *Chip) { c.Name = "" }},
		{"unknown provenance", func(c *Chip) { c.Provenance = "guessed" }},
		{"zero bf16 peak", func(c *Chip) { c.BF16Peak = 0 }},
		{"zero bandwidth", func(c *Chip) { c.MemoryBandwidthTBs = 0 }},
		{"zero memory", func(c *Chip) { c.MemoryGiB = 0 }},
		{"zero intra-node bandwidth", func(c *Chip) { c.IntraNodeBwGBps = 0 }},
		{"negative fp8 peak", func(c *Chip) { c.FP8Peak = -1 }},
		{"fp8 not above bf16", func(c *Chip) { c.FP8Peak = 500 }},
		{"rack not a multiple of node", func(c *Chip) { c.GPUsPerRack = 70 }},
		{"intra-rack bandwidth with no rack", func(c *Chip) { c.IntraRackBwGBps = 900 }},
		// SMCount is required and positive: a missing (zero) count is an omission, not a
		// legitimate value, so it is rejected alongside a negative one.
		{"missing SMCount", func(c *Chip) { c.SMCount = 0 }},
		{"negative SMCount", func(c *Chip) { c.SMCount = -1 }},
		// A non-finite datasheet figure must be rejected before its magnitude check,
		// since NaN compares false to every bound and would otherwise slip through.
		{"NaN bandwidth", func(c *Chip) { c.MemoryBandwidthTBs = math.NaN() }},
		{"Inf bf16 peak", func(c *Chip) { c.BF16Peak = math.Inf(1) }},
		// A NaN intra-rack bandwidth must not pass as "no tier": NaN > 0 is false, so the
		// presence guard alone would silently accept it.
		{"NaN intra-rack bandwidth", func(c *Chip) { c.IntraRackBwGBps = math.NaN() }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := h200()
			tc.mutate(c)
			if p := c.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// TestNonFiniteFieldReportsOnce pins that a non-finite field is reported exactly once, at
// the field that is wrong, rather than also tripping a dependent check. A +Inf BF16 peak
// must not make the FP8-vs-BF16 transcription check complain about the FP8 field, and a
// -Inf must not also fire the positivity check — both would misdirect the fix.
func TestNonFiniteFieldReportsOnce(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Chip)
	}{
		{"+Inf bf16 peak", func(c *Chip) { c.BF16Peak = math.Inf(1) }},
		{"-Inf bf16 peak", func(c *Chip) { c.BF16Peak = math.Inf(-1) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := h200()
			tc.mutate(c)
			errs := c.Validate().Errors()
			if len(errs) != 1 {
				t.Fatalf("want exactly one error, got %d:\n%s", len(errs), c.Validate().Error())
			}
			if errs[0].Path != "TFlopsPeak" {
				t.Errorf("the single error should be at TFlopsPeak, got %q", errs[0].Path)
			}
		})
	}
}

func TestFabricAndDeviceRejects(t *testing.T) {
	bad := []*Fabric{
		{Provenance: vocab.ProvenanceVendorSpec, InterNodeBwGBps: 50}, // no name
		{Name: "x", Provenance: "invented", InterNodeBwGBps: 50},
		{Name: "x", Provenance: vocab.ProvenanceVendorSpec}, // zero bandwidth
	}
	for i, f := range bad {
		if p := f.Validate(); p.OK() {
			t.Errorf("fabric case %d: expected rejection", i)
		}
	}
	// A non-finite inter-node bandwidth must be rejected: NaN <= 0 is false, so the
	// positivity check alone would accept it.
	if p := (&Fabric{Name: "x", Provenance: vocab.ProvenanceVendorSpec,
		InterNodeBwGBps: math.NaN()}).Validate(); p.OK() {
		t.Error("fabric with a NaN inter-node bandwidth: expected rejection")
	}
	good := &StorageDevice{Name: "nvme_gen4", ReadBandwidthMBs: 7000,
		WriteBandwidthMBs: 5000, BaseLatencyUs: 80}
	if p := good.Validate(); !p.OK() {
		t.Fatalf("nvme_gen4 rejected:\n%s", p.Error())
	}
	badDevices := []*StorageDevice{
		{ReadBandwidthMBs: 7000, WriteBandwidthMBs: 5000, BaseLatencyUs: 80},
		{Name: "d", WriteBandwidthMBs: 5000, BaseLatencyUs: 80},
		{Name: "d", ReadBandwidthMBs: 7000, BaseLatencyUs: 80},
		{Name: "d", ReadBandwidthMBs: 7000, WriteBandwidthMBs: 5000},                           // zero latency
		{Name: "d", ReadBandwidthMBs: math.Inf(1), WriteBandwidthMBs: 5000, BaseLatencyUs: 80}, // non-finite read
	}
	for i, d := range badDevices {
		if p := d.Validate(); p.OK() {
			t.Errorf("device case %d: expected rejection", i)
		}
	}
}

// cleanChipYAML is a minimal well-formed chip document, used as the base a decode test
// mutates one key at a time so each case isolates the key it adds. It is kept consistent
// with cleanChip in cmd/validate-catalog/main_test.go (the end-to-end counterpart); the
// two live in different packages so they cannot share one literal, and both must gain any
// field this schema makes newly required — SMCount was the last such change.
const cleanChipYAML = `Provenance: vendor_spec
TFlopsPeak: 989.5
TFlopsFP8: 1979.0
BwPeakTBs: 3.35
MemoryGiB: 80.0
IntraNodeBwGBps: 450
SMCount: 132
`

// TestChipCommentConventionIsNarrow pins the dimensionless-field ban at the decode
// layer: only "_comment"-prefixed keys are accepted as the catalog's provenance prose;
// every other key — including any other underscore-prefixed one — is a data field and
// must fail strict decoding. This is the gap the deleted blis-catalog validate_catalog.py
// closed: hardware/ carries only dimensioned physical quantities, so a fitted or
// dimensionless factor must not slip in behind an underscore.
func TestChipCommentConventionIsNarrow(t *testing.T) {
	accepted := []struct {
		name, extra string
	}{
		{"_comment", `_comment: "prose"` + "\n"},
		{"_comment_sm", `_comment_sm: "SM count source: https://example.com/datasheet"` + "\n"},
		{"_comment_interconnect", `_comment_interconnect: "nominal, not measured"` + "\n"},
	}
	for _, c := range accepted {
		t.Run("accept "+c.name, func(t *testing.T) {
			var chip Chip
			if err := yaml.Unmarshal([]byte(cleanChipYAML+c.extra), &chip); err != nil {
				t.Fatalf("a %s key should be accepted as catalog prose, got: %v", c.name, err)
			}
			if chip.SMCount != 132 {
				t.Errorf("comment stripping dropped a real field: SMCount = %d, want 132", chip.SMCount)
			}
		})
	}
	rejected := []struct {
		name, extra string
	}{
		// A bare dimensionless factor is an unknown field, as it always was.
		{"bare dimensionless field", `mfu: 0.85` + "\n"},
		// The gap this change closes: a dimensionless factor must not hide under an
		// underscore. "_mfu" is not a comment, so it fails like any unknown field.
		{"underscore-hidden dimensionless field", `_mfu: 0.85` + "\n"},
		// A misspelled real field was and stays an unknown field.
		{"misspelled real field", `TFlopsPeakk: 1.0` + "\n"},
		// The separator matters: a bare "_comment" prefix without the "_" boundary would
		// swallow these, so a dimensionless factor could hide as "_commentmfu". The
		// predicate requires exactly "_comment" or a "_comment_" suffix form, so these
		// fail like any unknown field.
		{"_comment prefix without separator (fitted factor)", `_commentmfu: 0.85` + "\n"},
		{"_comment-ish word", `_commentary: 1.0` + "\n"},
	}
	for _, c := range rejected {
		t.Run("reject "+c.name, func(t *testing.T) {
			var chip Chip
			if err := yaml.Unmarshal([]byte(cleanChipYAML+c.extra), &chip); err == nil {
				t.Fatalf("a %s should be rejected as an unknown field, but decoding succeeded", c.name)
			}
		})
	}
}

// TestCommentNarrowingAppliesToFabricAndDevice pins the dimensionless-field ban on the
// other two types the narrowing touches. All three UnmarshalYAML methods route through
// the one IgnoreCatalogComments predicate, but the networks/ and devices/ namespaces are
// exactly the ones #32 names, so each gets an explicit assertion that an underscore-hidden
// field fails there too — a _comment key still passes, a _mfu does not.
func TestCommentNarrowingAppliesToFabricAndDevice(t *testing.T) {
	t.Run("fabric accepts _comment, rejects _mfu", func(t *testing.T) {
		base := "Provenance: vendor_spec\nInterNodeBwGBps: 50\n"
		var f Fabric
		if err := yaml.Unmarshal([]byte(base+`_comment: "prose"`+"\n"), &f); err != nil {
			t.Fatalf("fabric should accept a _comment key, got: %v", err)
		}
		if err := yaml.Unmarshal([]byte(base+`_mfu: 0.85`+"\n"), &f); err == nil {
			t.Fatal("fabric should reject an underscore-hidden _mfu as an unknown field")
		}
	})
	t.Run("storage device accepts _comment, rejects _mfu", func(t *testing.T) {
		base := "read_bandwidth_mb_s: 7000\nwrite_bandwidth_mb_s: 5000\nbase_latency_us: 80\n"
		var d StorageDevice
		if err := yaml.Unmarshal([]byte(base+`_comment: "prose"`+"\n"), &d); err != nil {
			t.Fatalf("storage device should accept a _comment key, got: %v", err)
		}
		if err := yaml.Unmarshal([]byte(base+`_mfu: 0.85`+"\n"), &d); err == nil {
			t.Fatal("storage device should reject an underscore-hidden _mfu as an unknown field")
		}
	})
}

func TestComputeCapability(t *testing.T) {
	for _, c := range []struct {
		cc    string
		major int
		ok    bool
	}{
		{"9.0", 9, true}, {"10.0", 10, true}, {"10.3", 10, true}, {"8.9", 8, true},
		{"", 0, false}, {"9", 0, false}, {"sm90", 0, false}, {"9.x", 0, false},
		{"0.0", 0, false}, {"9.+1", 0, false}, {"+9.0", 0, false}, {"09.0", 0, false}, {"9.-0", 0, false},
	} {
		chip := &Chip{ComputeCapability: c.cc}
		if m, ok := chip.CapabilityMajor(); m != c.major || ok != c.ok {
			t.Errorf("%q: CapabilityMajor() = %d, %v; want %d, %v", c.cc, m, ok, c.major, c.ok)
		}
	}
}

// A chip's validation agrees with CapabilityMajor: a well-formed capability is accepted and
// a malformed one is reported at its field, never silently treated as unstated.
func TestChipValidatesComputeCapability(t *testing.T) {
	for _, cc := range []string{"", "9.0", "10.3"} {
		c := h200()
		c.ComputeCapability = cc
		if p := c.Validate(); !p.OK() {
			t.Errorf("%q rejected:\n%s", cc, p.Error())
		}
	}
	for _, cc := range []string{"sm90", "9", "+9.0", "09.0", "9.x"} {
		c := h200()
		c.ComputeCapability = cc
		p := c.Validate()
		if p.OK() || !strings.Contains(p.Error(), "ComputeCapability") {
			t.Errorf("%q accepted:\n%s", cc, p.Error())
		}
	}
}

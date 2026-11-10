package hardware

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
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
		{"negative GPUsPerNode", func(c *Chip) { c.GPUsPerNode = -8 }},
		{"negative GPUsPerRack", func(c *Chip) { c.GPUsPerRack = -72 }},
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

// SUB-16: catalog peaks are dense figures, and the ratio between data types is the datasheet's.
// A sparse figure entered as dense doubles a ratio, and the check refuses it.
func TestPeaksAreDenseFigures(t *testing.T) {
	chip := func(bf16, fp8, nvfp4 float64) *Chip {
		return &Chip{Name: "x", Provenance: vocab.ProvenanceVendorSpec, BF16Peak: bf16, FP8Peak: fp8, NVFP4Peak: nvfp4,
			MemoryBandwidthTBs: 1, MemoryGiB: 80, IntraNodeBwGBps: 100, SMCount: 100}
	}
	// The shipped parts: H100 (989.5, 1979), L40S (362.05, 733), B300 (2250, 4500, 14000).
	for name, c := range map[string]*Chip{"H100": chip(989.5, 1979, 0), "L40S": chip(362.05, 733, 0), "B300": chip(2250, 4500, 14000), "A100": chip(312, 0, 0)} {
		if p := c.Validate(); !p.OK() {
			t.Errorf("%s: %s", name, p.Error())
		}
	}
	for name, c := range map[string]struct {
		chip *Chip
		want string
	}{
		"sparse FP8 entered as dense":   {chip(989.5, 3958, 0), "looks like the sparse figure"},
		"sparse NVFP4 entered as dense": {chip(2250, 4500, 30000), "looks like the sparse figure"},
		"NVFP4 below FP8":               {chip(2250, 4500, 4000), "does not exceed FP8"},
		"FP8 below BF16":                {chip(2250, 2000, 0), "does not exceed BF16"},
	} {
		if p := c.chip.Validate(); p.OK() || !strings.Contains(p.Error(), c.want) {
			t.Errorf("%s: %v", name, p)
		}
	}
}

// SUB-18: provenance is recorded per field. A field the map does not name has the file's
// provenance, and that has to be a source of a catalog fact too: a source other than
// datasheet, published or measured:<device> is refused whether the map names it or the file
// leaves it to its Provenance.
func TestFieldProvenance(t *testing.T) {
	chip := func(prov vocab.Provenance, fp map[string]string) *Chip {
		return &Chip{Name: "x", Provenance: prov, BF16Peak: 100, MemoryBandwidthTBs: 1,
			MemoryGiB: 80, IntraNodeBwGBps: 100, SMCount: 100, FieldProvenance: fp}
	}
	vs := vocab.ProvenanceVendorSpec
	for _, ok := range []map[string]string{nil, {}, {"MemoryGiB": "measured:NVIDIA H100"}, {"SMCount": "published", "TFlopsPeak": "datasheet"}} {
		if p := chip(vs, ok).Validate(); !p.OK() {
			t.Errorf("%v refused: %s", ok, p.Error())
		}
	}
	for name, c := range map[string]struct {
		fp   map[string]string
		want string
	}{
		"fitted is a coefficient":        {map[string]string{"MemoryGiB": "fitted"}, "belongs in the coefficient registry"},
		"assumed is a coefficient":       {map[string]string{"SMCount": "assumed"}, "belongs in the coefficient registry"},
		"measured with no device":        {map[string]string{"MemoryGiB": "measured:"}, "names the device"},
		"measured without the colon":     {map[string]string{"MemoryGiB": "measured"}, "names the device"},
		"datasheet naming a device":      {map[string]string{"MemoryGiB": "datasheet:H100"}, "only a measured figure names a device"},
		"a source nobody defined":        {map[string]string{"MemoryGiB": "rumour"}, "is not datasheet"},
		"a field the chip does not have": {map[string]string{"NoSuchFact": "datasheet"}, "names no catalog fact"},
		"the provenance itself":          {map[string]string{"Provenance": "datasheet"}, "names no catalog fact"},
		"a fact the file does not state": {map[string]string{"TFlopsFP8": "datasheet"}, "does not state"},
	} {
		p := chip(vs, c.fp).Validate()
		if p.OK() || !strings.Contains(p.Error(), c.want) {
			t.Errorf("%s: %v", name, p)
		}
	}
	f := &Fabric{Name: "n", Provenance: vocab.ProvenanceVendorSpec, InterNodeBwGBps: 50, FieldProvenance: map[string]string{"InterNodeBwGBps": "fitted"}}
	if p := f.Validate(); p.OK() {
		t.Error("a fitted fabric bandwidth was accepted")
	}
	f.FieldProvenance = map[string]string{"InterNodeBwGBps": "datasheet"}
	if p := f.Validate(); !p.OK() {
		t.Errorf("a datasheet fabric bandwidth was refused: %s", p.Error())
	}
}

// The file-level Provenance is what an omitted field inherits, so it has to be a source. The
// vocabulary still has derived, which says how a figure was obtained and not where it was read:
// a chip that is derived and records nothing used to validate, and every figure in it passed
// as if it had a source. Now each fact it states needs its own.
func TestInheritedProvenanceObeysTheFieldSourceContract(t *testing.T) {
	chip := func(prov vocab.Provenance, fp map[string]string) *Chip {
		return &Chip{Name: "x", Provenance: prov, BF16Peak: 100, MemoryBandwidthTBs: 1,
			MemoryGiB: 80, IntraNodeBwGBps: 100, SMCount: 100, FieldProvenance: fp}
	}
	p := chip(vocab.ProvenanceDerived, nil).Validate()
	if p.OK() {
		t.Fatal("a derived chip that records no field source validated")
	}
	for _, field := range []string{"TFlopsPeak", "BwPeakTBs", "MemoryGiB", "IntraNodeBwGBps", "SMCount"} {
		if !strings.Contains(p.Error(), "FieldProvenance."+field) {
			t.Errorf("the refusal does not name %s: %s", field, p.Error())
		}
	}
	if strings.Contains(p.Error(), "TFlopsFP8") || strings.Contains(p.Error(), "ComputeCapability") {
		t.Errorf("the refusal names a fact the chip does not state: %s", p.Error())
	}
	if !strings.Contains(p.Error(), "not where it was read") {
		t.Errorf("the refusal does not say why derived is not a source: %s", p.Error())
	}
	// One entry short is still refused, naming the one missing.
	all := map[string]string{"TFlopsPeak": "datasheet", "BwPeakTBs": "datasheet", "MemoryGiB": "measured:NVIDIA X",
		"IntraNodeBwGBps": "published", "SMCount": "published"}
	if p := chip(vocab.ProvenanceDerived, all).Validate(); !p.OK() {
		t.Errorf("a derived chip that records every fact's source was refused: %s", p.Error())
	}
	for missing := range all {
		fp := map[string]string{}
		for k, v := range all {
			if k != missing {
				fp[k] = v
			}
		}
		p := chip(vocab.ProvenanceDerived, fp).Validate()
		if p.OK() || !strings.Contains(p.Error(), "FieldProvenance."+missing) {
			t.Errorf("a derived chip that omits %s: %v", missing, p)
		}
	}
	// A fabric reads the same rule.
	f := &Fabric{Name: "n", Provenance: vocab.ProvenanceDerived, InterNodeBwGBps: 50}
	if p := f.Validate(); p.OK() || !strings.Contains(p.Error(), "FieldProvenance.InterNodeBwGBps") {
		t.Errorf("a derived fabric with no field source: %v", p)
	}
	f.FieldProvenance = map[string]string{"InterNodeBwGBps": "datasheet"}
	if p := f.Validate(); !p.OK() {
		t.Errorf("a derived fabric that records its bandwidth's source was refused: %s", p.Error())
	}
	// SourceOf is the one reading of the rule: the record, else the file's word.
	c := chip(vocab.ProvenanceVendorSpec, map[string]string{"MemoryGiB": "measured:NVIDIA X"})
	for field, want := range map[string]string{"MemoryGiB": "measured:NVIDIA X", "SMCount": "datasheet", "TFlopsPeak": "datasheet"} {
		if got, err := c.SourceOf(field); err != nil || got.String() != want {
			t.Errorf("SourceOf(%s) = %v, %v; want %s", field, got, err, want)
		}
	}
	if _, err := c.SourceOf("NoSuchFact"); err == nil {
		t.Error("SourceOf accepted a field the type does not have")
	}
	if _, err := chip(vocab.ProvenanceDerived, nil).SourceOf("SMCount"); err == nil {
		t.Error("SourceOf gave a derived file's omitted field a source")
	}
}

func TestParseFieldSourceRoundTrips(t *testing.T) {
	for _, s := range []string{"datasheet", "published", "measured:NVIDIA B300 SXM6 AC"} {
		f, err := vocab.ParseFieldSource(s)
		if err != nil || f.String() != s {
			t.Errorf("%q -> %+v, %v", s, f, err)
		}
	}
}

// exception is one per-field source a committed chip records beside its file's Provenance.
type exception struct {
	Chip   string // the file's stem under catalog/hardware
	Field  string // the fact's wire name
	Kind   vocab.FieldSourceKind
	Device string // the measured device, for FieldMeasured; empty otherwise
}

func (e exception) wire() string {
	return vocab.FieldSource{Kind: e.Kind, Device: e.Device}.String()
}

// committedExceptions is every exception the committed chips record, exactly. Each is a figure
// its _comment says is not the datasheet's: the driver-reported memory of a named device, or an
// SM count read off one. Adding one to a chip file needs a line here, with the comment that
// supports it; deleting or changing one in a chip file fails the test, so a measurement cannot
// silently fall back to the file's vendor_spec label.
var committedExceptions = []exception{
	{"a100-80", "SMCount", vocab.FieldPublished, ""},
	{"a100-sxm", "SMCount", vocab.FieldPublished, ""},
	{"b200", "MemoryGiB", vocab.FieldMeasured, "NVIDIA B200"},
	{"b200", "SMCount", vocab.FieldPublished, ""},
	{"b300", "MemoryGiB", vocab.FieldMeasured, "NVIDIA B300 SXM6 AC"},
	{"b300", "SMCount", vocab.FieldMeasured, "NVIDIA B300 SXM6 AC"},
	{"gb200-nvl72", "MemoryGiB", vocab.FieldMeasured, "NVIDIA GB200"},
	{"gb200-nvl72", "SMCount", vocab.FieldMeasured, "NVIDIA GB200"},
	{"gb300", "MemoryGiB", vocab.FieldMeasured, "NVIDIA GB300"},
	{"gb300", "SMCount", vocab.FieldMeasured, "NVIDIA GB300"},
	{"h100", "MemoryGiB", vocab.FieldMeasured, "H100 SXM5 80GB HBM3"},
	{"h200", "MemoryGiB", vocab.FieldMeasured, "NVIDIA H200"},
}

// exceptionProblems compares the chips' raw files, by stem, with the list of exceptions they
// must record, in both directions, and returns one problem per difference. For each measured
// figure it also requires that a _comment of the file names the device, so the record and the
// prose cannot drift apart.
func exceptionProblems(files map[string][]byte, want []exception) []string {
	var out []string
	wanted := map[string]exception{}
	for _, e := range want {
		wanted[e.Chip+"/"+e.Field] = e
	}
	stems := make([]string, 0, len(files))
	for stem := range files {
		stems = append(stems, stem)
	}
	sort.Strings(stems)
	seen := map[string]bool{}
	for _, stem := range stems {
		var c Chip
		if err := yaml.Unmarshal(files[stem], &c); err != nil {
			out = append(out, fmt.Sprintf("%s: %v", stem, err))
			continue
		}
		c.Name = stem
		if p := c.Validate(); !p.OK() {
			out = append(out, fmt.Sprintf("%s: %s", stem, p.Error()))
		}
		fields := make([]string, 0, len(c.FieldProvenance))
		for f := range c.FieldProvenance {
			fields = append(fields, f)
		}
		sort.Strings(fields)
		for _, field := range fields {
			key := stem + "/" + field
			seen[key] = true
			w, listed := wanted[key]
			if !listed {
				out = append(out, fmt.Sprintf("%s records %s as %q, which committedExceptions does not list", stem, field, c.FieldProvenance[field]))
				continue
			}
			if got := c.FieldProvenance[field]; got != w.wire() {
				out = append(out, fmt.Sprintf("%s records %s as %q; committedExceptions requires %q", stem, field, got, w.wire()))
			}
			if w.Kind == vocab.FieldMeasured && !commentsMention(files[stem], w.Device) {
				out = append(out, fmt.Sprintf("%s: %s is measured on %q, which no _comment of the file mentions", stem, field, w.Device))
			}
		}
	}
	for _, e := range want {
		if _, ok := files[e.Chip]; !ok {
			out = append(out, fmt.Sprintf("%s: no such chip file; committedExceptions lists %s as %q", e.Chip, e.Field, e.wire()))
		} else if !seen[e.Chip+"/"+e.Field] {
			out = append(out, fmt.Sprintf("%s no longer records %s as %q: the figure falls back to the file's Provenance", e.Chip, e.Field, e.wire()))
		}
	}
	return out
}

func readCatalogFiles(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("../../../catalog", dir, "*.yaml"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no %s files: %v", dir, err)
	}
	out := map[string][]byte{}
	for _, f := range paths {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		out[strings.TrimSuffix(filepath.Base(f), ".yaml")] = raw
	}
	return out
}

// The committed chips record exactly the exceptions committedExceptions lists, no more and no
// fewer, each with the source and device it names.
func TestCommittedChipsRecordTheirExceptions(t *testing.T) {
	files := readCatalogFiles(t, "hardware")
	for _, p := range exceptionProblems(files, committedExceptions) {
		t.Error(p)
	}
	if len(committedExceptions) != 12 {
		t.Errorf("%d exceptions listed; the committed chips record the six driver-reported memory totals and six SM counts", len(committedExceptions))
	}
}

// Deleting any one required exception, or changing its source, fails the check. The test breaks
// each of the committed files in turn and requires the check to name the damage. Before this
// the check counted measured entries, and deleting H100's MemoryGiB left eight of the required
// six standing.
func TestLosingAnExceptionFailsTheCheck(t *testing.T) {
	files := readCatalogFiles(t, "hardware")
	if p := exceptionProblems(files, committedExceptions); len(p) != 0 {
		t.Fatalf("the committed chips are not clean to begin with: %v", p)
	}
	edit := func(stem string, change func(fp map[string]any)) map[string][]byte {
		var doc map[string]any
		if err := yaml.Unmarshal(files[stem], &doc); err != nil {
			t.Fatal(err)
		}
		fp, _ := doc["FieldProvenance"].(map[string]any)
		change(fp)
		raw, err := yaml.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		out := map[string][]byte{}
		for k, v := range files {
			out[k] = v
		}
		out[stem] = raw
		return out
	}
	for _, e := range committedExceptions {
		e := e
		deleted := edit(e.Chip, func(fp map[string]any) { delete(fp, e.Field) })
		if p := exceptionProblems(deleted, committedExceptions); len(p) == 0 || !strings.Contains(strings.Join(p, "\n"), e.Chip+" no longer records "+e.Field) {
			t.Errorf("deleting %s/%s was not caught: %v", e.Chip, e.Field, p)
		}
		changed := edit(e.Chip, func(fp map[string]any) { fp[e.Field] = "datasheet" })
		if e.Kind != vocab.FieldDatasheet {
			if p := exceptionProblems(changed, committedExceptions); len(p) == 0 {
				t.Errorf("relabelling %s/%s as datasheet was not caught", e.Chip, e.Field)
			}
		}
		if e.Kind == vocab.FieldMeasured {
			elsewhere := edit(e.Chip, func(fp map[string]any) { fp[e.Field] = "measured:Some Other Device" })
			if p := exceptionProblems(elsewhere, committedExceptions); len(p) == 0 {
				t.Errorf("moving %s/%s to another device was not caught", e.Chip, e.Field)
			}
		}
	}
	// An exception recorded but not listed is as much a difference as one listed and lost.
	extra := edit("l40s", func(fp map[string]any) {})
	var doc map[string]any
	if err := yaml.Unmarshal(extra["l40s"], &doc); err != nil {
		t.Fatal(err)
	}
	doc["FieldProvenance"] = map[string]any{"SMCount": "published"}
	raw, _ := yaml.Marshal(doc)
	extra["l40s"] = raw
	if p := exceptionProblems(extra, committedExceptions); len(p) == 0 {
		t.Error("an exception that committedExceptions does not list was not caught")
	}
}

// committedSMCounts is the SM count of each committed chip, and what it was read from. A count
// is a device report, so a chip does not take another's: GB200 carries B200's two dies, and an
// earlier version of its entry carried B200's 148 over on that assumption, where the device
// reports 152. Changing a count means citing the report that shows it, in the chip's
// _comment_sm, and changing it here.
//
//	a100-80, a100-sxm   108  NVIDIA A100 whitepaper, Figure 6; TensorRT cookbook dump, A100-SXM4-80GB
//	b200                148  NVIDIA blog on CUDA MPS; TensorRT cookbook dump, NVIDIA B200
//	b300                148  TensorRT cookbook dump, NVIDIA B300 SXM6 AC
//	gb200-nvl72         152  Triton CLC tutorial output "Device: NVIDIA GB200, SMs: 152"; FlashInfer benchmark table
//	gb300               152  FlashInfer SM103 dispatch-plan tests, "Measured SM counts ... GB300 152"
//	h100, h200          132  TensorRT cookbook dump, NVIDIA H100 and H200
//	l40s                142  TensorRT cookbook dump, NVIDIA L40S
var committedSMCounts = map[string]int{
	"a100-80": 108, "a100-sxm": 108, "b200": 148, "b300": 148, "gb200-nvl72": 152,
	"gb300": 152, "h100": 132, "h200": 132, "l40s": 142,
}

func TestCommittedSMCountsAreTheDeviceReports(t *testing.T) {
	files := readCatalogFiles(t, "hardware")
	if len(files) != len(committedSMCounts) {
		t.Errorf("%d chips committed, %d SM counts pinned", len(files), len(committedSMCounts))
	}
	for stem, raw := range files {
		var c Chip
		if err := yaml.Unmarshal(raw, &c); err != nil {
			t.Fatalf("%s: %v", stem, err)
		}
		if want, ok := committedSMCounts[stem]; !ok || c.SMCount != want {
			t.Errorf("%s: SMCount %d; the report cited in its _comment_sm shows %d", stem, c.SMCount, want)
		}
	}
	// GB200 is not B200: the shared-die assumption that once supplied its count was wrong.
	if committedSMCounts["gb200-nvl72"] == committedSMCounts["b200"] {
		t.Error("GB200 takes B200's SM count; the device reports its own")
	}
}

// Every fact a committed chip or fabric states has a source: the one it records, or the one its
// file's Provenance names. The check reads each stated fact through SourceOf, so a file that
// inherits a word that is no source fails here as well as in Validate.
func TestEveryCommittedFactHasASource(t *testing.T) {
	read := func(dir string, load func(stem string, raw []byte) (stated map[string]bool, source func(string) (vocab.FieldSource, error), err error)) {
		for stem, raw := range readCatalogFiles(t, dir) {
			stated, source, err := load(stem, raw)
			if err != nil {
				t.Errorf("%s/%s: %v", dir, stem, err)
				continue
			}
			n := 0
			for field, isStated := range stated {
				if !isStated {
					continue
				}
				n++
				if _, err := source(field); err != nil {
					t.Errorf("%s/%s: %s has no source: %v", dir, stem, field, err)
				}
			}
			if n < 1 {
				t.Errorf("%s/%s: only %d stated facts found; the reading is wrong", dir, stem, n)
			}
		}
	}
	read("hardware", func(stem string, raw []byte) (map[string]bool, func(string) (vocab.FieldSource, error), error) {
		var c Chip
		if err := yaml.Unmarshal(raw, &c); err != nil {
			return nil, nil, err
		}
		return statedFields(reflect.ValueOf(c)), c.SourceOf, nil
	})
	read("networks", func(stem string, raw []byte) (map[string]bool, func(string) (vocab.FieldSource, error), error) {
		var f Fabric
		if err := yaml.Unmarshal(raw, &f); err != nil {
			return nil, nil, err
		}
		return statedFields(reflect.ValueOf(f)), f.SourceOf, nil
	})
}

// commentsMention reports whether a _comment* value of the chip file names the device. The record
// itself names the device in FieldProvenance, so a search of the whole file would find it there and
// prove nothing; the comment is the prose the record has to agree with.
func commentsMention(raw []byte, device string) bool {
	var doc map[string]any
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return false
	}
	for k, v := range doc {
		if s, ok := v.(string); ok && strings.HasPrefix(k, "_comment") && strings.Contains(s, device) {
			return true
		}
	}
	return false
}

// A measured figure whose device no comment names is found: the check reads the comments, not the
// FieldProvenance line that carries the device itself.
func TestCommentsMentionIgnoresTheRecordItself(t *testing.T) {
	raw := []byte("_comment_memory: \"read on NVIDIA H100 SXM5 80GB HBM3\"\nFieldProvenance:\n  MemoryGiB: \"measured:Totally Unrelated Device\"\n")
	if commentsMention(raw, "Totally Unrelated Device") {
		t.Error("the device was found in the record, not in a comment")
	}
	if !commentsMention(raw, "NVIDIA H100 SXM5 80GB HBM3") {
		t.Error("a device named in a comment was not found")
	}
}

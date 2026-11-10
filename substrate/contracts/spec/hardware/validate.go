package hardware

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// MaxDenseRatio is the largest ratio of one data type's dense peak to the next wider type's on
// a shipped part. NVIDIA's B300 datasheet gives NVFP4 at 3.11 times FP8; a sparse figure would be
// at least twice the dense one, so 3.5 admits every dense ratio and refuses a sparse number.
const MaxDenseRatio = 3.5

// Validate performs field-level validation of a chip: required positives,
// recognized provenance, and internal consistency between the packaging fields.
func (c *Chip) Validate() *validate.Problems {
	p := &validate.Problems{}
	if c.Name == "" {
		p.Field("name", "required")
	}
	if !c.Provenance.Valid() {
		p.Field("Provenance", "%q is not a recognized provenance", c.Provenance)
	}
	checkFieldProvenance(p, c.Provenance, c.FieldProvenance, reflect.ValueOf(*c), chipFields)
	if c.ComputeCapability != "" {
		if _, ok := c.CapabilityMajor(); !ok {
			p.Field("ComputeCapability", "%q must be major.minor, such as 9.0 or 10.3", c.ComputeCapability)
		}
	}
	// Every numeric field must be finite before its magnitude check runs, so a NaN or
	// Inf is caught rather than slipping past a comparison that is false either way. Each
	// FiniteField reports whether the field is finite; the magnitude and relational
	// checks below are gated on that, so a non-finite field is reported once — at the
	// field that is wrong — rather than also tripping its positivity check (a -Inf) or a
	// relational check that would misname a different field (a +Inf BF16 peak making the
	// FP8-vs-BF16 check complain about FP8).
	bf16Finite := p.FiniteField("TFlopsPeak", c.BF16Peak)
	fp8Finite := p.FiniteField("TFlopsFP8", c.FP8Peak)
	nvfp4Finite := p.FiniteField("TFlopsNVFP4", c.NVFP4Peak)
	bwFinite := p.FiniteField("BwPeakTBs", c.MemoryBandwidthTBs)
	memFinite := p.FiniteField("MemoryGiB", c.MemoryGiB)
	intraNodeFinite := p.FiniteField("IntraNodeBwGBps", c.IntraNodeBwGBps)
	intraRackFinite := p.FiniteField("IntraRackBwGBps", c.IntraRackBwGBps)
	if bf16Finite && c.BF16Peak <= 0 {
		p.Field("TFlopsPeak", "must be positive")
	}
	if bwFinite && c.MemoryBandwidthTBs <= 0 {
		p.Field("BwPeakTBs", "must be positive")
	}
	if memFinite && c.MemoryGiB <= 0 {
		p.Field("MemoryGiB", "must be positive")
	}
	if intraNodeFinite && c.IntraNodeBwGBps <= 0 {
		p.Field("IntraNodeBwGBps", "must be positive")
	}
	if fp8Finite && c.FP8Peak < 0 {
		p.Field("TFlopsFP8", "must not be negative; omit it on parts without native FP8")
	}
	if nvfp4Finite && c.NVFP4Peak < 0 {
		p.Field("TFlopsNVFP4", "must not be negative; omit it where the format is emulated")
	}
	// FP8 at or below BF16 would mean the narrower format buys nothing, which is not
	// a property any shipped part has. It is far more likely a transcription slip. Only
	// meaningful when both peaks are finite — a non-finite one is already reported, and
	// comparing against it would misattribute the fault to FP8.
	if fp8Finite && bf16Finite && c.FP8Peak > 0 && c.FP8Peak <= c.BF16Peak {
		p.Field("TFlopsFP8",
			"FP8 peak %.1f does not exceed BF16 peak %.1f; check the transcription",
			c.FP8Peak, c.BF16Peak)
	}
	// The peaks are dense figures (design SUB-16). A vendor datasheet quotes a dense and a sparse
	// figure side by side, and the sparse one is twice the dense one, so a sparse figure entered
	// as dense doubles the ratio between data types. Each narrower format runs at up to
	// MaxDenseRatio times the wider one on every shipped part (FP8 over BF16 is 2.0 to 2.03,
	// NVFP4 over FP8 is 2.0 to 3.11), so a ratio above it is a sparse figure or a slip.
	if fp8Finite && bf16Finite && c.FP8Peak > 0 && c.FP8Peak/c.BF16Peak > MaxDenseRatio {
		p.Field("TFlopsFP8", "FP8 peak %.1f is %.2fx the BF16 peak %.1f; dense FP8 is about twice dense BF16, so this looks like the sparse figure (at most %.1fx)",
			c.FP8Peak, c.FP8Peak/c.BF16Peak, c.BF16Peak, MaxDenseRatio)
	}
	if fp8Finite && nvfp4Finite && c.FP8Peak > 0 && c.NVFP4Peak > 0 {
		if c.NVFP4Peak <= c.FP8Peak {
			p.Field("TFlopsNVFP4", "NVFP4 peak %.1f does not exceed FP8 peak %.1f; the narrower format buys nothing, which no shipped part shows", c.NVFP4Peak, c.FP8Peak)
		} else if c.NVFP4Peak/c.FP8Peak > MaxDenseRatio {
			p.Field("TFlopsNVFP4", "NVFP4 peak %.1f is %.2fx the FP8 peak %.1f; this looks like the sparse figure (at most %.1fx)",
				c.NVFP4Peak, c.NVFP4Peak/c.FP8Peak, c.FP8Peak, MaxDenseRatio)
		}
	}
	// SMCount is required and positive. The field has no omitempty, so a missing count
	// decodes to zero and is rejected here alongside a negative one: every chip must
	// declare how many SMs it ships, since a model sizes a withheld-SM derate as a
	// fraction of it.
	if c.SMCount <= 0 {
		p.Field("SMCount", "must be positive; every chip must declare its SM count")
	}
	// Packaging counts are zero when unstated; a negative one is a mistake that the scenario
	// packaging would otherwise read as unset.
	if c.GPUsPerNode < 0 {
		p.Field("GPUsPerNode", "%d is negative; leave it out when the chip has no node size", c.GPUsPerNode)
	}
	if c.GPUsPerRack < 0 {
		p.Field("GPUsPerRack", "%d is negative; leave it out when the chip has no rack size", c.GPUsPerRack)
	}
	if c.GPUsPerRack > 0 && c.GPUsPerNode > 0 && c.GPUsPerRack%c.GPUsPerNode != 0 {
		p.Field("GPUsPerRack", "%d is not a multiple of GPUsPerNode %d",
			c.GPUsPerRack, c.GPUsPerNode)
	}
	if intraRackFinite && c.IntraRackBwGBps > 0 && c.GPUsPerRack == 0 {
		p.Field("IntraRackBwGBps",
			"an intra-rack bandwidth without GPUsPerRack describes a tier with no extent")
	}
	return p
}

// Validate performs field-level validation of a fabric.
func (f *Fabric) Validate() *validate.Problems {
	p := &validate.Problems{}
	if f.Name == "" {
		p.Field("name", "required")
	}
	if !f.Provenance.Valid() {
		p.Field("Provenance", "%q is not a recognized provenance", f.Provenance)
	}
	checkFieldProvenance(p, f.Provenance, f.FieldProvenance, reflect.ValueOf(*f), fabricFields)
	if p.FiniteField("InterNodeBwGBps", f.InterNodeBwGBps) && f.InterNodeBwGBps <= 0 {
		p.Field("InterNodeBwGBps", "must be positive")
	}
	return p
}

// Validate performs field-level validation of a storage device class.
func (d *StorageDevice) Validate() *validate.Problems {
	p := &validate.Problems{}
	if d.Name == "" {
		p.Field("name", "required")
	}
	if p.FiniteField("read_bandwidth_mb_s", d.ReadBandwidthMBs) && d.ReadBandwidthMBs <= 0 {
		p.Field("read_bandwidth_mb_s", "must be positive")
	}
	if p.FiniteField("write_bandwidth_mb_s", d.WriteBandwidthMBs) && d.WriteBandwidthMBs <= 0 {
		p.Field("write_bandwidth_mb_s", "must be positive")
	}
	// Zero base latency would make an arbitrarily small transfer free, which no
	// device is. It is the term that dominates small transfers.
	if p.FiniteField("base_latency_us", d.BaseLatencyUs) && d.BaseLatencyUs <= 0 {
		p.Field("base_latency_us", "must be positive; it dominates small transfers")
	}
	return p
}

// IntraToInterRatio returns the bandwidth ratio a cross-node collective pays,
// given a chip and the fabric it is paired with. It is a method on neither type
// alone because the ratio is a property of the pairing, which is the reason the two
// schemas are separate.
func IntraToInterRatio(c Chip, f Fabric) float64 {
	if f.InterNodeBwGBps <= 0 {
		return 1
	}
	return c.IntraNodeBwGBps / f.InterNodeBwGBps
}

// chipFields and fabricFields are the facts a FieldProvenance entry may name: every catalog
// fact of the type, by its wire name. Provenance itself is not one.
var chipFields = fieldNames(Chip{})
var fabricFields = fieldNames(Fabric{})

// SourceOf returns the source of one of the chip's catalog facts, by its wire name: the entry
// FieldProvenance records for it, or, where it records none, the source the file's Provenance
// names for every field it leaves out (SUB-18). It is an error for a field the type does not
// have, and for a file whose Provenance names no source and whose FieldProvenance is silent on
// the field.
func (c *Chip) SourceOf(field string) (vocab.FieldSource, error) {
	return sourceOf(c.Provenance, c.FieldProvenance, chipFields, field)
}

// SourceOf is Chip.SourceOf for a fabric.
func (f *Fabric) SourceOf(field string) (vocab.FieldSource, error) {
	return sourceOf(f.Provenance, f.FieldProvenance, fabricFields, field)
}

func sourceOf(prov vocab.Provenance, m map[string]string, fields map[string]bool, field string) (vocab.FieldSource, error) {
	if !fields[field] {
		return vocab.FieldSource{}, fmt.Errorf("%q is no catalog fact of this type", field)
	}
	if s, ok := m[field]; ok {
		return vocab.ParseFieldSource(s)
	}
	return prov.FieldSource()
}

// checkFieldProvenance refuses a per-field source that names no field of the type, or that is
// not datasheet, published or measured:<device> (SUB-18). A fitted or assumed value is refused
// with the reason: it is a coefficient and belongs in the coefficient registry.
//
// Every stated fact has a source, whether the file records one for it or leaves it to the
// file's Provenance, so the second case is checked too: a file whose Provenance is not itself a
// source (derived) must record one for each fact it states. A source recorded for a fact the
// file does not state is refused, because it describes nothing. A boolean is not judged either
// way, because false cannot be told from unstated.
func checkFieldProvenance(p *validate.Problems, prov vocab.Provenance, m map[string]string, v reflect.Value, fields map[string]bool) {
	stated := statedFields(v)
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		if !fields[k] {
			p.Field("FieldProvenance."+k, "names no catalog fact of this type; the facts are %s", strings.Join(sortedKeys(fields), ", "))
			continue
		}
		if _, err := vocab.ParseFieldSource(m[k]); err != nil {
			p.Field("FieldProvenance."+k, "%v", err)
			continue
		}
		if isStated, judged := stated[k]; judged && !isStated {
			p.Field("FieldProvenance."+k, "records a source for a fact the file does not state")
		}
	}
	if !prov.Valid() {
		return // reported once, as an unrecognized Provenance
	}
	for _, k := range sortedKeys(fields) {
		if _, recorded := m[k]; recorded || !stated[k] {
			continue
		}
		if _, err := prov.FieldSource(); err != nil {
			p.Field("FieldProvenance."+k, "%v", err)
		}
	}
}

// statedFields maps the wire name of each catalog fact the value states (a non-zero number or
// a non-empty string) to true, and each it leaves out to false. A boolean is absent from the
// map: it is not judged.
func statedFields(v reflect.Value) map[string]bool {
	out := map[string]bool{}
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		name, _, _ := strings.Cut(t.Field(i).Tag.Get("yaml"), ",")
		if name == "" || name == "-" || name == "Provenance" || name == "FieldProvenance" {
			continue
		}
		f := v.Field(i)
		if f.Kind() == reflect.Bool {
			continue
		}
		out[name] = !f.IsZero()
	}
	return out
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// fieldNames returns the yaml wire names of a struct's catalog facts: every tagged field except
// the identity and the provenance records themselves.
func fieldNames(v any) map[string]bool {
	out := map[string]bool{}
	t := reflect.TypeOf(v)
	for i := 0; i < t.NumField(); i++ {
		name, _, _ := strings.Cut(t.Field(i).Tag.Get("yaml"), ",")
		if name == "" || name == "-" || name == "Provenance" || name == "FieldProvenance" {
			continue
		}
		out[name] = true
	}
	return out
}

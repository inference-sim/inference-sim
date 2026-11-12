package coefficient

import (
	"fmt"
	"math"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Validate performs field-level validation of a coefficient set: the vocabulary
// terms are recognized, the required fields are present, and the evidence fields
// are internally consistent.
//
// What it does not do is judge whether a value is right, or whether a set is the
// right one for a given cost model. The first needs a measurement; the second is a
// scenario-level question.
func (s *Set) Validate() *validate.Problems {
	p := &validate.Problems{}
	if s.Kind != "CoefficientSet" {
		p.Field("kind", "must be %q, got %q", "CoefficientSet", s.Kind)
	}
	if s.Name == "" {
		p.Field("name", "required: a set's identity is its name")
	}
	if len(s.Coefficients) == 0 {
		p.Field("coefficients", "a set with no entries carries nothing")
	}
	// An entry's identity is its name, its scope AND its method, not its name alone
	// (design §3.3: one coefficient is identified by its set, its name, its scope and
	// its method).
	//
	// A set covering several parts carries one entry per (name, scope): the GEMM
	// efficiency asymptote is a different number on H100 than on H200, and both are
	// called gemm_eps_max_bf16 because a resolver selects between them by scope.
	// Keying on the name alone would forbid that shape, and pushing the part into the
	// name instead would duplicate in the name what the scope already states —
	// leaving a resolver no way to tell that two entries are the same quantity
	// measured on different hardware.
	//
	// The kernel implementations an entry is valid for are part of its identity too (see
	// Entry.Identity): two builds of one kernel have two values.
	//
	// Two entries sharing a name, a scope, a method AND a kernel key are still an error: a resolver
	// would keep whichever came last, so the file would mean something other than it says.
	// The same name and scope under different methods is a different coefficient: rank is
	// a function of method (VAL-4), so a measured and an assumed value of one quantity may
	// coexist. Until rank selection lands (M2) the cost model's resolver refuses a lookup
	// that finds both, rather than letting their order decide.
	//
	// This is a deliberate, documented LOOSENING of blis-registry's own validator, which
	// keys duplicate identity on the name ALONE and so rejects one set holding
	// gemm_eps_max_bf16 at both {hardware: h100} and {hardware: h200}. The (name, scope,
	// method) rule is the more expressive one and is authoritative once the registry adopts this
	// validator as its single gate (blis-registry#29); until then a set this accepts could
	// be rejected by the registry's Python gate, so the loosening is a contract change, not
	// a like-for-like migration.
	seen := map[string]bool{}
	for i, e := range s.Coefficients {
		at := fmt.Sprintf("coefficients[%d]", i)
		if e.Name == "" {
			p.Field(at+".name", "required")
		} else {
			key := e.Identity()
			if seen[key] {
				p.Field(at+".name",
					"duplicate entry name %q at the same scope (%s), method %q and kernel implementations; "+
						"two entries with one name, one scope, one method and one kernel key cannot both be resolved",
					e.Name, e.Scope.Describe(), e.Method)
			}
			seen[key] = true
		}
		e.validate(p, at)
	}
	return p
}

func (e Entry) validate(p *validate.Problems, at string) {
	// A coefficient feeds the latency model directly, so a non-finite value poisons it
	// exactly as a non-finite hardware figure poisons the cost model. The registry
	// loads these through this schema, so the check belongs here. A NaN would also
	// escape the interval bounds check below (NaN compares false to both < and >), so
	// gate the not_charged and containment checks on finiteness to avoid a misleading
	// second complaint about a value already reported non-finite.
	valueFinite := p.FiniteField(at+".value", e.Value)
	if !e.Units.Valid() {
		p.Field(at+".units", "%q is not one of %v", e.Units, vocab.AllUnits())
	}
	// The unit must be the one the formula that reads this family requires (SUB-16): a value in
	// another unit would be multiplied by a size or a count and price wrongly without an error.
	if f, ok := RequiredUnit(e.Name); ok && e.Units.Valid() && e.Units != f.Unit {
		p.Field(at+".units", "%q has units %q, but the formula that reads it requires %q (%s)",
			e.Name, e.Units, f.Unit, f.Why)
	}
	if !e.Method.Valid() {
		p.Field(at+".method", "%q is not one of %v", e.Method, vocab.AllMethods())
	}
	// A scope that states nothing is not "holds everywhere": it is "applicability
	// unknown", and a coefficient whose applicability is unknown cannot be applied.
	if e.Scope.Empty() {
		p.Field(at+".scope",
			"at least one dimension is required; an empty scope states no applicability rather than universal applicability")
	}
	// A fabric names a catalog networks/ identity, a bare file stem. A blank entry names
	// nothing; a path or a .yaml suffix is a reference to a file rather than its identity
	// and would never match a deployment's fabric; and a repeated name is a copy-paste
	// that changes nothing but reads as two links.
	seenFabric := map[string]bool{}
	for j, f := range e.Scope.Fabric {
		fat := fmt.Sprintf("%s.scope.fabric[%d]", at, j)
		switch {
		case strings.TrimSpace(f) == "":
			p.Field(fat, "names a catalog networks/ fabric and must not be empty")
		case strings.ContainsAny(f, `/\`) || strings.HasSuffix(strings.ToLower(f), ".yaml"):
			p.Field(fat, "%q is a path; a fabric is named by its catalog identity, the stem of networks/<name>.yaml", f)
		case seenFabric[strings.ToLower(f)]:
			p.Field(fat, "duplicate fabric %q", f)
		}
		seenFabric[strings.ToLower(f)] = true
	}
	// value 0 is reserved for a deliberately not-charged term. A zero under any other
	// method reads as "free" when it almost always means "unfilled", so require the method
	// to declare the intent; the converse (not_charged must be zero) is just below. Both
	// gate on finiteness so a non-finite value is reported once, not also here.
	if valueFinite && e.Value == 0 && e.Method != vocab.MethodNotCharged {
		p.Field(at+".value",
			"value 0 requires method %q (a priced-at-zero term), got method %q",
			vocab.MethodNotCharged, e.Method)
	}
	if valueFinite && e.Method == vocab.MethodNotCharged && e.Value != 0 {
		p.Field(at+".value",
			"method not_charged declares a deliberate zero, but value is %v", e.Value)
	}
	// fitted marks a value obtained by fitting a curve, either to its own measured term or,
	// for calibrated, through the cost model to end-to-end cases, so it is coherent only
	// with those two methods.
	if e.Fitted && e.Method != vocab.MethodMeasured && e.Method != vocab.MethodCalibrated {
		p.Field(at+".fitted",
			"fitted: true requires method %q or %q, got method %q",
			vocab.MethodMeasured, vocab.MethodCalibrated, e.Method)
	}
	// rate_basis (SUB-5): a closed word, meaningful only for a rate in bytes per microsecond,
	// and a work rate must say why it is not a memory rate.
	if e.RateBasis != "" {
		switch {
		case !e.RateBasis.Valid():
			p.Field(at+".rate_basis", "%q is not one of %v", e.RateBasis, vocab.AllRateBases())
		case e.Units != vocab.UnitBytesPerMicrosecond:
			p.Field(at+".rate_basis", "applies only to units %q, got units %q",
				vocab.UnitBytesPerMicrosecond, e.Units)
		case e.RateBasis == vocab.RateBasisWork && strings.TrimSpace(e.Rationale) == "":
			p.Field(at+".rationale", "rate_basis work requires a rationale that says why the rate is not a memory rate")
		}
	}
	e.validateEvidenceRecords(p, at)
	// Companion fields required by method. Gated on a known method so an unknown one is
	// reported once (as an unknown method) rather than also as a missing companion.
	if e.Method.Valid() {
		// Every method but measured rests on a judgement a reader must be able to weigh,
		// so it must carry its reasoning. (A measured value cites a source instead.)
		if e.Method != vocab.MethodMeasured && e.Rationale == "" {
			p.Field(at+".rationale", "method %q requires a rationale", e.Method)
		}
		// A value drawn from the literature or a datasheet must cite where. measured is
		// evidenced too, but the registry carries pre-existing uncited measured entries,
		// so a source is not required there — matching blis-registry's own validator.
		if (e.Method == vocab.MethodLiterature || e.Method == vocab.MethodVendorSpec) &&
			len(e.Sources) == 0 {
			p.Field(at+".sources", "method %q requires at least one source", e.Method)
		}
		// VAL-1: each method needs the sources that make it checkable. A transferred value
		// names the coefficient it came from and the rule that moved it; a roofline value
		// cites the catalog facts it was computed from.
		if e.Method == vocab.MethodTransferred {
			if strings.TrimSpace(e.TransferredFrom) == "" {
				p.Field(at+".transferred_from", "method transferred requires transferred_from")
			}
			if strings.TrimSpace(e.TransferRule) == "" {
				p.Field(at+".transfer_rule", "method transferred requires transfer_rule, the stated rule that moved the value")
			}
		}
		if e.Method == vocab.MethodRoofline && len(e.Sources) == 0 {
			p.Field(at+".sources", "method roofline requires at least one source naming the catalog facts it was computed from")
		}
		if e.Method != vocab.MethodTransferred && (e.TransferredFrom != "" || e.TransferRule != "") {
			p.Field(at+".transferred_from", "transferred_from and transfer_rule apply only to method transferred, got method %q", e.Method)
		}
		// A calibrated value is only as good as the cases it was fitted to and the ones it
		// was held out against, so both must be on the entry.
		if e.Method == vocab.MethodCalibrated {
			if !e.citesKind(vocab.SourceMeasuredCase) {
				p.Field(at+".sources", "method calibrated requires at least one source of kind %q",
					vocab.SourceMeasuredCase)
			}
			if len(e.Validated) == 0 {
				p.Field(at+".validated",
					"method calibrated requires the held-out metrics the value was checked against")
			}
		}
	}
	// Intentional minor divergence from the Python validator: it also rejects a present-
	// but-blank optional string (supersedes, or copied_from on a non-copied method). Go
	// stores these as plain strings, so "" cannot be told from absent without presence
	// tracking, and no committed set carries a blank one — so this is left unchecked.
	for j, src := range e.Sources {
		sat := fmt.Sprintf("%s.sources[%d]", at, j)
		if !src.Kind.Valid() {
			p.Field(sat+".kind", "%q is not one of %v", src.Kind, vocab.AllSourceKinds())
		}
		if !src.Role.Valid() {
			p.Field(sat+".role", "%q is not one of %v", src.Role, vocab.AllSourceRoles())
		}
		if src.Cite == "" {
			p.Field(sat+".cite", "required")
		}
	}
	if e.CI95 != nil {
		// A non-finite interval endpoint would silently defeat the ordering and
		// containment checks below (every comparison with NaN is false), so reject it
		// first, as the value itself is, and gate those checks on finiteness so a
		// non-finite endpoint is reported once rather than also as a spurious ordering
		// or containment failure.
		lowFinite := p.FiniteField(at+".ci95.low", e.CI95.Low)
		highFinite := p.FiniteField(at+".ci95.high", e.CI95.High)
		if lowFinite && highFinite && e.CI95.Low > e.CI95.High {
			p.Field(at+".ci95", "low %v exceeds high %v", e.CI95.Low, e.CI95.High)
		}
		if valueFinite && lowFinite && highFinite &&
			(e.Value < e.CI95.Low || e.Value > e.CI95.High) {
			p.Field(at+".value", "%v lies outside its own interval [%v, %v]",
				e.Value, e.CI95.Low, e.CI95.High)
		}
	}
}

func (e Entry) citesKind(k vocab.SourceKind) bool {
	for _, s := range e.Sources {
		if s.Kind == k {
			return true
		}
	}
	return false
}

// validateEvidenceRecords checks the optional records of design D5, VAL-8, VAL-9 and VAL-10:
// the kernel implementations an entry is valid for, its measurement boundary, and the range
// its fit covered. Each is checked whenever it is present; the ratchet over the entries that
// state none is a registry test, because the fields arrive with the fitting scripts.
func (e Entry) validateEvidenceRecords(p *validate.Problems, at string) {
	ValidateKernelImpls(p, at+".kernel_impls", e.KernelImpls)
	ValidateKernelImpls(p, at+".measured_on", e.MeasuredOn)
	if len(e.KernelImpls) > 0 && !MayStateKernelImpls(e.Method) {
		// A vendor_spec, roofline, assumed or not_charged value says nothing about a kernel; a
		// list on it would key a lookup to something the value does not depend on (design 6.2).
		p.Field(at+".kernel_impls", "method %q states nothing about a kernel implementation, so it states none; "+
			"it is what a lookup falls back to when no entry matches the one that runs", e.Method)
	}
	if len(e.KernelImpls) > 0 && PrimitiveOf(e.Name) == "" && e.Name != "" {
		p.Field(at+".kernel_impls", "%q prices no lowered kernel (it binds to no primitive, coefficient.PrimitiveOf), "+
			"so a kernel key on it could never match", e.Name)
	}
	if len(e.MeasuredOn) > 0 && !MayStateMeasuredOn(e.Method) {
		p.Field(at+".measured_on", "method %q is measured on the implementations kernel_impls lists (D5), "+
			"so it states no other; only a transferred value is measured on one kernel and used for another", e.Method)
	}
	if e.Method == vocab.MethodTransferred && len(e.KernelImpls) > 0 && len(e.MeasuredOn) == 0 {
		p.Field(at+".measured_on", "a transferred value that names the implementations it is valid for must "+
			"name the ones it was measured on, or the transfer cannot be checked (D5)")
	}
	if e.Measures != nil {
		m := e.Measures
		if !m.Time.Valid() {
			p.Field(at+".measures.time", "%q is not one of %v", m.Time, vocab.AllMeasurementTimes())
		}
		if !m.Execution.Valid() {
			p.Field(at+".measures.execution", "%q is not one of %v", m.Execution, vocab.AllMeasurementExecutions())
		}
		if strings.TrimSpace(m.Harness) == "" {
			p.Field(at+".measures.harness", "required: the harness that collected the data")
		}
	}
	ValidateFitRange(p, at+".fit_range", e.FitRange)
	if len(e.FitRange) > 0 && !MayStateFitRange(e.Method) {
		p.Field(at+".fit_range", "method %q was not fitted to data, so it states no fit range", e.Method)
	}
	if e.Measures != nil && e.Method != vocab.MethodMeasured && e.Method != vocab.MethodCalibrated {
		p.Field(at+".measures", "method %q has no measurement boundary", e.Method)
	}
}

// MayStateFitRange reports whether an entry of method m may record the range its fit covered.
// The data a measured, calibrated or transferred value was fitted to is part of what it claims:
// a transferred value still has the rows it came from, and a lookup outside them is
// extrapolated. A fit range on any other method would describe data that did not produce the
// value. Both validators call this; no other place decides it. A 95% interval (ci95) has no
// method rule: any method may state one.
func MayStateFitRange(m vocab.Method) bool {
	return m == vocab.MethodMeasured || m == vocab.MethodCalibrated || m == vocab.MethodTransferred
}

// ValidateFitRange checks a fit range (design VAL-10): each bound names a known dimension
// once, with finite ends and low not above high. at names the list in a message.
func ValidateFitRange(p *validate.Problems, at string, bounds []Bound) {
	seenDim := map[string]bool{}
	for i, b := range bounds {
		bat := fmt.Sprintf("%s[%d]", at, i)
		if !vocab.FitDim(b.Dim).Valid() {
			p.Field(bat+".dim", "%q is not one of %v", b.Dim, vocab.AllFitDims())
		}
		if seenDim[b.Dim] {
			p.Field(bat+".dim", "duplicate dimension %q", b.Dim)
		}
		seenDim[b.Dim] = true
		lowFinite := p.FiniteField(bat+".low", b.Low)
		highFinite := p.FiniteField(bat+".high", b.High)
		if lowFinite && highFinite && b.Low > b.High {
			p.Field(bat, "low %v exceeds high %v", b.Low, b.High)
		}
	}
}

// Unidentified reports whether the entry's data do not determine it (design VAL-10): its
// 95% interval is unbounded, or wider than half its value. An unidentified entry keeps its
// method and its citations, and ranks with assumed for VAL-3. An entry with no interval is
// not unidentified; it has not said.
func (e Entry) Unidentified() bool {
	if e.CI95 == nil {
		return false
	}
	lo, hi := e.CI95.Low, e.CI95.High
	if math.IsInf(lo, 0) || math.IsInf(hi, 0) || math.IsNaN(lo) || math.IsNaN(hi) {
		return true
	}
	return hi-lo > 0.5*math.Abs(e.Value)
}

// Extrapolated reports whether a lookup at point lies outside the range the entry's fit
// covered (design VAL-10), on any dimension it recorded. in maps a dimension name to its
// value at the lookup; a recorded dimension the lookup does not supply is not judged.
// An entry that recorded no range is never extrapolated: it has not said.
func (e Entry) Extrapolated(in map[string]float64) bool {
	for _, b := range e.FitRange {
		if v, ok := in[b.Dim]; ok && (v < b.Low || v > b.High) {
			return true
		}
	}
	return false
}

// EffectiveRank is the rank VAL-3 reports for a price that used this entry: the method's
// rank, or transferred's when the lookup is extrapolated, or assumed's when the entry is
// unidentified, whichever is the least trusted of those that apply. not_charged has none.
func (e Entry) EffectiveRank(extrapolated bool) (rank int, ranked bool) {
	rank, ranked = e.Method.Rank()
	if !ranked {
		return 0, false
	}
	if extrapolated {
		if t, _ := vocab.MethodTransferred.Rank(); t > rank {
			rank = t
		}
	}
	if e.Unidentified() {
		if a, _ := vocab.MethodAssumed.Rank(); a > rank {
			rank = a
		}
	}
	return rank, true
}

func hexHash(h string) bool {
	if len(h) < 12 {
		return false
	}
	for _, c := range h {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// Package vocab holds the closed vocabularies every schema draws on: units,
// evidence methods, scope keys, provenance. They live here, once, because
// blis-catalog and blis-registry each validate against the same terms today from
// separate Python enums, and two copies of a vocabulary drift.
//
// A vocabulary is closed on purpose. An unrecognized unit or method is an error,
// not a pass-through: the distinction between a measured number and an assumed one
// is the property these documents exist to preserve, and it survives only if the
// words for it cannot be invented at the call site.
package vocab

import "fmt"

// Unit is the dimension of a coefficient's value. The set matches
// blis-registry's schema, plus three members the cost model requires and the
// registry does not yet carry: Tokens, SMCount and MicrosecondsPerToken.
//
// The registry documents the third gap itself. Its output_token_processing entry
// records a per-token quantity under UnitMicrosecondsPerRequest with the note
// that "the schema's units enum has no us_per_token member" — a missing unit
// rather than a per-request quantity.
type Unit string

const (
	UnitDimensionless          Unit = "dimensionless"
	UnitMicrosecondsPerLayer   Unit = "us_per_layer"
	UnitMicrosecondsPerRequest Unit = "us_per_request"
	UnitMicrosecondsPerStep    Unit = "us_per_step"
	UnitMicrosecondsPerHop     Unit = "us_per_hop"
	UnitBytesPerMicrosecond    Unit = "bytes_per_us"
	UnitBytesPerRank           Unit = "bytes_per_rank"
	UnitMicrosecondsPerLoad    Unit = "us_per_load"
	UnitMicrosecondsPerXfer    Unit = "us_per_transfer"

	// Required by the cost model, absent from blis-registry today. Each is a
	// schema addition upstream, tracked in the realization document's §11.1.
	UnitTokens               Unit = "tokens"       // M_half, a token count
	UnitSMCount              Unit = "sm_count"     // SMs withheld by a concurrent fetch
	UnitMicrosecondsPerToken Unit = "us_per_token" // per-emitted-token host work
)

var units = map[Unit]bool{
	UnitDimensionless: true, UnitMicrosecondsPerLayer: true,
	UnitMicrosecondsPerRequest: true, UnitMicrosecondsPerStep: true,
	UnitMicrosecondsPerHop: true, UnitBytesPerMicrosecond: true,
	UnitBytesPerRank: true, UnitMicrosecondsPerLoad: true,
	UnitMicrosecondsPerXfer: true, UnitTokens: true, UnitSMCount: true,
	UnitMicrosecondsPerToken: true,
}

// Valid reports whether u is a recognized unit.
func (u Unit) Valid() bool { return units[u] }

// Method records how a value was obtained. It is the field that keeps an estimate
// distinguishable from a measurement, so it is never inferred from context.
type Method string

const (
	// MethodMeasured: obtained from a measurement, whose source the entry cites.
	MethodMeasured Method = "measured"
	// MethodLiterature: taken from a published result not independently reproduced.
	MethodLiterature Method = "literature"
	// MethodVendorSpec: a datasheet figure. Nominal, not achieved.
	MethodVendorSpec Method = "vendor_spec"
	// MethodCopied: carried over from another scope, which copied_from names.
	MethodCopied Method = "copied"
	// MethodAssumed: a placeholder or an estimate arrived at by reasoning. The
	// reasoning belongs in Rationale; a careful estimate is still assumed.
	MethodAssumed Method = "assumed"
	// MethodNotCharged: deliberately zero, so a reader can tell "no cost" from
	// "cost unknown".
	MethodNotCharged Method = "not_charged"
	// MethodCalibrated: back-solved through the cost model from end-to-end measured
	// cases. It is not "derived", which ProvenanceDerived already means: computed from
	// other catalog facts. A calibrated entry cites its cases as SourceMeasuredCase and
	// records the held-out metrics it was checked against.
	MethodCalibrated Method = "calibrated"
)

var methods = map[Method]bool{
	MethodMeasured: true, MethodLiterature: true, MethodVendorSpec: true,
	MethodCopied: true, MethodAssumed: true, MethodNotCharged: true,
	MethodCalibrated: true,
}

// Valid reports whether m is a recognized method.
func (m Method) Valid() bool { return methods[m] }

// Evidenced reports whether m rests on an observation rather than a judgement.
// Callers use it to separate a calibrated deployment from an estimated one; it is
// not a quality ranking, since a vendor spec is evidenced but nominal.
func (m Method) Evidenced() bool {
	return m == MethodMeasured || m == MethodLiterature || m == MethodVendorSpec ||
		m == MethodCalibrated
}

// ScopeKey names a dimension a coefficient's validity is bounded along. The set
// is open by design upstream — adding a key is a schema addition, not a redesign —
// but a key this package does not know is an error rather than a silent pass.
//
// Deliberately absent: dtype and backend. Both vary a coefficient, and both ride
// in the entry NAME instead (gemm_eps_max_bf16, a2a_floor_deepep_low_latency),
// following the convention blis-registry established for its per-backend MoE
// dials: "the backend is identified by the entry NAME".
type ScopeKey string

const (
	ScopeHardware     ScopeKey = "hardware"
	ScopeTP           ScopeKey = "tp"
	ScopeEP           ScopeKey = "ep"
	ScopeNodesSpanned ScopeKey = "nodes_spanned"
	ScopeModel        ScopeKey = "model"
	// ScopeFabric bounds a coefficient to an inter-node interconnect class, named by its
	// catalog identity (networks/<name>: roce-200g, ib-400g). nodes_spanned says whether a
	// collective leaves a node; fabric says over which link, which is what separates two
	// clusters of comparable silicon on different networks. A link inside one chip's
	// NVLink domain, a rack included, is the chip's and is scoped by hardware.
	ScopeFabric ScopeKey = "fabric"
)

var scopeKeys = map[ScopeKey]bool{
	ScopeHardware: true, ScopeTP: true, ScopeEP: true,
	ScopeNodesSpanned: true, ScopeModel: true, ScopeFabric: true,
}

// Valid reports whether k is a recognized scope key.
func (k ScopeKey) Valid() bool { return scopeKeys[k] }

// Provenance records where a catalog fact came from. blis-catalog validates the
// same two terms.
type Provenance string

const (
	ProvenanceVendorSpec Provenance = "vendor_spec"
	ProvenanceDerived    Provenance = "derived"
)

var provenances = map[Provenance]bool{
	ProvenanceVendorSpec: true, ProvenanceDerived: true,
}

// Valid reports whether p is a recognized provenance.
func (p Provenance) Valid() bool { return provenances[p] }

// SourceKind and SourceRole describe one citation supporting a coefficient.
type SourceKind string

const (
	SourceDiscussion  SourceKind = "discussion"
	SourcePublication SourceKind = "publication"
	SourceDatasheet   SourceKind = "datasheet"
	SourceModel       SourceKind = "model"
	SourceVendorDoc   SourceKind = "vendor_doc"
	// SourceMeasuredCase cites a measured run in blis-catalog's measurements/, by its
	// name, that a calibrated entry was fitted against.
	SourceMeasuredCase SourceKind = "measured_case"
)

var sourceKinds = map[SourceKind]bool{
	SourceDiscussion: true, SourcePublication: true, SourceDatasheet: true,
	SourceModel: true, SourceVendorDoc: true, SourceMeasuredCase: true,
}

// Valid reports whether k is a recognized source kind.
func (k SourceKind) Valid() bool { return sourceKinds[k] }

type SourceRole string

const (
	RolePrimary    SourceRole = "primary"
	RoleSupporting SourceRole = "supporting"
	RoleUpperBound SourceRole = "upper_bound"
)

var sourceRoles = map[SourceRole]bool{
	RolePrimary: true, RoleSupporting: true, RoleUpperBound: true,
}

// Valid reports whether r is a recognized source role.
func (r SourceRole) Valid() bool { return sourceRoles[r] }

// Enumerate returns every member of a vocabulary, sorted, for error messages that
// tell a reader what was allowed rather than only what was rejected.
func Enumerate[T ~string](set map[T]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, string(k))
	}
	sortStrings(out)
	return out
}

// AllUnits, AllMethods and the rest expose each vocabulary for error text and for
// tests that assert a vocabulary has not silently grown.
func AllUnits() []string       { return Enumerate(units) }
func AllMethods() []string     { return Enumerate(methods) }
func AllScopeKeys() []string   { return Enumerate(scopeKeys) }
func AllProvenances() []string { return Enumerate(provenances) }
func AllSourceKinds() []string { return Enumerate(sourceKinds) }
func AllSourceRoles() []string { return Enumerate(sourceRoles) }

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// UnknownValueError reports a value outside a closed vocabulary, naming the
// allowed set so the message is actionable without opening this file.
type UnknownValueError struct {
	Field   string
	Value   string
	Allowed []string
}

func (e *UnknownValueError) Error() string {
	return fmt.Sprintf("%s: %q is not one of %v", e.Field, e.Value, e.Allowed)
}

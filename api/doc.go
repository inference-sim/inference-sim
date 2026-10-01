// Package api defines the envelope shared by every BLIS declarative document — the
// single versioned, self-describing input/output format introduced by epic #1855.
//
// # The envelope
//
// Every document carries [Version] as its apiVersion and a [Kind]. An INPUT document
// carries a spec; an OUTPUT document (a "…Result" kind) carries a result:
//
//	apiVersion: llm-d-perf-simulator/v1
//	kind: Run
//	spec: {}
//
// # Scope at this stage (API-1, #1856)
//
// The package is deliberately INERT: no CLI verb reads or writes a document, no flag
// changes, and `blis run` stdout is byte-for-byte unchanged. [Body] is an OPAQUE open
// object — the per-verb spec sections (deployment, latency, admission, …) and the result
// body (summary, conservation, kv, …), together with their uniform strict parsing, are
// defined by API-2 onward (#1857 …). The schema and the examples under examples/
// therefore constrain envelope STRUCTURE only: a correct apiVersion, a kind drawn from
// the enum, and spec present for input kinds / result present for output kinds.
//
// # The JSON Schema
//
// schema/llm-d-perf-simulator-v1.json is DERIVED from the Go types in this package by
// [GenerateSchema] and committed, so external tools can validate against a stable URL.
// Regenerate it after any change to the envelope types:
//
//	go generate ./api/...
//
// TestCommittedSchemaIsCurrent fails if the committed copy is stale, so a type change that
// was not regenerated is refused. Until ./api/... is in ci.yml's test matrix (#1866), the
// job that actually runs in CI is the root-package bridge in ../api_schema_gate_test.go,
// which mirrors that gate and runs this package's suite as a subprocess.
//
// This package depends only on the standard library and gopkg.in/yaml.v3; it imports no
// other BLIS package, which keeps the document format free of the simulator's dependency
// direction (cmd/ → sim/cluster/ → sim/).
package api

//go:generate go run ./schemagen -out schema/llm-d-perf-simulator-v1.json

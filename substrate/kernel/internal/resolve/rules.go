package resolve

import (
	"github.com/inference-sim/blis-substrate/contracts/rules"
)

// EngineRules is the version-specific behaviour a resolver needs: blis-schemas'
// rules.EngineRules, the interface a release's rules are consumed through.
//
// It is an interface rather than a set of constants because these facts belong to one
// engine release. A scenario names its version; the caller supplies the matching rules.
// Compiling them in would mean this package churns whenever an engine ships, and would
// silently misjudge a scenario pinned to an older release.
//
// An alias rather than a narrower interface of this package's own: a narrower one had no
// method for facts the kernel needs -- the release's async-compatible draft methods, its
// Engram default, its decode-context backend names, its mega-MoE backends -- so the kernel
// read them by asserting to *rules.Pack and quietly skipped them for any other rules value.
// The shared interface names every one, so a caller's rules are read through it alone.
type EngineRules = rules.EngineRules

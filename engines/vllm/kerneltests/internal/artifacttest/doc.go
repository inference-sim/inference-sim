// Moved from blis-latency-kernel/internal/artifacttest/doc.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// Package artifacttest holds the rule every test package uses to decide whether a failure
// to load a catalog or registry artifact is a skip or a failure.
//
// Its own package, rather than a helper inside internal/harness, for two reasons. It
// imports "testing", which has no business in a package the commands link; and the rule
// has to be reachable from the root package's tests, which cannot import harness at all
// (harness imports the root package, so that would close an import cycle). A leaf package
// is the only place all three test packages can share one copy.
//
// One copy matters here. The first attempt at this fix put the rule in the root package's
// own test file, which left cmd/shape and cmd/score still turning a corrupt vendored
// catalog into a green "ok" with their assertions silently skipped.
package artifacttest

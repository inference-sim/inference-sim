// Package artifacttest holds the rule every test package uses to decide whether a failure
// to load a catalog or registry artifact is a skip or a failure.
//
// It is its own leaf package because it imports "testing", which has no place in a package
// a command links, and because one copy of the rule must serve every test package that
// reads data. A second copy is how a corrupt catalog once read as a passing suite: one
// package failed on it while another skipped.
package artifacttest

package scripts_test

import (
	"fmt"
	"regexp"
	"testing"
)

// The Dockerfile builds the release image from a `golang:<version>-alpine` base, and go.mod pins
// the toolchain the module requires. Go refuses to build a module whose `go` directive is newer
// than the toolchain, so the two must name the SAME version: v0.12.0 and v0.13.0 were tagged with
// a `golang:1.24-alpine` base while go.mod required 1.27.2, so `docker build` failed and no image
// was ever published for those releases. Nothing in CI builds the image, so this guard is the only
// thing that can catch the drift before a release tag is cut.

// golangBaseRe matches the builder base image, capturing the Go version.
var golangBaseRe = regexp.MustCompile(`(?m)^FROM\s+golang:([0-9][^-\s]*)-alpine(?:\s|$)`)

// goDirectiveRe matches go.mod's `go` directive, capturing the required version.
var goDirectiveRe = regexp.MustCompile(`(?m)^go\s+(\S+)\s*$`)

// dockerfileGoVersion returns the Go version of the builder base image in dockerfile.
func dockerfileGoVersion(dockerfile string) (string, error) {
	m := golangBaseRe.FindStringSubmatch(dockerfile)
	if m == nil {
		return "", fmt.Errorf("no `FROM golang:<version>-alpine` builder stage found")
	}
	return m[1], nil
}

// goModVersion returns the version named by go.mod's `go` directive.
func goModVersion(gomod string) (string, error) {
	m := goDirectiveRe.FindStringSubmatch(gomod)
	if m == nil {
		return "", fmt.Errorf("no `go <version>` directive found")
	}
	return m[1], nil
}

// The real files must agree. This is the regression guard: bump go.mod without bumping the
// Dockerfile base image (or the reverse) and CI fails here instead of at release time.
func TestDockerfileGoVersionMatchesGoMod(t *testing.T) {
	want, err := goModVersion(readRepoFile(t, "go.mod"))
	if err != nil {
		t.Fatalf("go.mod: %v", err)
	}
	got, err := dockerfileGoVersion(readRepoFile(t, "Dockerfile"))
	if err != nil {
		t.Fatalf("Dockerfile: %v", err)
	}

	if got != want {
		t.Errorf("Dockerfile builder uses golang:%s-alpine but go.mod requires go %s — "+
			"bump the Dockerfile `FROM golang:<version>-alpine` line to match", got, want)
	}
}

// The parser must read the version the way the release build does: patch versions included, and
// only the builder stage (the runtime stage is plain alpine).
func TestDockerfileGoVersionParsesTheBuilderStage(t *testing.T) {
	const dockerfile = `# syntax=docker/dockerfile:1
FROM golang:1.27.2-alpine AS builder
RUN go build ./...
FROM alpine:3.19
`
	got, err := dockerfileGoVersion(dockerfile)
	if err != nil {
		t.Fatalf("dockerfileGoVersion: %v", err)
	}
	if got != "1.27.2" {
		t.Errorf("dockerfileGoVersion = %q, want 1.27.2", got)
	}
}

// A Dockerfile with no golang base is an error, not an empty string: silently returning "" would
// make the guard pass whenever the builder stage was renamed or replaced.
func TestDockerfileGoVersionRequiresAGolangBase(t *testing.T) {
	if got, err := dockerfileGoVersion("FROM alpine:3.19\n"); err == nil {
		t.Errorf("dockerfileGoVersion = %q, want an error for a Dockerfile with no golang base", got)
	}
}

// go.mod's directive is what `go build` enforces; the parser must pick the version and not, say,
// a comment or a require line that mentions a version.
func TestGoModVersionParsesTheGoDirective(t *testing.T) {
	const gomod = `module example.com/x

go 1.27.2

// toolchain notes for go1.99.0
require example.com/y v1.0.0
`
	got, err := goModVersion(gomod)
	if err != nil {
		t.Fatalf("goModVersion: %v", err)
	}
	if got != "1.27.2" {
		t.Errorf("goModVersion = %q, want 1.27.2", got)
	}
}

// go.mod with no directive must be an error for the same reason as above.
func TestGoModVersionRequiresTheDirective(t *testing.T) {
	if got, err := goModVersion("module example.com/x\n"); err == nil {
		t.Errorf("goModVersion = %q, want an error for go.mod with no go directive", got)
	}
}

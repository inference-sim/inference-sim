package scripts_test

import (
	"regexp"
	"strings"
	"testing"
)

// The Dockerfile does not state a Go version (X23). It takes the version as a required build
// argument, `ARG GO_VERSION`, and every caller passes the version that go.mod names. Go refuses
// to build a module whose `go` directive is newer than the toolchain, so a stale or missing
// version must fail the build rather than select an old image: v0.12.0 and v0.13.0 were tagged
// with a base image older than go.mod requires, and no image was published for them. Nothing in
// CI builds the image, so these checks are what catch the drift before a release tag is cut.

// goVersionArgRe matches the build argument declaration with no default value.
var goVersionArgRe = regexp.MustCompile(`(?m)^ARG\s+GO_VERSION\s*$`)

// golangBaseRe matches the builder base image and captures the version expression.
var golangBaseRe = regexp.MustCompile(`(?m)^FROM\s+golang:([^-\s]+)-alpine(?:\s|$)`)

// dockerfileTakesGoVersionArg reports whether dockerfile builds from golang:${GO_VERSION} with a
// GO_VERSION argument declared (before that FROM) and no default.
func dockerfileTakesGoVersionArg(dockerfile string) bool {
	m := golangBaseRe.FindStringSubmatchIndex(dockerfile)
	if m == nil {
		return false
	}
	if dockerfile[m[2]:m[3]] != "${GO_VERSION}" {
		return false
	}
	return goVersionArgRe.MatchString(dockerfile[:m[0]])
}

func TestDockerfileTakesTheGoVersionAsABuildArgument(t *testing.T) {
	if !dockerfileTakesGoVersionArg(readRepoFile(t, "Dockerfile")) {
		t.Error("Dockerfile must declare `ARG GO_VERSION` (no default) and build `FROM golang:${GO_VERSION}-alpine`")
	}
}

// The release workflow is the one place CI builds the image. It must read the version from go.mod
// and pass it as the GO_VERSION build argument.
func TestReleaseWorkflowPassesTheGoModVersion(t *testing.T) {
	wf := readRepoFile(t, ".github", "workflows", "release.yml")
	if !strings.Contains(wf, "s/^go //p' go.mod") {
		t.Error("release.yml must read the Go version from the go directive of go.mod")
	}
	if !strings.Contains(wf, "GO_VERSION=${{ steps.gomod.outputs.version }}") {
		t.Error("release.yml must pass GO_VERSION=<go.mod version> as a build-arg to the image build")
	}
}

func TestDockerfileGoVersionArgParser(t *testing.T) {
	cases := map[string]bool{
		"# syntax=docker/dockerfile:1\nARG GO_VERSION\nFROM golang:${GO_VERSION}-alpine AS builder\nFROM alpine:3.19\n": true,
		"ARG GO_VERSION=1.20\nFROM golang:${GO_VERSION}-alpine AS builder\n":                                            false, // default hides a missing argument
		"FROM golang:${GO_VERSION}-alpine AS builder\n":                                                                 false, // ARG not declared before FROM
		"ARG GO_VERSION\nFROM golang:1.20-alpine AS builder\n":                                                          false, // literal version
		"FROM alpine:3.19\n": false,
	}
	for in, want := range cases {
		if got := dockerfileTakesGoVersionArg(in); got != want {
			t.Errorf("dockerfileTakesGoVersionArg(%q) = %v, want %v", in, got, want)
		}
	}
}
